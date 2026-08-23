"""P5 step4 real LLM skeleton compiler.

This module implements the free-form half of decision 9 while preserving the
decision 7 static-v1 constraints. LLM calls are wrapped in read-only atom
contracts so Codex-style runners are steered away from executing tools or
reading/writing files while compiling a skeleton.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from atomic_agents.adapters import RunnerAdapter
from atomic_agents.linter import VALID_INPUT_FIELDS
from atomic_agents.models import AtomContract, AtomResult, Skeleton
from atomic_agents.templates import TEMPLATES
from atomic_agents.validation import ValidationError, validate_skeleton


_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_RAW_PREVIEW_CHARS = 500


class SkeletonParseError(Exception):
    """Raised when LLM output cannot be parsed into a JSON object."""

    def __init__(self, message: str, *, raw_text: str | None = None) -> None:
        self.raw_text = raw_text
        self.preview = None if raw_text is None else raw_text[:_RAW_PREVIEW_CHARS]
        if self.preview is not None:
            message = f"{message}; raw preview={self.preview!r}"
        super().__init__(message)


class SkeletonValidationError(Exception):
    """Raised when parsed skeleton JSON fails schema validation."""

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        self.reason = reason
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message)


def extract_skeleton_json(text: str) -> dict[str, Any]:
    """Extract a skeleton JSON object from raw LLM text.

    Markdown code fences are preferred because many LLMs wrap JSON in fenced
    blocks. If no fence is present, the first ``{`` through the last ``}`` is
    parsed. This function only parses text to a dict; schema and semantic
    validation are handled by higher layers.
    """

    candidates = [match.group(1).strip() for match in _FENCED_BLOCK_RE.finditer(text)]
    if not candidates:
        stripped = text.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            candidates = [stripped]
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise SkeletonParseError("LLM output does not contain a JSON object", raw_text=text)
            candidates = [text[start : end + 1].strip()]

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(value, dict):
            raise SkeletonParseError("LLM output JSON is not an object", raw_text=text)
        return value

    reason = f": {last_error}" if last_error is not None else ""
    raise SkeletonParseError(f"LLM output JSON could not be parsed{reason}", raw_text=text)


def build_generate_prompt(request: str) -> str:
    """Build the fixed prompt used to generate static skeleton JSON."""

    examples = [_format_example(index, template.build("示例需求").to_dict()) for index, template in enumerate(TEMPLATES, 1)]
    # 单一真相源：合法 inputs[].field 取自 linter.VALID_INPUT_FIELDS，避免 prompt 与
    # linter/build_contract 口径漂移（三处历史上不一致：prompt 曾只写 result/artifacts，
    # 而 linter 与 build_contract 都接受 output_file，编排模板又重度依赖 output_file）。
    allowed_fields = "、".join(f'"{field}"' for field in sorted(VALID_INPUT_FIELDS))
    return "\n\n".join(
        [
            "你是骨架编译器，把用户需求编译成静态协作骨架 JSON。",
            "硬约束：\n"
            "- 只生成 v1 静态骨架，禁止 dynamic_hooks 字段。\n"
            '- 每个写节点（write_scope 非空）的下游必须可达到一个 role=="reviewer" 的节点。\n'
            "- reviewer 节点必须有非空 reviewer_criteria。\n"
            "- 所有节点字段必填：id、role、task、depends_on、inputs、write_scope、read_only、required_capabilities、reviewer_criteria。\n"
            "- run_limits 必须有 max_repair_attempts 和 max_total_cost。\n"
            "- edges 必须与 depends_on 一致。\n"
            f"- inputs[].field 只能是 {allowed_fields}（下游读上游产出：result=主产出文本，"
            "artifacts=产物文件列表，output_file=上游声明的产出文件，按文件交接时优先用 output_file）。\n"
            "- 不要有环。",
            "输出要求：只输出一个 JSON 对象，不要 markdown 围栏，不要解释文字，不要调用任何工具或读写文件。",
            "范例（仅供格式参考，请按用户实际需求生成）：\n\n" + "\n\n".join(examples),
            f"用户需求：{request}",
        ]
    )


def build_repair_prompt(skeleton: Skeleton, violations: list[Any]) -> str:
    """Build the prompt used by meta_compile repair rounds."""

    skeleton_json = json.dumps(skeleton.to_dict(), ensure_ascii=False, indent=2)
    violation_lines = "\n".join(f"- {_violation_text(violation)}" for violation in violations) or "- 无"
    return "\n\n".join(
        [
            "你是骨架编译器，当前骨架未通过语义 lint。请修正这些问题后重新输出完整 JSON 骨架。",
            "仍须遵守：v1 静态骨架，禁止 dynamic_hooks；写节点下游必须可达到 reviewer；reviewer 必须有验收标准；edges 与 depends_on 一致；无环；只输出 JSON。",
            "当前骨架 JSON：\n" + skeleton_json,
            "违规列表：\n" + violation_lines,
            "请修正这些问题后重新输出完整 JSON 骨架，仍只输出 JSON，不要 markdown 围栏，不要解释文字，不要调用任何工具或读写文件。",
        ]
    )


class LLMCompiler:
    """SkeletonCompiler implementation backed by a real RunnerAdapter."""

    def __init__(
        self,
        adapter: RunnerAdapter,
        *,
        workspace: str = ".",
        timeout_sec: int = 1800,
        logical_role: str = "meta-orchestrator",
        model_hint: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.workspace = workspace
        self.timeout_sec = timeout_sec
        self.logical_role = logical_role
        self.model_hint = model_hint

    def generate(self, request: str) -> Skeleton:
        """Generate a schema-valid skeleton from a natural-language request."""

        return self._generate_from_prompt(build_generate_prompt(request))

    def repair(self, skeleton: Skeleton, violations: list[Any]) -> Skeleton:
        """Repair a schema-valid but lint-invalid skeleton."""

        return self._generate_from_prompt(build_repair_prompt(skeleton, violations))

    def _generate_from_prompt(self, prompt: str) -> Skeleton:
        contract = self._build_contract(prompt)
        result = self.adapter.invoke(contract, timeout_sec=self.timeout_sec)
        return self._parse_result(result)

    def _parse_result(self, result: AtomResult) -> Skeleton:
        if result.status != "success":
            raise SkeletonParseError(f"LLM 调用未成功: status={result.status} error={result.error}")

        data = extract_skeleton_json(result.result)
        try:
            validate_skeleton(data)
        except ValidationError as exc:
            raise SkeletonValidationError("LLM output failed skeleton schema validation", reason=str(exc)) from exc
        return Skeleton.from_dict(data)

    def _build_contract(self, prompt: str) -> AtomContract:
        now = datetime.now(timezone.utc).isoformat()
        return AtomContract(
            task=prompt,
            inputs=[],
            context_files=[],
            workspace=self.workspace,
            read_only=True,
            write_scope=[],
            required_capabilities=[],
            status="success",
            result="",
            artifacts=[],
            output_file="",
            output_schema_ref=None,
            handoff={"completed": [], "pending": [], "decisions": [], "risks": []},
            consult=None,
            atom_id="meta-compile",
            correlation_id="meta",
            logical_role=self.logical_role,
            resolved_runner=None,
            session_id=None,
            hop_count=1,
            limits={"max_cost": 5.0, "timeout_sec": self.timeout_sec, "max_internal_turns": 10},
            cost=0.0,
            duration_sec=0.0,
            timestamps={"started_at": now, "finished_at": now},
        )


def _format_example(index: int, data: dict[str, Any]) -> str:
    return f"范例 {index}:\n{json.dumps(data, ensure_ascii=False, indent=2)}"


def _violation_text(violation: Any) -> str:
    code = getattr(violation, "code", None)
    message = getattr(violation, "message", None)
    node_id = getattr(violation, "node_id", None)
    if isinstance(violation, dict):
        code = violation.get("code", code)
        message = violation.get("message", message)
        node_id = violation.get("node_id", node_id)
    return f"code={code or ''} node_id={node_id or ''} message={message or violation}"


__all__ = [
    "LLMCompiler",
    "SkeletonParseError",
    "SkeletonValidationError",
    "build_generate_prompt",
    "build_repair_prompt",
    "extract_skeleton_json",
]
