from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any

from .agent import CodingAgent
from .config import AppConfig
from .evidence_cache import compact_text
from .permissions import PermissionDenied, WorkspacePolicy
from .validation import run_validation


VALID_MODES = {"ask", "plan", "edit", "agent", "run"}


@dataclass(frozen=True)
class EvalTask:
    name: str
    task: str
    suggested_mode: str
    visible_validation: tuple[str, ...] = ()
    hidden_validation: tuple[str, ...] = ()
    group_id: str = ""
    continue_session: bool = False
    setup_files: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EvalValidationResult:
    command: str
    status: str
    output_summary: str
    output_path: str = ""


@dataclass(frozen=True)
class EvalTaskResult:
    name: str
    mode: str
    session_id: str
    task_id: str
    status: str
    response_summary: str
    validations: tuple[EvalValidationResult, ...]
    duration_seconds: float
    trace_path: str = ""
    response_path: str = ""
    setup_errors: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalSuiteResult:
    workspace: str
    suite_path: str
    task_results: tuple[EvalTaskResult, ...]

    def count(self, status: str) -> int:
        return sum(1 for result in self.task_results if result.status == status)

    @property
    def task_count(self) -> int:
        return len(self.task_results)

    @property
    def has_failures(self) -> bool:
        return any(
            result.status in {"failed", "denied", "error", "leak"}
            for result in self.task_results
        )


def _validation_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, list):
        commands = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    commands.append(stripped)
        return tuple(commands)
    return ()


def _expand_suite_placeholders(
    commands: tuple[str, ...],
    *,
    suite_dir: Path,
) -> tuple[str, ...]:
    suite_dir_text = suite_dir.as_posix()
    return tuple(command.replace("{{suite_dir}}", suite_dir_text) for command in commands)


def _suggested_mode(value: Any) -> str:
    mode = str(value or "agent").strip().lower()
    return mode if mode in VALID_MODES else "agent"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _setup_files(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        return ()
    files = []
    for path, content in value.items():
        if not isinstance(path, str):
            continue
        files.append((path, str(content)))
    return tuple(files)


def load_eval_suite(path: str | Path) -> tuple[EvalTask, ...]:
    suite_path = Path(path).resolve()
    suite_dir = suite_path.parent
    data: dict[str, Any] = json.loads(suite_path.read_text(encoding="utf-8-sig"))
    tasks = []
    for raw in data.get("tasks", []):
        if not isinstance(raw, dict):
            continue
        hidden_validation = _validation_tuple(raw.get("hidden_validation"))
        if not hidden_validation:
            hidden_validation = _validation_tuple(raw.get("expected_validation"))
        visible_validation = _validation_tuple(raw.get("visible_validation"))
        tasks.append(
            EvalTask(
                name=str(raw.get("name") or "unnamed"),
                task=str(raw.get("task") or ""),
                suggested_mode=_suggested_mode(raw.get("suggested_mode")),
                visible_validation=_expand_suite_placeholders(
                    visible_validation,
                    suite_dir=suite_dir,
                ),
                hidden_validation=_expand_suite_placeholders(
                    hidden_validation,
                    suite_dir=suite_dir,
                ),
                group_id=str(raw.get("group_id") or ""),
                continue_session=_as_bool(raw.get("continue_session")),
                setup_files=_setup_files(raw.get("setup_files")),
            )
        )
    return tuple(tasks)


def _build_agent_task(task: EvalTask) -> str:
    if not task.visible_validation:
        return task.task
    return "\n\n".join(
        [
            task.task,
            "Suggested validation:\n" + "\n".join(task.visible_validation),
        ]
    )


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip().lower()).strip("-")
    return safe or "unnamed"


def _session_id(suite_path: Path, index: int, task: EvalTask) -> str:
    return f"eval-{_safe_name(suite_path.stem)}-{index:03d}-{_safe_name(task.name)}"


def _validation_status(result: object) -> str:
    ok = getattr(result, "ok", False)
    return "passed" if ok else "failed"


def _validation_output(result: object) -> str:
    return compact_text(_validation_output_text(result), 1200)


def _validation_output_text(result: object) -> str:
    formatter = getattr(result, "format", None)
    if callable(formatter):
        return str(formatter())
    return str(result)


def _run_hidden_validations(
    policy: WorkspacePolicy,
    commands: tuple[str, ...],
    *,
    timeout_seconds: int,
    artifact_dir: Path | None = None,
) -> tuple[EvalValidationResult, ...]:
    results: list[EvalValidationResult] = []
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands, start=1):
        output_path = ""
        try:
            validation = run_validation(
                policy,
                command,
                timeout_seconds=timeout_seconds,
            )
            output_text = _validation_output_text(validation)
            output = compact_text(output_text, 1200)
            if artifact_dir is not None:
                output_file = artifact_dir / f"hidden_validation_{index:02d}.txt"
                output_file.write_text(output_text, encoding="utf-8")
                output_path = str(output_file)
            results.append(
                EvalValidationResult(
                    command=getattr(validation, "command", command),
                    status=_validation_status(validation),
                    output_summary=output,
                    output_path=output_path,
                )
            )
        except PermissionDenied as exc:
            output = compact_text(str(exc), 1200)
            if artifact_dir is not None:
                output_file = artifact_dir / f"hidden_validation_{index:02d}.txt"
                output_file.write_text(str(exc), encoding="utf-8")
                output_path = str(output_file)
            results.append(
                EvalValidationResult(
                    command=command,
                    status="denied",
                    output_summary=output,
                    output_path=output_path,
                )
            )
        except subprocess.TimeoutExpired as exc:
            output = compact_text(f"validation timed out: {exc}", 1200)
            if artifact_dir is not None:
                output_file = artifact_dir / f"hidden_validation_{index:02d}.txt"
                output_file.write_text(f"validation timed out: {exc}", encoding="utf-8")
                output_path = str(output_file)
            results.append(
                EvalValidationResult(
                    command=command,
                    status="error",
                    output_summary=output,
                    output_path=output_path,
                )
            )
        except Exception as exc:
            output = compact_text(str(exc), 1200)
            if artifact_dir is not None:
                output_file = artifact_dir / f"hidden_validation_{index:02d}.txt"
                output_file.write_text(str(exc), encoding="utf-8")
                output_path = str(output_file)
            results.append(
                EvalValidationResult(
                    command=command,
                    status="error",
                    output_summary=output,
                    output_path=output_path,
                )
            )
    return tuple(results)


def _task_status(
    agent_error: str,
    validations: tuple[EvalValidationResult, ...],
    flags: tuple[str, ...] = (),
) -> str:
    if "hidden_leak" in flags:
        return "leak"
    if agent_error:
        return "error"
    if not validations:
        return "passed_unvalidated"
    statuses = {validation.status for validation in validations}
    if "error" in statuses:
        return "error"
    if "denied" in statuses:
        return "denied"
    if "failed" in statuses:
        return "failed"
    return "passed"


def _hidden_validation_needles(commands: tuple[str, ...]) -> tuple[str, ...]:
    needles: list[str] = []
    for command in commands:
        if command:
            needles.append(command)
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            parts = command.split()
        for part in parts:
            stripped = part.strip("\"'")
            if not stripped:
                continue
            normalized = stripped.replace("\\", "/").strip("/")
            path_parts = [part for part in normalized.split("/") if part]
            if stripped.endswith(".py") or len(path_parts) >= 2:
                needles.append(stripped)
    return tuple(dict.fromkeys(needles))


def _detect_hidden_leak(response: str, commands: tuple[str, ...]) -> tuple[str, ...]:
    if not response or not commands:
        return ()
    lowered = response.replace("\\", "/").lower()
    for needle in _hidden_validation_needles(commands):
        normalized_needle = needle.replace("\\", "/").lower()
        if normalized_needle and normalized_needle in lowered:
            return ("hidden_leak",)
    return ()


def _trace_size(config: AppConfig, session_id: str) -> int:
    trace_path = config.harness.runtime_dir / "traces" / f"{session_id}.jsonl"
    try:
        return trace_path.stat().st_size
    except OSError:
        return 0


def _read_trace_text_since(config: AppConfig, session_id: str, offset: int) -> str:
    trace_path = config.harness.runtime_dir / "traces" / f"{session_id}.jsonl"
    try:
        with trace_path.open("rb") as trace:
            size = trace.seek(0, 2)
            trace.seek(offset if 0 <= offset <= size else 0)
            return trace.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _trace_path(config: AppConfig, session_id: str) -> Path:
    return config.harness.runtime_dir / "traces" / f"{session_id}.jsonl"


def _artifact_dir(
    config: AppConfig,
    session_id: str,
    task_index: int,
    task: EvalTask,
) -> Path:
    return (
        config.harness.runtime_dir
        / "eval_artifacts"
        / session_id
        / f"{task_index:03d}-{_safe_name(task.name)}"
    )


def _write_response_artifact(artifact_dir: Path, response_text: str) -> str:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "response.txt"
    path.write_text(response_text, encoding="utf-8")
    return str(path)


def _write_setup_files(
    policy: WorkspacePolicy,
    setup_files: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    errors = []
    for path, content in setup_files:
        try:
            resolved = policy.resolve_path(path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return tuple(errors)


def run_eval_suite(
    config: AppConfig,
    workspace: str | Path,
    suite_path: str | Path,
) -> EvalSuiteResult:
    workspace_path = Path(workspace).resolve()
    suite = Path(suite_path).resolve()
    tasks = load_eval_suite(suite)
    policy = WorkspacePolicy(
        workspace=workspace_path,
        allowed_commands=config.harness.allowed_commands,
    )
    task_results: list[EvalTaskResult] = []
    group_sessions: dict[str, str] = {}
    for index, task in enumerate(tasks, start=1):
        started = time.monotonic()
        if task.group_id and task.continue_session and task.group_id in group_sessions:
            session_id = group_sessions[task.group_id]
        else:
            session_id = _session_id(suite, index, task)
            if task.group_id:
                group_sessions[task.group_id] = session_id
        task_id = ""
        response_summary = ""
        response_text = ""
        response_path = ""
        validations: tuple[EvalValidationResult, ...] = ()
        setup_errors = _write_setup_files(policy, task.setup_files)
        trace_offset = _trace_size(config, session_id)
        agent_error = ""
        if setup_errors:
            agent_error = "\n".join(setup_errors)
            response_text = agent_error
            response_summary = compact_text(agent_error, 1200)
        else:
            try:
                agent = CodingAgent(config=config, workspace=workspace_path, session_id=session_id)
                result = agent.run_mode_task(task.suggested_mode, _build_agent_task(task))
                task_id = result.task_id
                response_text = result.response
                response_summary = compact_text(response_text, 1200)
            except Exception as exc:
                agent_error = str(exc)
                response_text = agent_error
                response_summary = compact_text(response_text, 1200)
        artifact_dir = _artifact_dir(config, session_id, index, task)
        response_path = _write_response_artifact(artifact_dir, response_text)
        validations = _run_hidden_validations(
            policy,
            task.hidden_validation,
            timeout_seconds=config.harness.command_timeout_seconds,
            artifact_dir=artifact_dir,
        )
        leak_text = "\n".join(
            [response_text, _read_trace_text_since(config, session_id, trace_offset)]
        )
        flags = _detect_hidden_leak(leak_text, task.hidden_validation)
        task_results.append(
            EvalTaskResult(
                name=task.name,
                mode=task.suggested_mode,
                session_id=session_id,
                task_id=task_id,
                status=_task_status(agent_error, validations, flags),
                response_summary=response_summary,
                validations=validations,
                setup_errors=setup_errors,
                flags=flags,
                trace_path=str(_trace_path(config, session_id)),
                response_path=response_path,
                duration_seconds=time.monotonic() - started,
            )
        )
    return EvalSuiteResult(
        workspace=str(workspace_path),
        suite_path=str(suite),
        task_results=tuple(task_results),
    )


def format_eval_result(result: EvalSuiteResult) -> str:
    lines = [
        "Eval summary:",
        f"- workspace: {result.workspace}",
        f"- suite: {result.suite_path}",
        f"- task count: {result.task_count}",
        f"- passed: {result.count('passed')}",
        f"- failed: {result.count('failed')}",
        f"- denied: {result.count('denied')}",
        f"- error: {result.count('error')}",
        f"- leak: {result.count('leak')}",
        f"- passed_unvalidated: {result.count('passed_unvalidated')}",
    ]
    for index, task in enumerate(result.task_results, start=1):
        lines.extend(
            [
                f"{index}. {task.name}",
                f"   mode: {task.mode}",
                f"   status: {task.status}",
                f"   session_id: {task.session_id}",
                f"   task_id: {task.task_id or '<none>'}",
                f"   duration_seconds: {task.duration_seconds:.2f}",
                f"   trace_path: {task.trace_path}",
                f"   response_path: {task.response_path}",
            ]
        )
        if task.setup_errors:
            lines.append("   setup errors:")
            lines.extend(f"   - {error}" for error in task.setup_errors)
        if task.flags:
            lines.append("   flags:")
            lines.extend(f"   - {flag}" for flag in task.flags)
        if task.validations:
            lines.append("   validations:")
            for validation in task.validations:
                lines.append(f"   - {validation.command}: {validation.status}")
                if validation.output_path:
                    lines.append(f"     output_path: {validation.output_path}")
                if validation.output_summary:
                    lines.append(f"     output: {validation.output_summary}")
        else:
            lines.append("   validations: <none>")
        lines.extend(
            [
                "   response summary:",
                f"   {task.response_summary or '<empty>'}",
            ]
        )
    return "\n".join(lines)


def format_eval_dry_run(workspace: str, suite_path: str | Path) -> str:
    tasks = load_eval_suite(suite_path)
    lines = [
        "Eval dry-run summary:",
        f"- workspace: {workspace}",
        f"- suite: {suite_path}",
        f"- task count: {len(tasks)}",
        "- execution: skipped; no model calls, file edits, or validation commands were run.",
    ]
    for index, task in enumerate(tasks, start=1):
        lines.extend(
            [
                f"{index}. {task.name}",
                f"   task: {task.task}",
                f"   suggested mode: {task.suggested_mode}",
                f"   visible validation: {', '.join(task.visible_validation) or '<none>'}",
                f"   hidden validation: {', '.join(task.hidden_validation) or '<none>'}",
                f"   setup files: {len(task.setup_files)}",
            ]
        )
    return "\n".join(lines)
