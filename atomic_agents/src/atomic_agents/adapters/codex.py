"""Codex runner adapter backed by the ask_codex.sh contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from atomic_agents.adapters import (
    AdapterInvocationError,
    AdapterParseError,
    AdapterTimeout,
    RunnerAdapter,
    RunnerFeatureProfile,
    augment_task_with_output_directive,
    is_transient_error,
    run_with_timeout,
    transient_error_from_events,
)
from atomic_agents.models import Artifact, AtomContract, AtomResult, Status


_DEFAULT_ASK_CODEX_PATH = str(
    Path(__file__).resolve().parents[3] / "vendor" / "codex" / "scripts" / "ask_codex.sh"
)
_CONTRACT_LINE_RE = re.compile(r"^(session_id|output_path|raw_events_path)=(.*)$")

# Codex CLI exposes only token usage (no USD). We normalize cost to USD so that
# `max_total_cost` means dollars across every runner (ducc already reports USD).
# A single blended rate is applied to the summed token count; the token total is
# input-dominated in practice, so the default leans toward the input-side price.
# Override per call site via the `usd_per_mtok` constructor arg if pricing shifts.
_DEFAULT_USD_PER_MTOK = 10.0


class CodexAdapter(RunnerAdapter):
    """RunnerAdapter implementation for the Codex CLI wrapper script."""

    adapter_version = "codex-1"
    feature_profile = RunnerFeatureProfile(
        name="codex",
        supports_session_resume=True,
        supports_cost_capture=True,
        supports_raw_events=True,
        supports_internal_turn_count=True,
        permission_modes=["full", "read-only"],
    )

    def __init__(
        self,
        ask_codex_path: str = _DEFAULT_ASK_CODEX_PATH,
        model: str | None = None,
        usd_per_mtok: float = _DEFAULT_USD_PER_MTOK,
    ) -> None:
        self.ask_codex_path = os.path.expanduser(ask_codex_path)
        self.model = model
        self.usd_per_mtok = usd_per_mtok

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        started = time.monotonic()
        cmd = self._build_cmd(contract)
        prompt = augment_task_with_output_directive(contract)

        try:
            completed = run_with_timeout(
                cmd,
                timeout_sec,
                cwd=contract.workspace,
                stdin_text=prompt,
                watch_dir=contract.workspace,
            )
        except AdapterTimeout as exc:
            last_activity = _last_activity_from_timeout_error(str(exc))
            return self._result(
                status="timeout",
                duration_sec=time.monotonic() - started,
                session_id=contract.session_id,
                error=str(exc),
                last_activity=last_activity,
            )
        except AdapterInvocationError as exc:
            return self._result(
                status="failed",
                duration_sec=time.monotonic() - started,
                session_id=contract.session_id,
                error=str(exc),
            )

        last_activity = getattr(completed, "last_activity", None)
        duration_sec = time.monotonic() - started
        parsed_stdout = _parse_contract_stdout(completed.stdout)
        session_id = parsed_stdout.get("session_id") or contract.session_id
        raw_events_path = parsed_stdout.get("raw_events_path")

        if completed.returncode != 0 or "[ERROR]" in completed.stderr:
            error = _summarize_error(completed.stderr, completed.stdout, completed.returncode)
            # 网关限流/5xx 等瞬时基础设施错误判为 transient，让调度器退避重试且不烧 repair
            # 配额；任务本身的失败仍判 failed。
            transient_from_events, event_message = transient_error_from_events(raw_events_path)
            transient_from_stdio = is_transient_error(message=f"{completed.stderr}\n{completed.stdout}")
            status: Status = "transient" if transient_from_events or transient_from_stdio else "failed"
            if event_message and event_message not in error:
                error = f"{error}\nraw_event: {_truncate(event_message)}"
            return self._result(
                status=status,
                duration_sec=duration_sec,
                session_id=session_id,
                raw_events_path=raw_events_path,
                error=error,
                last_activity=last_activity,
            )

        output_path = parsed_stdout.get("output_path")
        result, output_error = _read_output(output_path)
        token_total, _internal_turn_count = _parse_raw_events(raw_events_path)
        cost = _tokens_to_usd(token_total, self.usd_per_mtok)
        artifacts = _collect_artifacts(contract.workspace, contract.write_scope)

        return self._result(
            status="success",
            result=result,
            artifacts=artifacts,
            session_id=session_id,
            cost=cost,
            duration_sec=duration_sec,
            raw_events_path=raw_events_path,
            error=output_error,
            last_activity=last_activity,
        )

    def _build_cmd(self, contract: AtomContract) -> list[str]:
        cmd = [self.ask_codex_path, "-w", contract.workspace]

        if contract.read_only:
            cmd.append("--read-only")
        for context_file in contract.context_files:
            cmd.extend(["-f", context_file])
        if contract.session_id:
            cmd.extend(["--session", contract.session_id])
        if self.model:
            cmd.extend(["--model", self.model])

        return cmd

    def _result(
        self,
        *,
        status: Status,
        result: str = "",
        artifacts: list[Artifact] | None = None,
        session_id: str | None = None,
        cost: float = 0.0,
        duration_sec: float,
        raw_events_path: str | None = None,
        error: str | None = None,
        last_activity: str | None = None,
    ) -> AtomResult:
        return AtomResult(
            status=status,
            result=result,
            artifacts=artifacts or [],
            session_id=session_id,
            cost=cost,
            duration_sec=duration_sec,
            raw_events_path=raw_events_path,
            error=error,
            last_activity=last_activity,
            output_file="",
            output_sha256=None,
        )


def _parse_contract_stdout(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        match = _CONTRACT_LINE_RE.match(line.strip())
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def _last_activity_from_timeout_error(error: str) -> str | None:
    for line in error.splitlines():
        if line.startswith("last_activity: "):
            return line.removeprefix("last_activity: ").strip() or None
    return None


def _read_output(output_path: str | None) -> tuple[str, str | None]:
    if not output_path:
        return "", "missing output_path in Codex stdout contract"

    path = Path(output_path)
    try:
        result = path.read_text(encoding="utf-8")
    except OSError as exc:
        return "", f"failed to read output_path {output_path}: {exc}"

    if not result:
        return "", f"output_path {output_path} is empty"
    return result, None


def _parse_raw_events(raw_events_path: str | None) -> tuple[float, int]:
    """Return (summed token usage, internal turn count) from the raw events file.

    Codex reports usage as token counts only; the first element is the raw token
    total, converted to USD by :func:`_tokens_to_usd` at the call site.
    """

    if not raw_events_path:
        return 0.0, 0

    path = Path(raw_events_path)
    if not path.exists():
        return 0.0, 0

    cost = 0.0
    internal_turn_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cost += _extract_cost(event)
                internal_turn_count += _count_internal_turn(event)
    except OSError:
        return 0.0, 0

    return cost, internal_turn_count


def _tokens_to_usd(token_total: float, usd_per_mtok: float) -> float:
    """Convert a summed token count to USD using a single blended rate.

    ``usd_per_mtok`` is the price per one million tokens. A zero or negative
    token total (e.g. a missing or empty events file) yields ``0.0`` rather than
    a negative cost.
    """

    if token_total <= 0:
        return 0.0
    return token_total / 1_000_000.0 * usd_per_mtok


def _extract_cost(value: Any) -> float:
    if isinstance(value, dict):
        total = 0.0
        for key, item in value.items():
            key_lower = str(key).lower()
            if _is_cost_key(key_lower):
                total += _numeric_value(item) or _extract_cost(item)
            elif _is_usage_key(key_lower):
                total += _extract_usage_value(item)
            elif _is_usage_total_key(key_lower):
                total += _numeric_value(item)
            else:
                total += _extract_cost(item)
        return total

    if isinstance(value, list):
        return sum(_extract_cost(item) for item in value)

    return 0.0


def _extract_usage_value(value: Any) -> float:
    if isinstance(value, dict):
        total = 0.0
        for key, item in value.items():
            key_lower = str(key).lower()
            if _is_usage_total_key(key_lower):
                total += _numeric_value(item)
            elif isinstance(item, (dict, list)):
                total += _extract_usage_value(item)
        return total

    if isinstance(value, list):
        return sum(_extract_usage_value(item) for item in value)

    return 0.0


def _is_cost_key(key: str) -> bool:
    return key in {"cost", "total_cost", "cost_usd", "usd"} or key.endswith("_cost")


def _is_usage_key(key: str) -> bool:
    return "usage" in key


def _is_usage_total_key(key: str) -> bool:
    return key in {
        "total",
        "total_token",
        "total_tokens",
        "tokens_total",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
    }


def _numeric_value(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _count_internal_turn(event: dict[str, Any]) -> int:
    event_type = str(event.get("type", ""))
    if "turn" in event_type or event_type == "item.completed":
        return 1
    return 0


def _collect_artifacts(workspace: str, write_scope: list[str]) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for scoped_path in write_scope:
        path = Path(scoped_path)
        if not path.is_absolute():
            path = Path(workspace) / path
        if not path.is_file():
            continue
        artifacts.append(
            {
                "path": scoped_path,
                "type": "file",
                "sha256": _sha256_file(path),
            }
        )
    return artifacts


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_error(stderr: str, stdout: str, returncode: int) -> str:
    parts = [f"Codex adapter exited with code {returncode}"]
    if stderr.strip():
        parts.append(f"stderr: {_truncate(stderr.strip())}")
    elif stdout.strip():
        parts.append(f"stdout: {_truncate(stdout.strip())}")
    return "\n".join(parts)


def _truncate(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


__all__ = [
    "CodexAdapter",
    "AdapterInvocationError",
    "AdapterParseError",
]
