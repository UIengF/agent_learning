"""Ducc runner adapter backed by the Claude Code CLI JSON contract."""

from __future__ import annotations

import hashlib
import json
import os
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
)
from atomic_agents.models import Artifact, AtomContract, AtomResult, Status


_DEFAULT_DUCC_PATH = os.path.expanduser("~/.comate/baidu-cc/bin/ducc")


class DuccAdapter(RunnerAdapter):
    """RunnerAdapter implementation for the ducc Claude Code CLI."""

    adapter_version = "ducc-1"
    feature_profile = RunnerFeatureProfile(
        name="ducc",
        supports_session_resume=True,
        supports_cost_capture=True,
        supports_raw_events=False,
        supports_internal_turn_count=False,
        permission_modes=["default", "plan"],
    )

    def __init__(self, ducc_path: str = _DEFAULT_DUCC_PATH, model: str | None = None) -> None:
        self.ducc_path = os.path.expanduser(ducc_path)
        self.model = model

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
            return self._result(
                status="timeout",
                duration_sec=time.monotonic() - started,
                session_id=contract.session_id,
                error=str(exc),
            )
        except AdapterInvocationError as exc:
            return self._result(
                status="failed",
                duration_sec=time.monotonic() - started,
                session_id=contract.session_id,
                error=str(exc),
            )

        duration_sec = time.monotonic() - started
        payload, parse_error = _parse_stdout_json(completed.stdout)
        if parse_error is not None:
            return self._result(
                status="failed",
                duration_sec=duration_sec,
                session_id=contract.session_id,
                error=f"{parse_error.__class__.__name__}: {parse_error}",
            )

        session_id = _string_value(payload.get("session_id")) or contract.session_id
        result = _string_value(payload.get("result")) or ""
        cost = _numeric_value(payload.get("total_cost_usd"))
        status = _status_from_payload(payload, completed.returncode)
        error = _error_from_payload(payload, completed.stderr, completed.stdout, completed.returncode, status)
        artifacts = _collect_artifacts(contract.workspace, contract.write_scope)

        return self._result(
            status=status,
            result=result,
            artifacts=artifacts,
            session_id=session_id,
            cost=cost,
            duration_sec=duration_sec,
            raw_events_path=None,
            error=error,
        )

    def _build_cmd(self, contract: AtomContract) -> list[str]:
        cmd = [self.ducc_path, "-p", "--output-format", "json"]

        if contract.read_only:
            cmd.extend(["--permission-mode", "plan"])
        else:
            cmd.append("--dangerously-skip-permissions")
        if contract.session_id:
            cmd.extend(["--resume", contract.session_id])
        if self.model:
            cmd.extend(["--model", self.model])
        for context_dir in _context_dirs(contract.context_files):
            cmd.extend(["--add-dir", context_dir])

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
            output_file="",
            output_sha256=None,
        )


def _parse_stdout_json(stdout: str) -> tuple[dict[str, Any], AdapterParseError | None]:
    text = stdout.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}, AdapterParseError("failed to parse ducc stdout: missing JSON object")

    json_text = text[start : end + 1]
    try:
        value = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {}, AdapterParseError(f"failed to parse ducc stdout JSON: {exc}")

    if not isinstance(value, dict):
        return {}, AdapterParseError("failed to parse ducc stdout: JSON value is not an object")
    return value, None


def _status_from_payload(payload: dict[str, Any], returncode: int) -> Status:
    # 先做 transient 判定，再看 returncode：流式连接闪断时 ducc 常以非零码退出且
    # is_error=True，但错误文本是连接级瞬时问题（"socket connection was closed
    # unexpectedly"）。若让 `returncode != 0` 抢先 return "failed"，瞬时退避就永远
    # 够不着——这正是 socket 断连被误判 atom_failed 连坐的根因。
    http_status = _http_status(payload)
    message = (
        _string_value(payload.get("result"))
        or _string_value(payload.get("error"))
        or _string_value(payload.get("message"))
    )
    if is_transient_error(http_status=http_status, message=message):
        return "transient"

    if returncode != 0:
        return "failed"
    if payload.get("is_error") is True:
        return "failed"

    subtype = payload.get("subtype")
    if subtype is not None and str(subtype) != "success":
        return "failed"
    return "success"


def _http_status(payload: dict[str, Any]) -> int | None:
    for key in ("api_error_status", "http_status", "status_code"):
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _error_from_payload(
    payload: dict[str, Any],
    stderr: str,
    stdout: str,
    returncode: int,
    status: Status,
) -> str | None:
    # success 无 error；failed 与 transient 都需保留错误文本（transient 用于上报基础设施原因）。
    if status == "success":
        return None

    parts = [f"Ducc adapter exited with code {returncode}"]
    subtype = payload.get("subtype")
    if subtype is not None:
        parts.append(f"subtype: {subtype}")
    if payload.get("is_error") is not None:
        parts.append(f"is_error: {payload.get('is_error')}")

    message = (
        _string_value(payload.get("error"))
        or _string_value(payload.get("message"))
        or _string_value(payload.get("result"))
    )
    if message:
        parts.append(f"error: {_truncate(message)}")
    elif stderr.strip():
        parts.append(f"stderr: {_truncate(stderr.strip())}")
    elif stdout.strip():
        parts.append(f"stdout: {_truncate(stdout.strip())}")

    return "\n".join(parts)


def _context_dirs(context_files: list[str]) -> list[str]:
    dirs: list[str] = []
    seen: set[str] = set()
    for context_file in context_files:
        parent = str(Path(context_file).parent)
        if parent == "":
            parent = "."
        if parent in seen:
            continue
        seen.add(parent)
        dirs.append(parent)
    return dirs


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


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _truncate(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


__all__ = [
    "DuccAdapter",
    "AdapterInvocationError",
    "AdapterParseError",
]
