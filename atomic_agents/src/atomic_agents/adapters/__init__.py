"""Runner adapter interfaces and shared process helpers."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from atomic_agents.models import AtomContract, AtomResult


@dataclass
class RunnerFeatureProfile:
    """Capability surface exposed by a runner adapter."""

    name: str
    supports_session_resume: bool
    supports_cost_capture: bool
    supports_raw_events: bool
    supports_internal_turn_count: bool
    permission_modes: list[str]


@runtime_checkable
class RunnerAdapter(Protocol):
    """Uniform interface for black-box runner implementations."""

    feature_profile: RunnerFeatureProfile

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        """Run one atom contract and return a normalized result."""


class AdapterError(Exception):
    """Base error for runner adapter failures."""


class AdapterTimeout(AdapterError):
    """Raised when a runner process exceeds its timeout."""


class AdapterInvocationError(AdapterError):
    """Raised when a runner cannot be invoked."""


class AdapterParseError(AdapterError):
    """Raised when adapter output cannot be parsed."""


# 推理网关临时不可用的 HTTP 状态码（限流 / 网关 / 上游不可用）。这些是【基础设施】
# 问题，不是任务失败——调度器应退避重试、不烧 repair 配额，而非误报为 atom_failed。
_TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})

# 网关临时错误的文本特征（保守匹配：只认明确是网关/限流/重试类的措辞）。
_TRANSIENT_ERROR_MARKERS = (
    "credentials exhausted",
    "server-side issue",
    "usually temporary",
    "try again",
    "retry later",
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
    "gateway timeout",
    "bad gateway",
    # 流式连接中途断开 / 网络层闪断：是基础设施瞬时问题，不是任务失败。
    # 典型来自 ducc/Claude CLI 的 fetch 层（"socket connection was closed unexpectedly"）
    # 与底层网络重置（"connection reset"）。保守起见仍要求是明确的连接级措辞。
    "socket connection was closed",
    "connection was closed unexpectedly",
    "connection reset",
    "connection closed",
    "econnreset",
    "socket hang up",
    "network error",
    # 网关读取上游推理服务响应体/建连失败：上游流式响应在网关侧被截断或上游不可达，
    # 是基础设施瞬时问题（典型来自 ducc/Claude CLI 网关："API Error: Upstream body
    # read failed" / "upstream connect error" / "upstream request timeout"）。
    "upstream body read failed",
    "upstream connect error",
    "upstream request timeout",
    "body read failed",
)

# ask_codex.sh (v1.1+) echoes this marker to stderr *before* spawning codex, giving
# the path to the live JSONL event file it writes incrementally while the child
# runs (unlike raw_events_path, which is only produced after the child exits).
# Older ask_codex.sh copies won't emit this line — callers must treat its absence
# as "no live tail available" and fall back to workspace-mtime/output-byte polling.
_LIVE_EVENTS_MARKER_RE = re.compile(r"^\[codex\] live_events_path=(.+)$")


def is_transient_error(*, http_status: int | None = None, message: str | None = None) -> bool:
    """Heuristically classify a runner error as a transient infrastructure problem.

    保守判定：HTTP 状态命中限流/网关码，或错误文本含明确的网关临时错误特征词，才判为
    瞬时。模糊情形一律返回 False（继续按真失败处理），避免把真正的任务失败误当瞬时而
    无意义重试。
    """

    if http_status is not None and int(http_status) in _TRANSIENT_HTTP_STATUSES:
        return True
    if message:
        lowered = message.lower()
        return any(marker in lowered for marker in _TRANSIENT_ERROR_MARKERS)
    return False


def transient_error_from_events(events_path: str | None) -> tuple[bool, str | None]:
    """Return whether a JSONL runner event stream contains a transient error."""

    if not events_path:
        return False, None

    transient_message: str | None = None
    try:
        with open(events_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                http_status = _extract_http_status(event)
                message = _extract_error_message(event)
                if is_transient_error(http_status=http_status, message=message):
                    transient_message = message or (f"HTTP {http_status}" if http_status is not None else None)
                    return True, transient_message
    except OSError:
        return False, None

    return False, None


def _extract_http_status(value: Any) -> int | None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in {"api_error_status", "http_status", "status_code", "status"}:
                status = _coerce_http_status(item)
                if status is not None:
                    return status
            nested = _extract_http_status(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _extract_http_status(item)
            if nested is not None:
                return nested
    return None


def _coerce_http_status(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _extract_error_message(value: Any) -> str | None:
    messages: list[str] = []
    _collect_error_messages(value, messages)
    if not messages:
        return None
    return "\n".join(messages)


def _collect_error_messages(value: Any, messages: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in {"message", "error", "error_message", "detail", "reason"}:
                text = _coerce_message(item)
                if text:
                    messages.append(text)
            _collect_error_messages(item, messages)
    elif isinstance(value, list):
        for item in value:
            _collect_error_messages(item, messages)


def _coerce_message(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _extract_error_message(value)
    if isinstance(value, list):
        return _extract_error_message(value)
    return None


def build_output_directive(contract: AtomContract) -> str:
    """Build prompt instructions that bind a runner to contract.output_file."""

    if not contract.output_file:
        return ""

    directive = (
        "【产出要求】完成后，你必须把最终产出完整写入文件："
        f"{contract.output_file}（相对 {contract.workspace}）。"
        "不要只在对话里输出；该文件就是本次任务的交付物。"
    )
    if contract.logical_role.strip().lower() == "reviewer":
        directive += (
            "\n【审查产出格式】你是审查者。把审查结论写成 JSON 对象到上述文件，字段："
            "passed(bool 是否通过), "
            "criteria(数组, 每项 {criterion, verdict: pass|fail, evidence, confidence: high|medium|low}), "
            "blocking_findings(未通过的 criterion 名数组), "
            "reviewer_session(审查会话标识, 没有则为 null), "
            "feedback(给上游的修改建议文本)。"
            "passed 应与 blocking_findings 为空一致。只写这个 JSON 到文件, 不要其他内容。"
        )
    return directive


def augment_task_with_output_directive(contract: AtomContract) -> str:
    """Prefix contract.task with output-file instructions when needed."""

    directive = build_output_directive(contract)
    context_directive = build_context_files_directive(contract)
    prefix = "\n\n".join(part for part in (directive, context_directive) if part)
    if not prefix:
        return contract.task
    return f"{prefix}\n\n{contract.task}"


def build_context_files_directive(contract: AtomContract) -> str:
    """Build a uniform "read these upstream files first" directive.

    统一 codex/ducc 的上游产出可见性：两者都在 prompt 里收到【优先阅读的具体文件清单】，
    而非靠 runner 各自的注入语义（codex 原本用 -f 注入文件清单、ducc 用 --add-dir 只授权
    整个目录而无文件级指引，导致同一契约下两 runner 行为不对称）。文件路径相对 workspace。
    """

    if not contract.context_files:
        return ""

    lines = "\n".join(f"- {path}" for path in contract.context_files)
    return (
        "【上游产出】先阅读以下来自上游原子的产出文件（路径相对 "
        f"{contract.workspace}），再开始本任务：\n{lines}"
    )


def run_with_timeout(
    cmd: list[str],
    timeout_sec: int,
    cwd: str | None = None,
    stdin_text: str | None = None,
    watch_dir: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command in a new process group, killing it after idle timeout."""

    initial_watch_mtime = _max_file_mtime_ns(watch_dir) if watch_dir is not None else None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise AdapterInvocationError(f"failed to start command {' '.join(cmd)}: {exc}") from exc

    if watch_dir is not None:
        return _communicate_with_idle_timeout(
            process,
            cmd,
            timeout_sec,
            stdin_text,
            watch_dir,
            initial_watch_mtime,
        )

    try:
        stdout, stderr = process.communicate(input=stdin_text, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()

        stdout = _coerce_text(stdout or exc.output)
        stderr = _coerce_text(stderr or exc.stderr)
        detail = f"command timed out after {timeout_sec}s: {' '.join(cmd)}"
        if stderr:
            detail = f"{detail}\nstderr: {_summarize_text(stderr)}"
        if stdout:
            detail = f"{detail}\nstdout: {_summarize_text(stdout)}"
        raise AdapterTimeout(detail) from exc

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _communicate_with_idle_timeout(
    process: subprocess.Popen[str],
    cmd: list[str],
    timeout_sec: int,
    stdin_text: str | None,
    watch_dir: str,
    initial_watch_mtime: int | None,
) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    last_progress = started
    last_mtime = initial_watch_mtime
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    last_output_len = 0
    stdout_thread = _start_drain_thread(process.stdout, stdout_chunks)
    stderr_thread = _start_drain_thread(process.stderr, stderr_chunks)
    stdin_thread = _start_stdin_thread(process, stdin_text)
    timed_out = False
    live_events_path: str | None = None
    last_activity: str | None = None

    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                _join_threads(stdout_thread, stderr_thread, stdin_thread)
                if live_events_path is None:
                    live_events_path = _find_live_events_path(stderr_chunks)
                if live_events_path is not None:
                    tailed = _tail_last_activity(live_events_path)
                    if tailed is not None:
                        last_activity = tailed
                return _completed_process(
                    cmd,
                    returncode,
                    "".join(stdout_chunks),
                    "".join(stderr_chunks),
                    last_activity=last_activity,
                )

            now = time.monotonic()
            current_mtime = _max_file_mtime_ns(watch_dir)
            if current_mtime is not None and (last_mtime is None or current_mtime > last_mtime):
                last_mtime = current_mtime
                last_progress = now
            current_output_len = sum(len(chunk) for chunk in stdout_chunks) + sum(len(chunk) for chunk in stderr_chunks)
            if current_output_len > last_output_len:
                last_output_len = current_output_len
                last_progress = now

            if live_events_path is None:
                live_events_path = _find_live_events_path(stderr_chunks)
            if live_events_path is not None:
                tailed = _tail_last_activity(live_events_path)
                if tailed is not None:
                    last_activity = tailed

            if now - last_progress > timeout_sec:
                timed_out = True
                _terminate_process_group(process)
                _join_threads(stdout_thread, stderr_thread, stdin_thread)
                stdout = "".join(stdout_chunks)
                stderr = "".join(stderr_chunks)
                detail = f"command timed out after {timeout_sec}s: {' '.join(cmd)} (idle: no workspace file changes and no subprocess output)"
                if last_activity:
                    detail = f"{detail}\nlast_activity: {last_activity}"
                if stderr:
                    detail = f"{detail}\nstderr: {_summarize_text(stderr)}"
                if stdout:
                    detail = f"{detail}\nstdout: {_summarize_text(stdout)}"
                raise AdapterTimeout(detail)

            remaining_idle = timeout_sec - (time.monotonic() - last_progress)
            sleep_for = min(5.0, max(0.1, remaining_idle))
            try:
                process.wait(timeout=sleep_for)
            except subprocess.TimeoutExpired:
                pass
    except BaseException:
        if not timed_out and process.poll() is None:
            _terminate_process_group(process)
        _join_threads(stdout_thread, stderr_thread, stdin_thread)
        raise


def _completed_process(
    args: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    last_activity: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
    completed.last_activity = last_activity  # type: ignore[attr-defined]
    return completed


def _find_live_events_path(stderr_chunks: list[str]) -> str | None:
    for line in "".join(stderr_chunks).splitlines():
        match = _LIVE_EVENTS_MARKER_RE.match(line)
        if match:
            return match.group(1).strip()
    return None


def _tail_last_activity(events_path: str) -> str | None:
    last_activity: str | None = None
    try:
        with open(events_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    continue
                summary = _activity_summary(event)
                if summary is not None:
                    last_activity = summary
    except Exception:
        return None
    return last_activity


def _activity_summary(event: object) -> str | None:
    if not isinstance(event, dict):
        return None
    item = event.get("item")
    if not isinstance(item, dict):
        return None

    if event.get("type") == "item.started" and item.get("type") == "command_execution":
        command = item.get("command")
        if isinstance(command, str):
            return f"running: {_truncate_activity_text(command)}"

    if event.get("type") == "item.completed" and item.get("type") == "agent_message":
        text = item.get("text")
        if isinstance(text, str):
            return f"message: {_truncate_activity_text(text)}"

    return None


def _truncate_activity_text(value: str, max_chars: int = 200) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}..."


def _start_drain_thread(stream: object, chunks: list[str]) -> threading.Thread:
    def drain() -> None:
        if stream is None:
            return
        try:
            while True:
                line = stream.readline()  # type: ignore[attr-defined]
                if not line:
                    break
                chunks.append(_coerce_text(line))
        except (OSError, ValueError):
            pass

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread


def _start_stdin_thread(process: subprocess.Popen[str], stdin_text: str | None) -> threading.Thread | None:
    if stdin_text is None or process.stdin is None:
        return None

    def write_stdin() -> None:
        try:
            process.stdin.write(stdin_text)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    thread = threading.Thread(target=write_stdin, daemon=True)
    thread.start()
    return thread


def _join_threads(*threads: threading.Thread | None) -> None:
    for thread in threads:
        if thread is not None:
            thread.join(timeout=2)


def _max_file_mtime_ns(watch_dir: str) -> int | None:
    max_mtime: int | None = None
    for root, _dirs, files in os.walk(watch_dir):
        for filename in files:
            try:
                stat = os.stat(os.path.join(root, filename))
            except OSError:
                continue
            if max_mtime is None or stat.st_mtime_ns > max_mtime:
                max_mtime = stat.st_mtime_ns
    return max_mtime


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        try:
            process.wait(timeout=0)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
        return

    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        try:
            process.wait(timeout=0)
        except (subprocess.TimeoutExpired, ChildProcessError):
            pass
        return
    process.wait()


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _summarize_text(value: str, max_chars: int = 1000) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


__all__ = [
    "AdapterError",
    "AdapterInvocationError",
    "AdapterParseError",
    "AdapterTimeout",
    "RunnerAdapter",
    "RunnerFeatureProfile",
    "augment_task_with_output_directive",
    "build_context_files_directive",
    "build_output_directive",
    "is_transient_error",
    "run_with_timeout",
    "transient_error_from_events",
]
