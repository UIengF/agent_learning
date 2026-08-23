from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest

from atomic_agents.adapters import (
    AdapterTimeout,
    _find_live_events_path,
    _tail_last_activity,
    augment_task_with_output_directive,
    build_context_files_directive,
    build_output_directive,
    is_transient_error,
    run_with_timeout,
    transient_error_from_events,
)
from atomic_agents.adapters.codex import CodexAdapter
from atomic_agents.adapters.ducc import DuccAdapter
import atomic_agents.adapters.codex as codex_module
import atomic_agents.adapters.ducc as ducc_module
from atomic_agents.models import AtomContract, AtomResult


FAKES_DIR = Path(__file__).parent / "fakes"
VALID_STATUSES = {"success", "failed", "blocked", "timeout", "transient"}


def make_fake_executable(tmp_path: Path, fake_name: str) -> str:
    source = FAKES_DIR / fake_name
    target = tmp_path / fake_name
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    os.chmod(target, 0o755)
    return str(target)


def assert_atom_result_shape(result: AtomResult) -> None:
    assert result.status in VALID_STATUSES
    assert isinstance(result.result, str)

    assert isinstance(result.artifacts, list)
    for artifact in result.artifacts:
        assert {"path", "type", "sha256"} <= set(artifact)
        assert isinstance(artifact["path"], str)
        assert isinstance(artifact["type"], str)
        assert isinstance(artifact["sha256"], str)

    assert isinstance(result.cost, float)
    assert result.cost >= 0
    assert isinstance(result.duration_sec, float)
    assert result.duration_sec >= 0
    assert result.session_id is None or isinstance(result.session_id, str)
    assert result.raw_events_path is None or isinstance(result.raw_events_path, str)
    assert result.error is None or isinstance(result.error, str)


def test_codex_success_normalizes_contract_result(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    scoped_file = tmp_path / "scoped.txt"
    scoped_file.write_text("artifact body\n", encoding="utf-8")
    contract = make_contract(write_scope=["scoped.txt"])
    adapter = CodexAdapter(ask_codex_path=make_fake_executable(tmp_path, "fake_codex_success.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "success"
    assert result.result == "codex ok\n"
    assert result.session_id == "sess-fake"
    assert result.cost > 0
    assert result.raw_events_path
    assert any(artifact["path"] == "scoped.txt" for artifact in result.artifacts)


def test_codex_success_preserves_completed_last_activity(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "codex-output.md"
    raw_events_path = tmp_path / "codex-events.jsonl"
    output_path.write_text("codex ok\n", encoding="utf-8")
    raw_events_path.write_text("", encoding="utf-8")

    completed = subprocess.CompletedProcess(
        args=["fake-codex"],
        returncode=0,
        stdout=f"session_id=sess-fake\noutput_path={output_path}\nraw_events_path={raw_events_path}\n",
        stderr="",
    )
    completed.last_activity = "message: completed tests"  # type: ignore[attr-defined]
    monkeypatch.setattr(codex_module, "run_with_timeout", lambda *args, **kwargs: completed)

    adapter = CodexAdapter(ask_codex_path=str(tmp_path / "fake-codex"))

    result = adapter.invoke(make_contract(), timeout_sec=3)

    assert_atom_result_shape(result)
    assert result.status == "success"
    assert result.last_activity == "message: completed tests"


def test_codex_failure_returns_failed_result(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract()
    adapter = CodexAdapter(ask_codex_path=make_fake_executable(tmp_path, "fake_codex_failed.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "failed"
    assert result.error is not None
    assert "boom" in result.error


def test_codex_socket_drop_returns_transient(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    # codex 长任务流式 socket 断开：非零退出 + [ERROR] 行是连接级瞬时问题，须判 transient
    # 让调度器退避重试，而非 failed。回归 codex stream 中断被误判 atom_failed 的根因。
    contract = make_contract()
    adapter = CodexAdapter(ask_codex_path=make_fake_executable(tmp_path, "fake_codex_socket_drop.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "transient"
    assert result.error is not None and "socket connection was closed" in result.error.lower()


def test_codex_failed_wrapper_reads_raw_events_for_transient_cause(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_events_path = tmp_path / "codex-events.jsonl"
    raw_events_path.write_text(
        json.dumps(
            {
                "type": "error",
                "message": "API Error: Upstream body read failed",
                "status_code": 503,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.CompletedProcess(
        args=["fake-codex"],
        returncode=1,
        stdout=f"raw_events_path={raw_events_path}\n",
        stderr="[ERROR] Codex reported a fatal event: error (no turn.completed)\n",
    )
    monkeypatch.setattr(codex_module, "run_with_timeout", lambda *args, **kwargs: completed)

    adapter = CodexAdapter(ask_codex_path=str(tmp_path / "fake-codex"))
    result = adapter.invoke(make_contract(), timeout_sec=3)

    assert_atom_result_shape(result)
    assert result.status == "transient"
    assert result.raw_events_path == str(raw_events_path)
    assert result.error is not None and "Upstream body read failed" in result.error


def test_codex_timeout_returns_timeout_without_waiting_for_sleep(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract(limits={"max_cost": 2.0, "timeout_sec": 1, "max_internal_turns": 30})
    adapter = CodexAdapter(ask_codex_path=make_fake_executable(tmp_path, "fake_codex_timeout.py"))

    started = time.monotonic()
    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])
    elapsed = time.monotonic() - started

    assert_atom_result_shape(result)
    assert result.status == "timeout"
    assert elapsed < 5


def test_ducc_success_normalizes_contract_result(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract()
    adapter = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_success.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "success"
    assert result.result == "ok"
    assert result.session_id == "sess-fake"
    assert result.cost == 0.07


def test_ducc_is_error_payload_returns_failed_result(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract()
    adapter = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_failed.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "failed"
    assert result.error is not None
    assert "is_error: True" in result.error


def test_ducc_malformed_stdout_returns_failed_result(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract()
    adapter = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_malformed.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "failed"
    assert result.error is not None
    assert "AdapterParseError" in result.error


def test_ducc_missing_cost_defaults_to_zero(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract()
    adapter = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_missing_cost.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "success"
    assert result.cost == 0.0


def test_ducc_gateway_503_returns_transient_not_failed(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    # 503 credentials exhausted 是基础设施瞬时错误，必须判 transient（让调度器退避重试、
    # 不烧 repair 配额），而不是 failed。错误文本须保留供如实归因。
    contract = make_contract()
    adapter = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_transient.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "transient"
    assert result.error is not None and "credentials exhausted" in result.error


def test_ducc_is_error_without_transient_markers_stays_failed(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    # 无网关/限流特征的 is_error 仍判 failed（保守：不把真任务失败误当瞬时）。
    contract = make_contract()
    adapter = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_failed.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert result.status == "failed"


def test_ducc_socket_drop_with_nonzero_exit_returns_transient(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    # 流式连接中途断开：ducc 以非零码退出且 is_error=True，但错误文本是连接级瞬时问题
    # （"socket connection was closed unexpectedly"）。必须判 transient 让调度器退避重试，
    # 而不是被 returncode!=0 抢先判 failed 连坐——回归 socket 断连误判 atom_failed 的根因。
    contract = make_contract()
    adapter = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_socket_drop.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert_atom_result_shape(result)
    assert result.status == "transient"
    assert result.error is not None and "socket connection was closed" in result.error.lower()


def test_is_transient_error_classifier_is_conservative() -> None:
    assert is_transient_error(http_status=503)
    assert is_transient_error(http_status=429)
    assert is_transient_error(message="503 All credentials exhausted; usually temporary")
    assert is_transient_error(message="rate limit exceeded, retry later")
    # 连接级闪断（流式 socket 断开 / 连接重置）也是基础设施瞬时问题。
    assert is_transient_error(message="The socket connection was closed unexpectedly.")
    assert is_transient_error(message="read ECONNRESET")
    assert is_transient_error(message="socket hang up")
    # 网关读取上游推理服务响应体失败 / 上游建连失败：上游流式响应在网关侧被截断或上游
    # 不可达，是基础设施瞬时问题（典型 ducc 网关报文 "API Error: Upstream body read failed"）。
    assert is_transient_error(message="API Error: Upstream body read failed")
    assert is_transient_error(message="upstream connect error or disconnect/reset before headers")
    # 真任务失败 / 客户端错误不应被误判为瞬时。
    assert not is_transient_error(http_status=400)
    assert not is_transient_error(message="3 unit tests failed")
    assert not is_transient_error()


def test_transient_error_from_events_extracts_nested_status_and_message(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                "{not json",
                json.dumps({"type": "agent_message", "message": "3 unit tests failed"}),
                json.dumps(
                    {
                        "type": "turn.failed",
                        "error": {
                            "status": "503",
                            "message": "Service unavailable, retry later",
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    is_transient, message = transient_error_from_events(str(events_path))

    assert is_transient is True
    assert message is not None and "Service unavailable" in message
    assert transient_error_from_events(str(tmp_path / "missing.jsonl")) == (False, None)


def test_ducc_timeout_returns_timeout_without_waiting_for_sleep(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract(limits={"max_cost": 2.0, "timeout_sec": 1, "max_internal_turns": 30})
    adapter = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_timeout.py"))

    started = time.monotonic()
    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])
    elapsed = time.monotonic() - started

    assert_atom_result_shape(result)
    assert result.status == "timeout"
    assert elapsed < 5


def test_run_with_timeout_allows_runtime_past_idle_threshold_when_workspace_changes(tmp_path: Path) -> None:
    script = (
        "import pathlib, sys, time\n"
        "path = pathlib.Path(sys.argv[1]) / 'progress.txt'\n"
        "path.write_text('start', encoding='utf-8')\n"
        "for index in range(3):\n"
        "    time.sleep(0.6)\n"
        "    path.write_text(str(index), encoding='utf-8')\n"
        "print('done')\n"
    )

    completed = run_with_timeout(
        [sys.executable, "-c", script, str(tmp_path)],
        timeout_sec=1,
        cwd=str(tmp_path),
        watch_dir=str(tmp_path),
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "done"


def test_run_with_timeout_times_out_when_workspace_is_idle(tmp_path: Path) -> None:
    with pytest.raises(AdapterTimeout, match=r"idle: no workspace file changes"):
        run_with_timeout(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_sec=1,
            cwd=str(tmp_path),
            watch_dir=str(tmp_path),
        )


def test_run_with_timeout_drains_stdout_while_polling(tmp_path: Path) -> None:
    script = "import sys; sys.stdout.write('x' * 200000); sys.stdout.flush()\n"

    completed = run_with_timeout(
        [sys.executable, "-c", script],
        timeout_sec=1,
        cwd=str(tmp_path),
        watch_dir=str(tmp_path),
    )

    assert completed.returncode == 0
    assert len(completed.stdout) == 200000


def test_find_live_events_path_extracts_marker_from_stderr_chunks() -> None:
    assert (
        _find_live_events_path(
            [
                "before\n[codex] live_events_",
                "path=/tmp/codex-events.jsonl\n",
                "after\n",
            ]
        )
        == "/tmp/codex-events.jsonl"
    )
    assert _find_live_events_path(["before\n", "after\n"]) is None


def test_tail_last_activity_reads_latest_supported_jsonl_event(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started"}),
                "{not valid json",
                json.dumps(
                    {
                        "type": "item.started",
                        "item": {"type": "command_execution", "command": "pytest -q"},
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "all tests passed"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert _tail_last_activity(str(events_path)) == "message: all tests passed"


def test_tail_last_activity_returns_none_for_empty_or_missing_files(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")

    assert _tail_last_activity(str(empty_path)) is None
    assert _tail_last_activity(str(tmp_path / "missing.jsonl")) is None


def test_run_with_timeout_attaches_last_activity_without_breaking_completed_process(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    script = (
        "import json, pathlib, sys\n"
        "events_path = pathlib.Path(sys.argv[1])\n"
        "sys.stderr.write(f'[codex] live_events_path={events_path}\\n')\n"
        "sys.stderr.flush()\n"
        "with events_path.open('w', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps({'type': 'item.started', 'item': {'type': 'command_execution', 'command': 'pytest -q'}}) + '\\n')\n"
        "    handle.write(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'final answer'}}) + '\\n')\n"
        "print('done')\n"
    )

    completed = run_with_timeout(
        [sys.executable, "-c", script, str(events_path)],
        timeout_sec=3,
        cwd=str(tmp_path),
        watch_dir=str(tmp_path),
    )

    assert isinstance(completed, subprocess.CompletedProcess)
    assert completed.returncode == 0
    assert completed.stdout.strip() == "done"
    assert getattr(completed, "last_activity") == "message: final answer"


def test_build_output_directive_for_writer(make_contract: Callable[..., AtomContract]) -> None:
    contract = make_contract(output_file="design.md", logical_role="implementer")

    directive = build_output_directive(contract)

    assert "design.md" in directive
    assert "写入文件" in directive
    assert "JSON" not in directive
    assert "blocking_findings" not in directive


def test_build_output_directive_for_reviewer(make_contract: Callable[..., AtomContract]) -> None:
    contract = make_contract(output_file="verdict.json", logical_role="reviewer")

    directive = build_output_directive(contract)

    assert "verdict.json" in directive
    assert "JSON" in directive
    assert "passed" in directive
    assert "criteria" in directive
    assert "blocking_findings" in directive


def test_build_output_directive_empty_when_no_output_file(make_contract: Callable[..., AtomContract]) -> None:
    contract = make_contract(output_file="", logical_role="implementer")

    assert build_output_directive(contract) == ""


def test_augment_task_prepends_directive(make_contract: Callable[..., AtomContract]) -> None:
    contract = make_contract(task="write the design", output_file="design.md", logical_role="implementer")
    directive = build_output_directive(contract)

    augmented = augment_task_with_output_directive(contract)

    assert augmented.startswith(directive)
    assert "write the design" in augmented
    assert augment_task_with_output_directive(make_contract(task="plain task", output_file="")) == "plain task"


def test_context_files_directive_lists_upstream_files() -> None:
    from atomic_agents.models import AtomContract

    contract = AtomContract(
        task="t",
        inputs=[],
        context_files=["plan.md", "research.md"],
        workspace="/ws",
        read_only=False,
        write_scope=[],
        required_capabilities=[],
        status="success",
        result="",
        artifacts=[],
        output_file="",
        output_schema_ref=None,
        handoff={"completed": [], "pending": [], "decisions": [], "risks": []},
        consult=None,
        atom_id="a",
        correlation_id="r",
        logical_role="implementer",
        resolved_runner=None,
        session_id=None,
        hop_count=1,
        limits={"max_cost": 1.0, "timeout_sec": 60, "max_internal_turns": 10},
        cost=0.0,
        duration_sec=0.0,
        timestamps={"started_at": "t", "finished_at": "t"},
    )

    directive = build_context_files_directive(contract)
    assert "plan.md" in directive
    assert "research.md" in directive
    assert "上游产出" in directive


def test_context_files_directive_empty_without_context(make_contract: Callable[..., AtomContract]) -> None:
    assert build_context_files_directive(make_contract(context_files=[])) == ""


def test_both_runners_receive_uniform_context_file_directive(make_contract: Callable[..., AtomContract]) -> None:
    # codex 与 ducc 对同一契约必须看到一致的「优先阅读上游文件」指引（消除 -f vs --add-dir
    # 的可见性不对称）。两者的 augmented prompt 都应内嵌相同的 context-files directive。
    contract = make_contract(
        task="merge upstream",
        output_file="summary.md",
        context_files=["a.md", "b.md"],
        logical_role="synthesizer",
    )

    augmented = augment_task_with_output_directive(contract)
    context_directive = build_context_files_directive(contract)

    assert context_directive  # non-empty
    assert context_directive in augmented
    assert "a.md" in augmented and "b.md" in augmented


def test_codex_adapter_passes_augmented_prompt_to_runner(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
    monkeypatch,
) -> None:
    contract = make_contract(task="original task", output_file="design.md", logical_role="implementer")
    captured: dict[str, str | None] = {}

    def fake_run_with_timeout(
        cmd: list[str],
        timeout_sec: int,
        cwd: str | None = None,
        stdin_text: str | None = None,
        watch_dir: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cmd, timeout_sec
        captured["stdin_text"] = stdin_text
        captured["watch_dir"] = watch_dir
        output_path = Path(cwd or tmp_path) / "codex-output.md"
        output_path.write_text("codex ok\n", encoding="utf-8")
        stdout = f"session_id=sess-fake\noutput_path={output_path}\n"
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(codex_module, "run_with_timeout", fake_run_with_timeout)

    adapter = CodexAdapter(ask_codex_path="codex")
    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert result.status == "success"
    assert captured["stdin_text"] == augment_task_with_output_directive(contract)
    assert captured["stdin_text"].startswith(build_output_directive(contract))
    assert "design.md" in (captured["stdin_text"] or "")
    assert "original task" in (captured["stdin_text"] or "")
    assert captured["watch_dir"] == contract.workspace


def test_ducc_adapter_passes_augmented_prompt_to_runner(
    make_contract: Callable[..., AtomContract],
    monkeypatch,
) -> None:
    contract = make_contract(task="review this", output_file="verdict.json", logical_role="reviewer")
    captured: dict[str, str | None] = {}

    def fake_run_with_timeout(
        cmd: list[str],
        timeout_sec: int,
        cwd: str | None = None,
        stdin_text: str | None = None,
        watch_dir: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cmd, timeout_sec, cwd
        captured["stdin_text"] = stdin_text
        captured["watch_dir"] = watch_dir
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": "ok",
                "session_id": "sess-fake",
                "total_cost_usd": 0.07,
                "is_error": False,
            }
        )
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(ducc_module, "run_with_timeout", fake_run_with_timeout)

    adapter = DuccAdapter(ducc_path="ducc")
    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert result.status == "success"
    assert captured["stdin_text"] == augment_task_with_output_directive(contract)
    assert captured["stdin_text"].startswith(build_output_directive(contract))
    assert captured["watch_dir"] == contract.workspace
    assert "verdict.json" in (captured["stdin_text"] or "")
    assert "blocking_findings" in (captured["stdin_text"] or "")
    assert "review this" in (captured["stdin_text"] or "")


def test_successful_adapters_return_same_atom_result_fields(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    codex_contract = make_contract()
    ducc_contract = make_contract()
    codex = CodexAdapter(ask_codex_path=make_fake_executable(tmp_path, "fake_codex_success.py"))
    ducc = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_success.py"))

    codex_result = codex.invoke(codex_contract, timeout_sec=codex_contract.limits["timeout_sec"])
    ducc_result = ducc.invoke(ducc_contract, timeout_sec=ducc_contract.limits["timeout_sec"])

    expected_fields = [field.name for field in fields(AtomResult)]
    assert [field.name for field in fields(codex_result)] == expected_fields
    assert [field.name for field in fields(ducc_result)] == expected_fields


def test_feature_profiles_record_runner_capability_differences(tmp_path: Path) -> None:
    codex = CodexAdapter(ask_codex_path=make_fake_executable(tmp_path, "fake_codex_success.py"))
    ducc = DuccAdapter(ducc_path=make_fake_executable(tmp_path, "fake_ducc_success.py"))

    assert codex.feature_profile.supports_raw_events is True
    assert ducc.feature_profile.supports_raw_events is False
    assert codex.feature_profile.supports_internal_turn_count is True
    assert ducc.feature_profile.supports_internal_turn_count is False
