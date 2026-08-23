from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from atomic_agents.adapters.codex import CodexAdapter, _parse_raw_events, _tokens_to_usd
from atomic_agents.models import AtomContract


FAKES_DIR = Path(__file__).parent / "fakes"


def make_fake_executable(tmp_path: Path, fake_name: str) -> str:
    source = FAKES_DIR / fake_name
    target = tmp_path / fake_name
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    os.chmod(target, 0o755)
    return str(target)


def write_events(path: Path, events: list[object]) -> None:
    path.write_text(
        "\n".join(event if isinstance(event, str) else json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def test_tokens_to_usd_converts_token_totals_to_usd() -> None:
    assert _tokens_to_usd(1_000_000, 10.0) == 10.0
    assert _tokens_to_usd(100_000, 10.0) == 1.0
    assert _tokens_to_usd(27138, 10.0) == pytest.approx(0.27138)
    assert _tokens_to_usd(0, 10.0) == 0.0
    assert _tokens_to_usd(-5, 10.0) == 0.0
    assert _tokens_to_usd(1_000_000, 15.0) == 15.0
    assert _tokens_to_usd(1_000_000, 0.0) == 0.0


def test_parse_raw_events_sums_usage_token_fields(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    write_events(
        events_path,
        [
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 50,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 5,
                },
            }
        ],
    )

    token_total, _internal_turn_count = _parse_raw_events(str(events_path))

    assert token_total == 175


def test_parse_raw_events_sums_multiple_events(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    write_events(
        events_path,
        [
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 50,
                    "output_tokens": 20,
                    "reasoning_output_tokens": 5,
                },
            },
            {"type": "item.completed", "usage": {"output_tokens": 30}},
        ],
    )

    token_total, _internal_turn_count = _parse_raw_events(str(events_path))

    assert token_total == 205


@pytest.mark.parametrize("raw_events_path", ["/nonexistent/path", None])
def test_parse_raw_events_missing_path_returns_zero(raw_events_path: str | None) -> None:
    assert _parse_raw_events(raw_events_path) == (0.0, 0)


def test_parse_raw_events_empty_file_returns_zero(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    assert _parse_raw_events(str(events_path)) == (0.0, 0)


def test_parse_raw_events_skips_bad_json_lines(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    write_events(
        events_path,
        [
            "not json",
            {"type": "item.completed", "usage": {"output_tokens": 30}},
        ],
    )

    token_total, _internal_turn_count = _parse_raw_events(str(events_path))

    assert token_total == 30


def test_codex_adapter_reports_default_usd_cost(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract()
    adapter = CodexAdapter(ask_codex_path=make_fake_executable(tmp_path, "fake_codex_success.py"))

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert result.status == "success"
    assert result.cost == pytest.approx(21 / 1_000_000.0 * 10.0)


def test_codex_adapter_uses_custom_usd_per_mtok(
    tmp_path: Path,
    make_contract: Callable[..., AtomContract],
) -> None:
    contract = make_contract()
    adapter = CodexAdapter(
        ask_codex_path=make_fake_executable(tmp_path, "fake_codex_success.py"),
        usd_per_mtok=1000.0,
    )

    result = adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])

    assert result.status == "success"
    assert result.cost == pytest.approx(21 / 1_000_000.0 * 1000.0)
    assert result.cost == pytest.approx(0.021)
