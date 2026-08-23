from __future__ import annotations

import argparse
import sys

import pytest

from examples.orchestrations import cli
from examples.orchestrations import common


def test_run_dispatches_template_specific_counts(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_design(request: str, **kwargs):
        captured.update(request=request, **kwargs)
        return object()

    monkeypatch.setattr(cli.design_mod, "design", fake_design)

    result = cli.run(
        "design",
        "request",
        counts={"designer": 4, "researcher": 5},
        role_runners={"designer": ["codex", "ducc"]},
        workspace="/tmp/workspace",
    )

    assert result is not None
    assert captured == {
        "request": "request",
        "n_stances": 4,
        "n_researchers": 5,
        "role_runners": {"designer": ["codex", "ducc"]},
        "workspace": "/tmp/workspace",
    }


def test_main_parses_count_and_role_options_into_run_dispatch(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(template: str, request: str, **kwargs):
        captured.update(template=template, request=request, **kwargs)
        return common.OrchestrationResult(
            template=template,
            request=request,
            workspace="/tmp/workspace",
            succeeded=True,
            artifacts=[],
            final_path=None,
            atom_runner_plan={},
        )

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "atomic-orchestration",
            "design",
            "request",
            "--count",
            "designer=4",
            "--count",
            "researcher=2",
            "--role",
            "designer=codex,ducc",
            "--role",
            "researcher=ducc",
        ],
    )

    assert cli.main() is True
    assert captured["counts"] == {"designer": 4, "researcher": 2}
    assert captured["role_runners"] == {
        "designer": ["codex", "ducc"],
        "researcher": "ducc",
    }
    assert '"succeeded": true' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("template", "role", "expected_kwarg"),
    [
        ("review", "analyst", "n_reviewers"),
        ("arena", "designer", "n_solutions"),
        ("scatter-gather", "explorer", "n_explorers"),
    ],
)
def test_run_dispatches_primary_count_roles(monkeypatch, template, role, expected_kwarg) -> None:
    captured: dict[str, object] = {}

    def fake_template(request: str, *args, **kwargs):
        captured.update(kwargs)
        return object()

    module = {
        "review": cli.review_mod,
        "arena": cli.arena_mod,
        "scatter-gather": cli.sg_mod,
    }[template]
    function_name = {"review": "review", "arena": "arena", "scatter-gather": "scatter_gather"}[template]
    monkeypatch.setattr(module, function_name, fake_template)

    kwargs = {"doc_path": "/tmp/doc.md"} if template == "review" else {}
    cli.run(template, "request", n=4, counts={role: 4}, **kwargs)

    assert captured[expected_kwarg] == 4


def test_count_parser_and_validation_errors() -> None:
    assert cli.parse_count_overrides(["designer=4", "researcher=2"]) == {
        "designer": 4,
        "researcher": 2,
    }
    with pytest.raises(ValueError, match="ROLE=VALUE"):
        cli.parse_count_overrides(["designer"])
    with pytest.raises(ValueError, match="must be an integer"):
        cli.parse_count_overrides(["designer=many"])
    with pytest.raises(ValueError, match="conflicting --count"):
        cli.parse_count_overrides(["designer=2", "designer=3"])
    with pytest.raises(ValueError, match="conflicting counts"):
        cli.run("design", "request", n=2, counts={"designer": 3})
    with pytest.raises(ValueError, match="does not support --count for role 'analyst'"):
        cli.run("design", "request", counts={"analyst": 2})
    with pytest.raises(ValueError, match="must be >= 2"):
        cli.run("arena", "request", counts={"designer": 1})
    with pytest.raises(ValueError, match="does not support --count"):
        cli.run("pipeline", "request", counts={"designer": 2})


def test_role_parser_supports_single_and_round_robin_values() -> None:
    overrides = cli.parse_role_overrides(
        ["implementer=codex", "analyst=codex,ducc"]
    )
    assert overrides == {
        "implementer": "codex",
        "analyst": ["codex", "ducc"],
    }

    nodes = [
        common.write_node(
            node_id=f"review_{index}",
            role="analyst",
            task="review",
            output_file=f"review-{index}.md",
        )
        for index in range(4)
    ]
    assert common.build_atom_runner_plan(nodes, common.merge_role_runners(overrides)) == {
        "review_0": "codex",
        "review_1": "ducc",
        "review_2": "codex",
        "review_3": "ducc",
    }


def test_role_validation_and_legacy_flag_conflicts() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        cli.parse_role_overrides(["architect=codex"])
    with pytest.raises(ValueError, match="unknown runner"):
        cli.parse_role_overrides(["designer=other"])
    with pytest.raises(ValueError, match="conflicting --role"):
        cli.parse_role_overrides(["designer=codex", "designer=ducc"])

    same = argparse.Namespace(
        role=["designer=codex"], designer="codex", reviewer=None, implementer=None
    )
    assert cli._role_runners_from_args(same) == {"designer": "codex"}

    conflict = argparse.Namespace(
        role=["designer=codex"], designer="ducc", reviewer=None, implementer=None
    )
    with pytest.raises(ValueError, match="conflicting runner overrides"):
        cli._role_runners_from_args(conflict)
