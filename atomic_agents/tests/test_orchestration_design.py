from __future__ import annotations

from pathlib import Path
from typing import Any

from atomic_agents.adapters import RunnerFeatureProfile
from atomic_agents.models import AtomContract, AtomResult
from atomic_agents.run import RunResult
from atomic_agents.scheduler import NodeState

from examples.orchestrations import common, design as design_mod


class _WritingAdapter:
    def __init__(self, name: str) -> None:
        self.adapter_version = f"{name}-test"
        self.feature_profile = RunnerFeatureProfile(
            name=name,
            supports_session_resume=False,
            supports_cost_capture=True,
            supports_raw_events=False,
            supports_internal_turn_count=False,
            permission_modes=["workspace-write"],
        )
        self.calls: list[str] = []

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        self.calls.append(contract.atom_id)
        if contract.output_file:
            output_path = Path(contract.workspace) / contract.output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"{self.feature_profile.name}:{contract.atom_id}\n", encoding="utf-8")
        return AtomResult(
            status="success",
            result=f"ok:{contract.atom_id}",
            artifacts=[],
            session_id=None,
            cost=0.0,
            duration_sec=0.0,
            raw_events_path=None,
            error=None,
            output_file=contract.output_file,
            output_sha256=None,
        )


def test_run_orchestration_explicit_atom_runners_override_role_plan(monkeypatch, tmp_path: Path) -> None:
    adapters = {"codex": _WritingAdapter("codex"), "ducc": _WritingAdapter("ducc")}
    monkeypatch.setattr(common, "get_adapter", lambda runner_name: adapters[runner_name])
    skeleton = common.make_skeleton(
        "explicit-runner-test",
        [
            common.write_node(
                node_id="a",
                role="designer",
                task="write a",
                output_file="a.md",
            ),
            common.write_node(
                node_id="b",
                role="designer",
                task="write b",
                output_file="b.md",
                depends_on=["a"],
                inputs=[{"from": "a", "field": "output_file"}],
            ),
        ],
        max_repair_attempts=0,
    )

    result = common.run_orchestration(
        "explicit-runner-test",
        "request",
        skeleton,
        {"designer": "ducc"},
        str(tmp_path),
        final_file="b.md",
        explicit_atom_runners={"b": "codex"},
    )

    assert result.succeeded is True
    assert result.atom_runner_plan == {"a": "ducc", "b": "codex"}
    assert adapters["ducc"].calls == ["a"]
    assert adapters["codex"].calls == ["b"]


def test_design_runs_research_then_design_skeleton_and_merges_runner_plan(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_orchestration(
        template: str,
        request: str,
        skeleton,
        role_runners: dict[str, object],
        workspace: str,
        *,
        final_file: str | None = None,
        extra: dict | None = None,
        explicit_atom_runners: dict[str, str] | None = None,
    ) -> common.OrchestrationResult:
        calls.append(
            {
                "template": template,
                "nodes": [node.id for node in skeleton.nodes],
                "workspace": workspace,
                "final_file": final_file,
                "role_runners": dict(role_runners),
                "explicit_atom_runners": dict(explicit_atom_runners or {}),
            }
        )
        if template == "design-research":
            (Path(workspace) / "research.md").write_text("research", encoding="utf-8")
            final_path = str(Path(workspace) / "research.md")
            run_result = _run_result({node.id: "succeeded" for node in skeleton.nodes}, True)
        else:
            (Path(workspace) / "final-solution.md").write_text("final", encoding="utf-8")
            final_path = str(Path(workspace) / "final-solution.md")
            run_result = _run_result({node.id: "succeeded" for node in skeleton.nodes}, True)
        return common.OrchestrationResult(
            template=template,
            request=request,
            workspace=workspace,
            succeeded=True,
            artifacts=sorted(path.name for path in Path(workspace).glob("*.md")),
            final_path=final_path,
            atom_runner_plan=dict(explicit_atom_runners or {}),
            run_result=run_result,
            extra=extra or {},
        )

    monkeypatch.setattr(design_mod, "run_orchestration", fake_run_orchestration)
    monkeypatch.setattr(
        design_mod,
        "_select_stances",
        lambda request, workspace, n_stances, runner="ducc": [
            {"stance_name": f"stance-{i}", "focus": "focus", "prompt_hint": "hint"}
            for i in range(n_stances)
        ],
    )

    result = design_mod.design("design request", n_stances=3, workspace=str(tmp_path))

    assert result.succeeded is True
    assert result.final_path == str(tmp_path / "final-solution.md")
    assert [call["template"] for call in calls] == ["design-research", "design"]
    assert calls[0]["workspace"] == calls[1]["workspace"] == str(tmp_path)
    assert calls[0]["nodes"] == ["research_0", "research_1", "research_2", "research_merge"]
    assert calls[0]["explicit_atom_runners"] == {
        "research_0": "codex",
        "research_1": "codex",
        "research_2": "ducc",
        "research_merge": "ducc",
    }
    assert calls[1]["explicit_atom_runners"]["design_0_v1"] == "ducc"
    assert calls[1]["explicit_atom_runners"]["design_1_v2"] == "codex"
    assert calls[1]["explicit_atom_runners"]["design_2_v3"] == "codex"
    assert calls[1]["explicit_atom_runners"]["redteam_r1"] == "codex"
    assert calls[1]["explicit_atom_runners"]["redteam_r2"] == "ducc"
    assert calls[1]["explicit_atom_runners"]["synth"] == "ducc"
    assert result.atom_runner_plan["research_0"] == "codex"
    assert result.atom_runner_plan["design_2_v3"] == "codex"
    assert result.extra["research_succeeded"] is True


def test_build_research_skeleton_uses_three_parallel_researchers_and_merge() -> None:
    skeleton = design_mod.build_research_skeleton("request")

    assert skeleton.name == "design-research"
    assert [node.id for node in skeleton.nodes] == ["research_0", "research_1", "research_2", "research_merge"]
    assert [node.output_file for node in skeleton.nodes[:3]] == ["research-0.md", "research-1.md", "research-2.md"]
    assert all(node.role == "researcher" for node in skeleton.nodes)
    assert all(node.depends_on == [] for node in skeleton.nodes[:3])
    assert skeleton.nodes[3].output_file == "research.md"
    assert skeleton.nodes[3].depends_on == ["research_0", "research_1", "research_2"]
    assert skeleton.nodes[3].inputs == [
        {"from": "research_0", "field": "output_file"},
        {"from": "research_1", "field": "output_file"},
        {"from": "research_2", "field": "output_file"},
    ]


def test_build_research_skeleton_supports_configurable_researcher_count() -> None:
    skeleton = design_mod.build_research_skeleton("request", n_researchers=5)

    assert [node.id for node in skeleton.nodes] == [
        "research_0", "research_1", "research_2", "research_3", "research_4", "research_merge"
    ]
    assert [node.output_file for node in skeleton.nodes[:5]] == [
        "research-0.md", "research-1.md", "research-2.md", "research-3.md", "research-4.md"
    ]
    assert skeleton.nodes[-1].depends_on == [
        "research_0", "research_1", "research_2", "research_3", "research_4"
    ]
    assert set(design_mod.research_runner_plan(5)) == {
        "research_0", "research_1", "research_2", "research_3", "research_4", "research_merge"
    }


def test_design_explicit_role_overrides_remove_matching_builtin_atom_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_orchestration(
        template: str,
        request: str,
        skeleton,
        role_runners: dict[str, object],
        workspace: str,
        *,
        final_file: str | None = None,
        extra: dict | None = None,
        explicit_atom_runners: dict[str, str] | None = None,
    ) -> common.OrchestrationResult:
        explicit = dict(explicit_atom_runners or {})
        plan = common.build_atom_runner_plan(skeleton.nodes, role_runners)
        plan.update(explicit)
        calls.append({"template": template, "explicit": explicit, "plan": plan})
        output = "research.md" if template == "design-research" else "final-solution.md"
        (Path(workspace) / output).write_text("output", encoding="utf-8")
        return common.OrchestrationResult(
            template=template,
            request=request,
            workspace=workspace,
            succeeded=True,
            artifacts=[output],
            final_path=str(Path(workspace) / output),
            atom_runner_plan=plan,
            run_result=_run_result({node.id: "succeeded" for node in skeleton.nodes}, True),
            extra=extra or {},
        )

    monkeypatch.setattr(design_mod, "run_orchestration", fake_run_orchestration)
    monkeypatch.setattr(
        design_mod,
        "_select_stances",
        lambda request, workspace, n_stances, runner="ducc": [
            {"stance_name": f"stance-{i}", "focus": "focus", "prompt_hint": "hint"}
            for i in range(n_stances)
        ],
    )

    result = design_mod.design(
        "request",
        n_stances=2,
        n_researchers=4,
        role_runners={
            "researcher": ["ducc", "codex"],
            "designer": "codex",
            "critic": "ducc",
            "synthesizer": "codex",
        },
        workspace=str(tmp_path),
    )

    assert not any(atom_id.startswith("research_") for atom_id in calls[0]["explicit"])
    assert calls[0]["plan"] == {
        "research_0": "ducc",
        "research_1": "codex",
        "research_2": "ducc",
        "research_3": "codex",
        "research_merge": "ducc",
    }
    assert not any(atom_id.startswith("design_") for atom_id in calls[1]["explicit"])
    assert not any(atom_id.startswith(("redteam_", "usability_")) for atom_id in calls[1]["explicit"])
    assert "synth" not in calls[1]["explicit"]
    assert all(
        runner == "codex"
        for atom_id, runner in result.atom_runner_plan.items()
        if atom_id.startswith("design_")
    )
    assert result.atom_runner_plan["redteam_r1"] == "ducc"
    assert result.atom_runner_plan["synth"] == "codex"


def test_design_skeleton_surfaces_research_report_to_all_designers() -> None:
    stances = [
        {"stance_name": "speed", "focus": "fast", "prompt_hint": ""},
        {"stance_name": "safe", "focus": "robust", "prompt_hint": ""},
    ]
    skeleton = design_mod.build_skeleton("request", stances, "research.md")

    design_nodes = [node for node in skeleton.nodes if node.id.startswith("design_")]

    assert [node.id for node in design_nodes] == [
        "design_0_v1",
        "design_1_v1",
        "design_0_v2",
        "design_1_v2",
        "design_0_v3",
        "design_1_v3",
    ]
    assert all(node.context_files == ["research.md"] for node in design_nodes)


def test_design_skeleton_runs_redteam_and_usability_review_in_parallel_each_round() -> None:
    # 红队(安全/并发/资源类)与可用性/可维护性评审专家并行独立质询，互不依赖、
    # 互不可见彼此产出，但同批下游设计者都要吸收两组质询。
    stances = [
        {"stance_name": "speed", "focus": "fast", "prompt_hint": ""},
        {"stance_name": "safe", "focus": "robust", "prompt_hint": ""},
    ]
    skeleton = design_mod.build_skeleton("request", stances, "research.md")
    nodes_by_id = {node.id: node for node in skeleton.nodes}

    for round_no, (redteam_id, usability_id, upstream_v) in enumerate(
        [("redteam_r1", "usability_r1", "v1"), ("redteam_r2", "usability_r2", "v2")], start=1
    ):
        redteam = nodes_by_id[redteam_id]
        usability = nodes_by_id[usability_id]
        upstream_deps = [f"design_{i}_{upstream_v}" for i in range(len(stances))]

        # 两个 critic 节点读同一批上游设计，互不依赖彼此。
        assert redteam.depends_on == upstream_deps
        assert usability.depends_on == upstream_deps
        assert redteam.output_file != usability.output_file
        assert redteam_id not in usability.depends_on
        assert usability_id not in redteam.depends_on

    # 每轮之后的下一版设计者必须同时依赖两个质询节点，不能只接红队或只接可用性。
    for i in range(len(stances)):
        v2_node = nodes_by_id[f"design_{i}_v2"]
        assert set(v2_node.depends_on) == {"redteam_r1", "usability_r1"}
        v3_node = nodes_by_id[f"design_{i}_v3"]
        assert set(v3_node.depends_on) == {"redteam_r2", "usability_r2"}

    # 最终合成必须同时依赖第二轮的两个质询节点。
    synth = nodes_by_id["synth"]
    assert "redteam_r2" in synth.depends_on
    assert "usability_r2" in synth.depends_on


def test_design_runner_plan_covers_both_critic_nodes_per_round() -> None:
    stances = [
        {"stance_name": "speed", "focus": "fast", "prompt_hint": ""},
        {"stance_name": "safe", "focus": "robust", "prompt_hint": ""},
    ]
    plan = design_mod.design_runner_plan(stances)

    for node_id in ("redteam_r1", "usability_r1", "redteam_r2", "usability_r2"):
        assert node_id in plan
    # 同一轮两个视角必须分配不同 runner，避免同一个 runner 连续跑两个角色。
    assert plan["redteam_r1"] != plan["usability_r1"]
    assert plan["redteam_r2"] != plan["usability_r2"]


def _run_result(statuses: dict[str, str], succeeded: bool) -> RunResult:
    return RunResult(
        run_id="run-test",
        states={
            node_id: NodeState(node_id=node_id, status=status)  # type: ignore[arg-type]
            for node_id, status in statuses.items()
        },
        succeeded=succeeded,
        lock_dir="locks",
    )
