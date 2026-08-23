from __future__ import annotations

import json
from typing import Any

import pytest

from atomic_agents.linter import Violation, lint_skeleton
from atomic_agents.llm_compiler import (
    LLMCompiler,
    SkeletonParseError,
    SkeletonValidationError,
    extract_skeleton_json,
)
from atomic_agents.meta import meta_compile
from atomic_agents.models import AtomContract, AtomResult, Edge, Skeleton, SkeletonNode
from atomic_agents.templates import TEMPLATES


def test_extract_skeleton_json_from_plain_json() -> None:
    assert extract_skeleton_json('{"name": "x"}') == {"name": "x"}


def test_extract_skeleton_json_from_json_fence() -> None:
    assert extract_skeleton_json('```json\n{"name": "x"}\n```') == {"name": "x"}


def test_extract_skeleton_json_from_untyped_fence() -> None:
    assert extract_skeleton_json('```\n{"name": "x"}\n```') == {"name": "x"}


def test_extract_skeleton_json_from_surrounding_text() -> None:
    text = '解释文字\n{"name": "x", "nested": {"ok": true}}\n结束'

    assert extract_skeleton_json(text) == {"name": "x", "nested": {"ok": True}}


@pytest.mark.parametrize("text", ["完全没有 JSON", '{"name": ', '```json\n{"name": \n```'])
def test_extract_skeleton_json_raises_for_missing_or_bad_json(text: str) -> None:
    with pytest.raises(SkeletonParseError):
        extract_skeleton_json(text)


def test_extract_skeleton_json_raises_for_array() -> None:
    with pytest.raises(SkeletonParseError):
        extract_skeleton_json('[{"name": "x"}]')


def test_generate_returns_valid_skeleton_from_plain_json() -> None:
    adapter = FakeAdapter([_json(_valid_template_dict())])
    compiler = LLMCompiler(adapter, workspace="/tmp/compiler-test", timeout_sec=123)

    skeleton = compiler.generate("x")

    assert isinstance(skeleton, Skeleton)
    assert lint_skeleton(skeleton).ok is True
    assert adapter.calls[0].read_only is True
    assert adapter.calls[0].write_scope == []
    assert adapter.calls[0].task
    assert adapter.timeouts == [123]


def test_generate_raises_parse_error_when_adapter_fails() -> None:
    compiler = LLMCompiler(FakeAdapter([_json(_valid_template_dict())], status="failed"))

    with pytest.raises(SkeletonParseError):
        compiler.generate("x")


def test_generate_raises_validation_error_for_schema_invalid_json() -> None:
    compiler = LLMCompiler(FakeAdapter([json.dumps({"name": "x"})]))

    with pytest.raises(SkeletonValidationError):
        compiler.generate("x")


def test_generate_accepts_fenced_valid_skeleton() -> None:
    compiler = LLMCompiler(FakeAdapter([f"```json\n{_json(_valid_template_dict())}\n```"]))

    skeleton = compiler.generate("x")

    assert lint_skeleton(skeleton).ok is True


def test_repair_returns_valid_skeleton() -> None:
    compiler = LLMCompiler(FakeAdapter([_json(_valid_template_dict())]))
    bad_skeleton = _writer_without_reviewer()
    violations = [
        Violation(
            code="writer_without_reviewer_downstream",
            message="writer node impl has no downstream reviewer",
            node_id="impl",
        )
    ]

    repaired = compiler.repair(bad_skeleton, violations)

    assert isinstance(repaired, Skeleton)
    assert lint_skeleton(repaired).ok is True
    assert "违规列表" in compiler.adapter.calls[0].task


def test_llm_compiler_meta_compile_repair_loop() -> None:
    adapter = FakeAdapter([_json(_writer_without_reviewer().to_dict()), _json(_valid_template_dict())])
    compiler = LLMCompiler(adapter)

    plan = meta_compile("把仓库迁移到全新的自研构建系统", "ducc", compiler)

    assert plan.source == "free"
    assert plan.lint_repaired is True
    assert lint_skeleton(plan.skeleton).ok is True
    assert len(adapter.calls) == 2
    assert "用户需求：把仓库迁移到全新的自研构建系统" in adapter.calls[0].task
    assert "违规列表" in adapter.calls[1].task


class FakeAdapter:
    def __init__(self, responses: list[str] | str, status: str = "success") -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.status = status
        self.calls: list[AtomContract] = []
        self.timeouts: list[int] = []

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        self.calls.append(contract)
        self.timeouts.append(timeout_sec)
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return AtomResult(
            status=self.status,  # type: ignore[arg-type]
            result=self.responses[index],
            artifacts=[],
            session_id=None,
            cost=0.1,
            duration_sec=1.0,
            raw_events_path=None,
            error=None if self.status == "success" else "failed by fake adapter",
            output_file="",
            output_sha256=None,
        )


def _valid_template_dict() -> dict[str, Any]:
    return TEMPLATES[0].build("x").to_dict()


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _writer_without_reviewer() -> Skeleton:
    return _make_skeleton([_make_node("impl", write_scope=["src/a.py"], read_only=False)])


def _make_node(
    id: str,
    *,
    role: str = "worker",
    deps: list[str] | None = None,
    write_scope: list[str] | None = None,
    read_only: bool = True,
    reviewer_criteria: list[str] | None = None,
) -> SkeletonNode:
    writes = list(write_scope or [])
    return SkeletonNode(
        id=id,
        role=role,
        task=f"Run {id}",
        depends_on=list(deps or []),
        inputs=[],
        write_scope=writes,
        read_only=read_only,
        required_capabilities=["write_files"] if writes else [],
        reviewer_criteria=list(reviewer_criteria or []),
        output_file=writes[0] if writes else "",
    )


def _make_skeleton(nodes: list[SkeletonNode], *, edges: list[Edge] | None = None) -> Skeleton:
    return Skeleton(
        name="llm-compiler-test",
        version=1,
        nodes=nodes,
        edges=edges if edges is not None else _edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": 2, "max_total_cost": 10.0},
        irreversible_ops=[],
    )


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    edges: list[Edge] = []
    for node in nodes:
        for dependency in node.depends_on:
            edges.append({"from": dependency, "to": node.id})
    return edges
