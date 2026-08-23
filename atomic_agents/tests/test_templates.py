from __future__ import annotations

from atomic_agents.linter import lint_skeleton
from atomic_agents.models import Edge, Skeleton
from atomic_agents.templates import MockCompiler, SkeletonCompiler, Template, build_from_template, fill_template, match_template
from atomic_agents.validation import validate_skeleton


def test_match_template_by_chinese_and_english_keywords() -> None:
    assert _matched_name("实现一个登录功能") == "plan-impl-review"
    assert _matched_name("ship a new feature") == "plan-impl-review"
    assert _matched_name("调研两种方案") == "scatter-gather"
    assert _matched_name("research possible approaches") == "scatter-gather"


def test_match_template_none_and_first_match_wins() -> None:
    assert match_template("你好") is None
    assert _matched_name("实现并调研这个功能") == "plan-impl-review"


def test_fill_plan_impl_review_template_builds_valid_skeleton() -> None:
    request = "实现用户偏好保存"
    template = match_template(request)
    assert template is not None

    skeleton = fill_template(template, request)

    assert skeleton.name == "plan-impl-review"
    assert len(skeleton.nodes) == 3
    assert request in _node(skeleton, "plan").task
    assert request in _node(skeleton, "impl").task
    assert request in _node(skeleton, "review").task
    assert _node(skeleton, "plan").write_scope == ["docs/plan.md"]
    assert _node(skeleton, "impl").write_scope == ["src/feature.py"]
    assert _node(skeleton, "review").role == "reviewer"
    assert _node(skeleton, "review").read_only is False
    assert _node(skeleton, "review").write_scope == ["docs/review.md"]
    assert _node(skeleton, "review").reviewer_criteria == ["实现满足需求描述", "无明显回归", "关键路径有测试"]
    _assert_edges_match_dependencies(skeleton)
    _assert_valid_skeleton(skeleton)


def test_fill_scatter_gather_template_builds_valid_skeleton() -> None:
    request = "调研缓存策略"
    template = match_template(request)
    assert template is not None

    skeleton = fill_template(template, request)

    assert skeleton.name == "scatter-gather"
    assert len(skeleton.nodes) == 4
    assert request in _node(skeleton, "explore1").task
    assert request in _node(skeleton, "explore2").task
    assert request in _node(skeleton, "gather").task
    assert _node(skeleton, "explore1").read_only is True
    assert _node(skeleton, "explore2").read_only is True
    assert _node(skeleton, "explore1").write_scope == []
    assert _node(skeleton, "explore2").write_scope == []
    assert _node(skeleton, "gather").write_scope == ["docs/summary.md"]
    assert _node(skeleton, "gather").required_capabilities == ["write_files"]
    assert _node(skeleton, "review").role == "reviewer"
    assert _node(skeleton, "review").read_only is False
    assert _node(skeleton, "review").write_scope == ["docs/review.md"]
    assert _node(skeleton, "review").reviewer_criteria == ["汇总覆盖所有调研输入", "结论有依据"]
    _assert_edges_match_dependencies(skeleton)
    _assert_valid_skeleton(skeleton)


def test_build_from_template_returns_skeleton_or_none() -> None:
    skeleton = build_from_template("fix a bug in parsing")

    assert skeleton is not None
    assert skeleton.name == "plan-impl-review"
    assert build_from_template("你好") is None


def test_mock_compiler_implements_protocol_and_prefers_scripted_skeleton() -> None:
    request = "custom request"
    scripted = fill_template(_matched_template("实现功能"), request)
    compiler: SkeletonCompiler = MockCompiler({request: scripted})

    generated = compiler.generate(request)

    assert generated.to_dict() == scripted.to_dict()
    assert generated is not scripted
    _assert_valid_skeleton(generated)


def test_mock_compiler_falls_back_to_template_or_default_seed() -> None:
    compiler = MockCompiler()

    templated = compiler.generate("research topic")
    defaulted = compiler.generate("你好")

    assert templated.name == "scatter-gather"
    assert defaulted.name == "plan-impl-review"
    assert compiler.calls == ["research topic", "你好"]
    _assert_valid_skeleton(templated)
    _assert_valid_skeleton(defaulted)


def _matched_name(request: str) -> str:
    return _matched_template(request).name


def _matched_template(request: str) -> Template:
    template = match_template(request)
    assert template is not None
    return template


def _node(skeleton: Skeleton, node_id: str):
    return next(node for node in skeleton.nodes if node.id == node_id)


def _assert_edges_match_dependencies(skeleton: Skeleton) -> None:
    expected_edges: list[Edge] = []
    for node in skeleton.nodes:
        for dependency in node.depends_on:
            expected_edges.append({"from": dependency, "to": node.id})

    assert skeleton.edges == expected_edges


def _assert_valid_skeleton(skeleton: Skeleton) -> None:
    result = lint_skeleton(skeleton)
    assert result.ok is True
    assert result.violations == []
    validate_skeleton(skeleton.to_dict())
