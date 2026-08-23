"""Data models for the atomic-agents P0 contract surface."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal, Mapping, Optional, TypeVar, TypedDict, Union, get_args, get_origin, get_type_hints


Status = Literal["success", "failed", "blocked", "timeout", "transient"]


InputRef = TypedDict("InputRef", {"from": str, "field": str})
Artifact = TypedDict("Artifact", {"path": str, "type": str, "sha256": str})
Edge = TypedDict("Edge", {"from": str, "to": str})


class Handoff(TypedDict):
    completed: list[str]
    pending: list[str]
    decisions: list[str]
    risks: list[str]


class ResolvedRunner(TypedDict):
    runner: str
    model: str
    permission: str


class AtomLimits(TypedDict):
    max_cost: float
    timeout_sec: int
    max_internal_turns: int


class Timestamps(TypedDict):
    started_at: str
    finished_at: str


class RunLimits(TypedDict):
    max_repair_attempts: int
    max_total_cost: float


class Reviewer(TypedDict):
    node: str
    criteria: list[str]


class ApprovalBudget(TypedDict):
    max_total_cost: float
    per_atom_timeout_sec: int


class ReviewerEvidence(TypedDict):
    criterion: str
    verdict: str
    evidence: str


T = TypeVar("T", bound="DictSerializable")


class DictSerializable:
    """Small recursive dataclass serializer used by the public models."""

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for field in fields(cls):
            if field.name in data:
                kwargs[field.name] = _from_plain(data[field.name], hints.get(field.name, Any))
        return cls(**kwargs)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _to_plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def _from_plain(value: Any, target_type: Any) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin in (Union, getattr(__import__("types"), "UnionType", object)):
        non_null_args = [arg for arg in args if arg is not type(None)]
        if value is None or not non_null_args:
            return value
        return _from_plain(value, non_null_args[0])

    if origin is list:
        item_type = args[0] if args else Any
        return [_from_plain(item, item_type) for item in value]

    if origin is dict:
        value_type = args[1] if len(args) == 2 else Any
        return {key: _from_plain(item, value_type) for key, item in value.items()}

    if _is_serializable_type(target_type) and isinstance(value, Mapping):
        return target_type.from_dict(value)

    return value


def _is_serializable_type(target_type: Any) -> bool:
    try:
        return isinstance(target_type, type) and issubclass(target_type, DictSerializable)
    except TypeError:
        return False


@dataclass(kw_only=True)
class AtomContract(DictSerializable):
    task: str
    inputs: list[InputRef]
    context_files: list[str]
    workspace: str
    read_only: bool
    write_scope: list[str]
    required_capabilities: list[str]
    status: Status
    result: str
    artifacts: list[Artifact]
    output_file: str
    output_schema_ref: Optional[str] = None
    handoff: Handoff
    consult: Optional[str]
    atom_id: str
    correlation_id: str
    logical_role: str
    resolved_runner: Optional[ResolvedRunner]
    session_id: Optional[str]
    hop_count: int
    limits: AtomLimits
    cost: float
    duration_sec: float
    timestamps: Timestamps


@dataclass(kw_only=True)
class SkeletonNode(DictSerializable):
    id: str
    role: str
    task: str
    depends_on: list[str]
    inputs: list[InputRef]
    write_scope: list[str]
    read_only: bool
    required_capabilities: list[str]
    reviewer_criteria: list[str]
    output_file: str
    context_files: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class Skeleton(DictSerializable):
    name: str
    version: int
    nodes: list[SkeletonNode]
    edges: list[Edge]
    run_limits: RunLimits
    irreversible_ops: list[str]


@dataclass(kw_only=True)
class ApprovalSummary(DictSerializable):
    task_restated: str
    stages: list[str]
    files_may_change: list[str]
    reviewers: list[Reviewer]
    budget: ApprovalBudget
    stop_points: list[str]
    irreversible_ops: list[str]
    risk_flags: list[str]
    editable_hints: str


@dataclass(kw_only=True)
class StopReport(DictSerializable):
    reason: str
    failed_atom: str
    attempts: int
    reviewer_evidence: list[ReviewerEvidence]
    files_changed: list[str]
    cost_consumed: float
    likely_causes: list[str]
    options: list[str]


@dataclass(kw_only=True)
class AtomResult(DictSerializable):
    status: Status
    result: str
    artifacts: list[Artifact]
    session_id: Optional[str]
    cost: float
    duration_sec: float
    raw_events_path: Optional[str]
    error: Optional[str]
    output_file: str
    output_sha256: Optional[str] = None
    last_activity: Optional[str] = None


__all__ = [
    "ApprovalSummary",
    "AtomContract",
    "AtomResult",
    "Skeleton",
    "SkeletonNode",
    "StopReport",
]
