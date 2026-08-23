"""Structured reviewer gate models and protocol.

Reviewers are invoked as stateless atoms: each review call evaluates one
``AtomResult`` against explicit criteria and returns auditable verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

from atomic_agents.models import AtomResult


CriterionConfidence = Literal["high", "medium", "low"]
CriterionVerdictValue = Literal["pass", "fail"]
ReviewVerdictValue = Literal["pass", "fail"]


@dataclass(kw_only=True)
class CriterionVerdict:
    """Reviewer verdict for one acceptance criterion."""

    criterion: str
    verdict: CriterionVerdictValue
    evidence: str
    confidence: CriterionConfidence

    def to_dict(self) -> dict[str, str]:
        """Return a plain JSON-serializable mapping."""

        return {
            "criterion": self.criterion,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CriterionVerdict:
        """Build a criterion verdict from a plain mapping."""

        return cls(
            criterion=str(data["criterion"]),
            verdict=_criterion_verdict(data["verdict"]),
            evidence=str(data["evidence"]),
            confidence=_criterion_confidence(data["confidence"]),
        )


@dataclass(kw_only=True)
class ReviewResult:
    """Structured output produced by a reviewer atom.

    ``passed`` is normalized from ``blocking_findings``: a review passes only
    when there are no blocking findings.
    """

    passed: bool
    criteria: list[CriterionVerdict]
    blocking_findings: list[str]
    reviewer_session: str | None
    feedback: str = field(default="")

    def __post_init__(self) -> None:
        self.criteria = list(self.criteria)
        self.blocking_findings = list(self.blocking_findings)
        fail_criteria = {criterion.criterion for criterion in self.criteria if criterion.verdict == "fail"}
        missing = fail_criteria - set(self.blocking_findings)
        if missing:
            raise ValueError(
                f"criteria marked fail but missing from blocking_findings: {sorted(missing)}"
            )
        self.passed = not self.blocking_findings
        if not self.feedback:
            self.feedback = build_review_feedback(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable mapping."""

        return {
            "passed": self.passed,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "blocking_findings": list(self.blocking_findings),
            "reviewer_session": self.reviewer_session,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewResult:
        """Build a review result from a plain mapping."""

        return cls(
            passed=bool(data["passed"]),
            criteria=[CriterionVerdict.from_dict(item) for item in data["criteria"]],
            blocking_findings=[str(item) for item in data["blocking_findings"]],
            reviewer_session=data.get("reviewer_session"),
            feedback=str(data.get("feedback", "")),
        )


class Reviewer(Protocol):
    """Protocol for stateless reviewer atoms.

    Each call is semantically a fresh reviewer session. Implementations should
    not depend on previous review context when deciding the verdict.
    """

    def review(self, node_id: str, result: AtomResult, criteria: list[str]) -> ReviewResult:
        """Review an atom result against explicit criteria."""


def build_review_feedback(review: ReviewResult) -> str:
    """Summarize failed criteria for injection into a repair retry."""

    failures = [criterion for criterion in review.criteria if criterion.verdict == "fail"]
    if not failures:
        return ""

    lines = ["Reviewer blocking feedback:"]
    for criterion in failures:
        lines.append(f"- {criterion.criterion}: {criterion.evidence}")
    return "\n".join(lines)


def review_to_lock_payload(review: ReviewResult) -> dict[str, Any]:
    """Convert a review result into the ``review_finished`` lock payload."""

    return {
        "verdict": "pass" if review.passed else "fail",
        "criteria": [
            {
                "id": criterion.criterion,
                "verdict": criterion.verdict,
                "evidence": criterion.evidence,
                "confidence": criterion.confidence,
            }
            for criterion in review.criteria
        ],
        "blocking_findings": list(review.blocking_findings),
        "reviewer_session": review.reviewer_session,
    }


def _criterion_verdict(value: Any) -> CriterionVerdictValue:
    if value in ("pass", "fail"):
        return value
    raise ValueError(f"invalid criterion verdict: {value!r}")


def _criterion_confidence(value: Any) -> CriterionConfidence:
    if value in ("high", "medium", "low"):
        return value
    raise ValueError(f"invalid criterion confidence: {value!r}")


__all__ = [
    "CriterionConfidence",
    "CriterionVerdict",
    "CriterionVerdictValue",
    "ReviewResult",
    "ReviewVerdictValue",
    "Reviewer",
    "build_review_feedback",
    "review_to_lock_payload",
]
