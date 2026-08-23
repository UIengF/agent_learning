"""Deterministic structured reviewer for tests and smoke flows."""

from __future__ import annotations

from copy import deepcopy

from atomic_agents.models import AtomResult
from atomic_agents.reviewer import CriterionVerdict, ReviewResult, Reviewer


class MockReviewer(Reviewer):
    """Programmable reviewer returning structured ``ReviewResult`` objects.

    Scripts are keyed by node id. When a scripted response list is exhausted,
    the last response is reused so repeated calls remain deterministic.
    """

    def __init__(
        self,
        script: dict[str, list[ReviewResult]] | None = None,
        default_pass: bool = True,
    ) -> None:
        self.script = {node_id: [deepcopy(review) for review in reviews] for node_id, reviews in (script or {}).items()}
        self.default_pass = default_pass
        self.review_counts: dict[str, int] = {}

    def review(self, node_id: str, result: AtomResult, criteria: list[str]) -> ReviewResult:
        """Return the next scripted review, or a generated default review."""

        del result
        call_index = self.review_counts.get(node_id, 0)
        self.review_counts[node_id] = call_index + 1

        if node_id in self.script and self.script[node_id]:
            review = self._scripted_result(node_id, call_index)
        else:
            review = make_pass_result(criteria) if self.default_pass else make_fail_result(criteria, criteria)

        review.reviewer_session = f"rev-{node_id}-{call_index}"
        review.__post_init__()
        return review

    def _scripted_result(self, node_id: str, call_index: int) -> ReviewResult:
        reviews = self.script[node_id]
        review_index = min(call_index, len(reviews) - 1)
        return deepcopy(reviews[review_index])


def make_pass_result(criteria: list[str]) -> ReviewResult:
    """Build a passing review result for the given criteria."""

    verdicts = [
        CriterionVerdict(
            criterion=criterion,
            verdict="pass",
            evidence="mock",
            confidence="high",
        )
        for criterion in criteria
    ]
    return ReviewResult(
        passed=True,
        criteria=verdicts,
        blocking_findings=[],
        reviewer_session=None,
        feedback="",
    )


def make_fail_result(criteria: list[str], failing: list[str]) -> ReviewResult:
    """Build a failing review result, marking matching criteria as blocking."""

    failing_set = set(failing)
    verdicts = [
        CriterionVerdict(
            criterion=criterion,
            verdict="fail" if criterion in failing_set else "pass",
            evidence="mock",
            confidence="high",
        )
        for criterion in criteria
    ]
    blocking_findings = [criterion for criterion in criteria if criterion in failing_set]
    return ReviewResult(
        passed=False,
        criteria=verdicts,
        blocking_findings=blocking_findings,
        reviewer_session=None,
        feedback="",
    )


__all__ = ["MockReviewer", "make_fail_result", "make_pass_result"]
