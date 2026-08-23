#!/usr/bin/env python3
"""Audit DOD evidence and the required independent final reviews."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from validate_contract import validate as validate_contract

EVIDENCE_STATUSES = {"pass", "fail", "blocked", "unverified"}
OPEN_FINDING_STATUSES = {"confirmed", "needs_evidence"}
BLOCKING_SEVERITIES = {"critical", "high"}
CAPABILITY_PROTOCOL = "goal-capability/v1"
PARTICIPANT_ROLES = {
    "formation_owner",
    "researcher",
    "architect",
    "design_lead",
    "design_synthesizer",
    "implementer",
    "direction_reviewer",
    "milestone_reviewer",
    "final_reviewer",
}
NON_BEHAVIORAL_EVIDENCE_KINDS = {
    "research_report",
    "final_design",
    "goal_brief",
    "goal_compilation",
    "goal_package",
    "goal_prompt",
    "user_decision_record",
    "integration_record",
    "completion_review",
}
LINEAGE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--max-age-hours",
        type=float,
        help="Optionally reject evidence older than this many hours",
    )
    parser.add_argument(
        "--expected-revision",
        help="Require revision-bound evidence and reviews to match this artifact revision",
    )
    return parser.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_time(value: Any) -> datetime | None:
    if not nonempty_string(value):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def canonical_lineage(value: Any) -> str | None:
    """Accept only stable absolute task-path lineages for independence checks."""
    if not nonempty_string(value) or value != value.strip():
        return None
    if not value.startswith("/") or value.endswith("/"):
        return None
    segments = value.split("/")[1:]
    if not segments or any(
        segment in {"", ".", ".."} or LINEAGE_SEGMENT_RE.fullmatch(segment) is None
        for segment in segments
    ):
        return None
    return value


def valid_observation_window(value: Any) -> bool:
    if nonempty_string(value):
        return True
    if not isinstance(value, dict):
        return False
    start = parse_time(value.get("start"))
    end = parse_time(value.get("end"))
    return start is not None and end is not None and start <= end


def require_empty_list(document: dict[str, Any], field: str, failures: list[str]) -> None:
    value = document.get(field)
    if not isinstance(value, list):
        failures.append(f"{field} must be an array")
    elif value:
        failures.append(f"{field} is not empty")


def check_revision(
    result: dict[str, Any],
    label: str,
    expected_revision: str | None,
    fallback_revision: Any,
    failures: list[str],
) -> None:
    revision = result.get("artifact_revision", fallback_revision)
    if not nonempty_string(revision):
        failures.append(f"{label}.artifact_revision is missing")
    elif expected_revision is not None and revision != expected_revision:
        failures.append(f"{label}.artifact_revision does not match current revision")


def audit_final_review(
    review: Any,
    expected_revision: str | None,
    fallback_revision: Any,
    failures: list[str],
) -> None:
    if not isinstance(review, dict):
        failures.append("final_review is required")
        return

    ocr = review.get("ocr")
    if not isinstance(ocr, dict):
        failures.append("final_review.ocr result is required")
    else:
        verdict = ocr.get("verdict")
        if verdict not in {"pass", "not_applicable"}:
            failures.append("final_review.ocr.verdict must be pass or not_applicable")
        elif verdict == "not_applicable":
            if not nonempty_string(ocr.get("reason")):
                failures.append("final_review.ocr not_applicable requires a reason")
            if not nonempty_string(ocr.get("source")):
                failures.append("final_review.ocr not_applicable requires its check source")
            if ocr.get("status") not in {"skipped", "not_applicable"}:
                failures.append("final_review.ocr not_applicable requires a skipped status")
        else:
            if ocr.get("status") != "success":
                failures.append("final_review.ocr.status must be success")
            if ocr.get("exit_code") != 0:
                failures.append("final_review.ocr.exit_code must be 0")
            if not nonempty_string(ocr.get("source")):
                failures.append("final_review.ocr.source is missing")
            warnings = ocr.get("warnings")
            if not isinstance(warnings, list):
                failures.append("final_review.ocr.warnings must be an array")
            elif warnings:
                failures.append("final_review.ocr has unresolved warnings")
            comments = ocr.get("comments")
            if not isinstance(comments, list):
                failures.append("final_review.ocr.comments must be an array")
            elif comments and ocr.get("comments_resolved") is not True:
                failures.append("final_review.ocr comments are not fully resolved")
            unresolved = ocr.get("unresolved_findings", [])
            if not isinstance(unresolved, list) or unresolved:
                failures.append("final_review.ocr has unresolved findings")
            check_revision(ocr, "final_review.ocr", expected_revision, fallback_revision, failures)

    subagent = review.get("subagent")
    if not isinstance(subagent, dict):
        failures.append("final_review.subagent result is required")
    else:
        if subagent.get("verdict") != "pass":
            failures.append("final_review.subagent.verdict must be pass")
        if not nonempty_string(subagent.get("source")):
            failures.append("final_review.subagent.source is missing")
        if not nonempty_string(subagent.get("summary")):
            failures.append("final_review.subagent.summary is missing")
        unresolved = subagent.get("unresolved_findings", [])
        if not isinstance(unresolved, list) or unresolved:
            failures.append("final_review.subagent has unresolved findings")
        check_revision(subagent, "final_review.subagent", expected_revision, None, failures)


def audit_capability_protocol(
    evidence_doc: dict[str, Any],
    contract_revision: str,
    failures: list[str],
    final_evidence_at: datetime | None = None,
) -> None:
    protocol = evidence_doc.get("capability_protocol")
    if protocol is None:
        failures.append(f"capability_protocol must be {CAPABILITY_PROTOCOL}")
        return
    if protocol != CAPABILITY_PROTOCOL:
        failures.append(f"capability_protocol must be {CAPABILITY_PROTOCOL}")
        return

    entries = evidence_doc.get("evidence", [])
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            evidence_kind = entry.get("evidence_kind")
            if not nonempty_string(evidence_kind):
                failures.append(f"evidence[{index}].evidence_kind is required for protocol v1")
            elif evidence_kind in NON_BEHAVIORAL_EVIDENCE_KINDS:
                failures.append(
                    f"evidence[{index}].evidence_kind {evidence_kind} cannot satisfy a DOD"
                )

    ledger = evidence_doc.get("participant_ledger")
    if not isinstance(ledger, list) or not ledger:
        failures.append("participant_ledger must be a non-empty array for protocol v1")
        return

    participants: dict[str, dict[str, Any]] = {}
    prior_lineages: set[str] = set()
    prior_end_times: list[datetime] = []
    final_reviewers: list[dict[str, Any]] = []
    direction_reviewers: list[dict[str, Any]] = []
    for index, participant in enumerate(ledger):
        prefix = f"participant_ledger[{index}]"
        if not isinstance(participant, dict):
            failures.append(f"{prefix} must be an object")
            continue
        participant_id = participant.get("participant_id")
        lineage = participant.get("lineage")
        role = participant.get("role")
        if not nonempty_string(participant_id):
            failures.append(f"{prefix}.participant_id is required")
        elif participant_id in participants:
            failures.append(f"{prefix}.participant_id is duplicated")
        else:
            participants[participant_id] = participant
        if role not in PARTICIPANT_ROLES:
            failures.append(f"{prefix}.role is invalid")
        if not nonempty_string(lineage):
            failures.append(f"{prefix}.lineage is required")
        elif canonical_lineage(lineage) is None:
            failures.append(f"{prefix}.lineage must be canonical")
        elif role == "final_reviewer":
            final_reviewers.append(participant)
        else:
            prior_lineages.add(canonical_lineage(lineage))
            if role == "direction_reviewer":
                direction_reviewers.append(participant)
        if participant.get("contract_revision") != contract_revision:
            failures.append(f"{prefix}.contract_revision does not match")
        if not isinstance(participant.get("history_inherited"), bool):
            failures.append(f"{prefix}.history_inherited must be boolean")
        artifact_ids = participant.get("artifact_ids")
        if not isinstance(artifact_ids, list) or not artifact_ids or not all(
            nonempty_string(item) for item in artifact_ids
        ):
            failures.append(f"{prefix}.artifact_ids must be a non-empty string array")
        participant_times: dict[str, datetime] = {}
        for field in ("started_at", "ended_at"):
            parsed_time = parse_time(participant.get(field))
            if parsed_time is None:
                failures.append(f"{prefix}.{field} must be a timezone-aware timestamp")
            else:
                participant_times[field] = parsed_time
        if (
            "started_at" in participant_times
            and "ended_at" in participant_times
            and participant_times["started_at"] > participant_times["ended_at"]
        ):
            failures.append(f"{prefix}.started_at must not be after ended_at")
        if role != "final_reviewer" and "ended_at" in participant_times:
            prior_end_times.append(participant_times["ended_at"])

    if len(final_reviewers) != 1:
        failures.append("participant_ledger must contain exactly one final_reviewer")

    for reviewer in direction_reviewers:
        reviewer_id = reviewer.get("participant_id")
        reviewer_lineage = reviewer.get("lineage")
        if reviewer.get("history_inherited") is not False:
            failures.append(
                f"direction_reviewer {reviewer_id} must have history_inherited=false"
            )
        if not nonempty_string(reviewer_lineage):
            continue
        normalized = canonical_lineage(reviewer_lineage)
        if normalized is None:
            failures.append(f"direction_reviewer {reviewer_id} lineage must be canonical")
            continue
        overlaps = any(
            participant.get("participant_id") != reviewer_id
            and nonempty_string(participant.get("lineage"))
            and canonical_lineage(participant.get("lineage")) is not None
            and (
                normalized == canonical_lineage(participant["lineage"])
                or normalized.startswith(canonical_lineage(participant["lineage"]) + "/")
                or canonical_lineage(participant["lineage"]).startswith(normalized + "/")
            )
            for participant in participants.values()
        )
        if overlaps:
            failures.append(
                f"direction_reviewer {reviewer_id} lineage overlaps another Goal participant"
            )

    review = evidence_doc.get("final_review")
    subagent = review.get("subagent") if isinstance(review, dict) else None
    if not isinstance(subagent, dict):
        return
    reviewer_id = subagent.get("participant_id")
    reviewer_lineage = subagent.get("lineage")
    if not nonempty_string(reviewer_id):
        failures.append("final_review.subagent.participant_id is required for protocol v1")
    if not nonempty_string(reviewer_lineage):
        failures.append("final_review.subagent.lineage is required for protocol v1")
    elif canonical_lineage(reviewer_lineage) is None:
        failures.append("final_review.subagent.lineage must be canonical")
    reviewer = participants.get(reviewer_id) if isinstance(reviewer_id, str) else None
    if reviewer is None:
        failures.append("final_review.subagent participant is missing from participant_ledger")
        return
    if reviewer.get("role") != "final_reviewer":
        failures.append("final_review.subagent participant must have role final_reviewer")
    if reviewer.get("lineage") != reviewer_lineage:
        failures.append("final_review.subagent lineage does not match participant_ledger")
    if reviewer.get("history_inherited") is not False:
        failures.append("final_reviewer must have history_inherited=false")
    if nonempty_string(reviewer_lineage):
        normalized_reviewer = canonical_lineage(reviewer_lineage)
        overlaps = any(
            normalized_reviewer == prior
            or normalized_reviewer.startswith(prior + "/")
            or prior.startswith(normalized_reviewer + "/")
            for prior in prior_lineages
        ) if normalized_reviewer is not None else False
        if overlaps:
            failures.append("final_reviewer lineage overlaps a prior Goal participant")
    reviewer_started = parse_time(reviewer.get("started_at"))
    if reviewer_started is not None and prior_end_times and reviewer_started < max(prior_end_times):
        failures.append("final_reviewer started before prior Goal participation ended")
    if (
        reviewer_started is not None
        and final_evidence_at is not None
        and reviewer_started <= final_evidence_at
    ):
        failures.append("final_reviewer did not start after final DOD evidence was recorded")


def main() -> int:
    args = parse_args()
    if args.max_age_hours is not None and args.max_age_hours <= 0:
        print("INCOMPLETE: --max-age-hours must be positive")
        return 2
    try:
        contract = load(args.contract)
        evidence_doc = load(args.evidence)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INCOMPLETE: {exc}")
        return 2
    if not isinstance(contract, dict) or not isinstance(evidence_doc, dict):
        print("INCOMPLETE: contract and evidence roots must be objects")
        return 2

    failures: list[str] = []
    for error in validate_contract(contract, allow_draft=False):
        failures.append(f"invalid contract: {error}")
    if contract.get("goal_id") != evidence_doc.get("goal_id"):
        failures.append("goal_id mismatch")
    contract_revision = contract.get("contract_revision")
    if not nonempty_string(contract_revision):
        failures.append("contract_revision is missing")
    elif evidence_doc.get("contract_revision") != contract_revision:
        failures.append("contract_revision mismatch")
    if contract.get("status") not in {"verifying", "complete"}:
        failures.append("contract status must be verifying or complete")

    unknowns = contract.get("unknowns", [])
    if not isinstance(unknowns, list):
        failures.append("unknowns must be an array")
    else:
        for unknown in unknowns:
            if isinstance(unknown, dict) and unknown.get("status") == "open":
                failures.append(f"open unknown remains: {unknown.get('id', '<unknown>')}")

    criteria = contract.get("definition_of_done")
    entries = evidence_doc.get("evidence")
    if not isinstance(criteria, list) or not isinstance(entries, list):
        print("INCOMPLETE: malformed definition_of_done or evidence array")
        return 2

    by_id: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("criterion_id"), str):
            by_id.setdefault(entry["criterion_id"], []).append(entry)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=args.max_age_hours) if args.max_age_hours is not None else None
    fallback_revision = evidence_doc.get("revision")
    for criterion in criteria:
        if not isinstance(criterion, dict) or not isinstance(criterion.get("id"), str):
            failures.append("contract contains criterion without a valid id")
            continue
        criterion_id = criterion["id"]
        candidates = by_id.get(criterion_id, [])
        timed: list[tuple[datetime, dict[str, Any]]] = []
        for entry in candidates:
            checked_at = parse_time(entry.get("checked_at"))
            if checked_at is not None:
                timed.append((checked_at, entry))
        if not timed:
            failures.append(f"{criterion_id}: no valid evidence")
            continue

        checked_at, latest = max(timed, key=lambda pair: pair[0])
        if checked_at > now:
            failures.append(f"{criterion_id}: checked_at is in the future")
        if cutoff is not None and checked_at < cutoff:
            failures.append(f"{criterion_id}: evidence is older than the requested maximum age")
        if latest.get("status") not in EVIDENCE_STATUSES:
            failures.append(f"{criterion_id}: latest evidence status is invalid")
        elif latest.get("status") != "pass":
            failures.append(f"{criterion_id}: latest status is {latest.get('status')}")
        if latest.get("contract_revision") != contract_revision:
            failures.append(f"{criterion_id}: contract_revision does not match")

        artifact_revision = latest.get("artifact_revision", fallback_revision)
        observation_window = latest.get("observation_window")
        if not nonempty_string(artifact_revision) and not valid_observation_window(observation_window):
            failures.append(f"{criterion_id}: artifact_revision or observation_window is required")
        elif (
            args.expected_revision is not None
            and nonempty_string(artifact_revision)
            and artifact_revision != args.expected_revision
        ):
            failures.append(f"{criterion_id}: artifact revision does not match current revision")
        if not nonempty_string(latest.get("source")):
            failures.append(f"{criterion_id}: evidence source is missing")
        if not nonempty_string(latest.get("summary")):
            failures.append(f"{criterion_id}: evidence summary is missing")

    if not criteria:
        failures.append("definition_of_done is empty")

    if contract.get("final_review_required") is not True:
        failures.append("final_review_required must be true")
    else:
        audit_final_review(
            evidence_doc.get("final_review"),
            args.expected_revision,
            fallback_revision,
            failures,
        )

    latest_dod_checked_at = None
    for criterion in criteria:
        if not isinstance(criterion, dict) or not isinstance(criterion.get("id"), str):
            continue
        timed_entries = [
            parse_time(entry.get("checked_at"))
            for entry in by_id.get(criterion["id"], [])
            if isinstance(entry, dict)
        ]
        valid_times = [timestamp for timestamp in timed_entries if timestamp is not None]
        if valid_times:
            candidate = max(valid_times)
            latest_dod_checked_at = (
                candidate
                if latest_dod_checked_at is None
                else max(latest_dod_checked_at, candidate)
            )
    if nonempty_string(contract_revision):
        audit_capability_protocol(
            evidence_doc,
            contract_revision,
            failures,
            final_evidence_at=latest_dod_checked_at,
        )

    require_empty_list(evidence_doc, "pending_approvals", failures)
    require_empty_list(evidence_doc, "remaining_required_work", failures)

    findings = evidence_doc.get("findings")
    if not isinstance(findings, list):
        failures.append("findings must be an array")
    else:
        for finding in findings:
            if not isinstance(finding, dict):
                failures.append("findings contains a non-object entry")
                continue
            if (
                finding.get("severity") in BLOCKING_SEVERITIES
                and finding.get("status") in OPEN_FINDING_STATUSES
            ):
                failures.append(f"blocking finding remains: {finding.get('finding_id', '<unknown>')}")
            if finding.get("status") == "accepted_risk" and not nonempty_string(finding.get("accepted_by")):
                failures.append(f"accepted risk lacks authorization: {finding.get('finding_id', '<unknown>')}")

    if failures:
        for failure in failures:
            print(f"INCOMPLETE: {failure}")
        return 1
    print(f"COMPLETE: {len(criteria)} criteria and both final review gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
