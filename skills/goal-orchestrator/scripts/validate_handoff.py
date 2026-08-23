#!/usr/bin/env python3
"""Validate goal-capability/v1 handoff artifacts and delegated bounds."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROTOCOL = "goal-capability/v1"
KINDS = {
    "research_brief",
    "research_report",
    "design_brief",
    "final_design",
    "goal_intake",
    "goal_brief",
    "goal_compile_request",
    "goal_compilation",
    "goal_package",
    "goal_prompt",
    "user_decision_record",
    "integration_record",
    "participant_ledger",
}
RUNTIME_PHASES = {"active", "verifying"}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ARTIFACT_CLASSES = {
    "research_report": "advisory_not_decision_not_completion_evidence",
    "final_design": "adjudicated_design_not_implementation_evidence",
    "goal_brief": "advisory_goal_brief_not_goal_state",
    "goal_compilation": "candidate_goal_contract_not_goal_state",
    "goal_package": "approved_goal_package_not_goal_state",
    "goal_prompt": "rendered_goal_prompt_not_goal_state",
    "user_decision_record": "user_authorization_not_completion_evidence",
    "integration_record": "integration_metadata_not_completion_evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--request",
        type=Path,
        help="Validate a report/design response against its original brief binding",
    )
    parser.add_argument("--expected-kind", choices=sorted(KINDS))
    parser.add_argument("--expected-goal-id")
    parser.add_argument("--expected-formation-id")
    parser.add_argument("--expected-contract-revision")
    parser.add_argument("--expected-input-fingerprint")
    parser.add_argument(
        "--allowed-read-path",
        action="append",
        default=[],
        help="Allowed path root; repeat for multiple roots",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow a delegated artifact to request public read-only network access",
    )
    return parser.parse_args()


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def string_list(value: Any, *, nonempty_list: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (not nonempty_list or bool(value))
        and all(nonempty(item) for item in value)
    )


def object_list(value: Any, *, nonempty_list: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (not nonempty_list or bool(value))
        and all(isinstance(item, dict) for item in value)
    )


def valid_time(value: Any) -> bool:
    if not nonempty(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def path_within(path: str, roots: list[str]) -> bool:
    candidate = Path(path).expanduser().resolve(strict=False)
    for root in roots:
        root_path = Path(root).expanduser().resolve(strict=False)
        try:
            candidate.relative_to(root_path)
            return True
        except ValueError:
            continue
    return False


def validate_binding(
    request: Any, response: Any, errors: list[str]
) -> None:
    if not isinstance(request, dict) or not isinstance(response, dict):
        errors.append("request and response roots must be objects")
        return
    expected_pair = {
        "research_brief": "research_report",
        "design_brief": "final_design",
        "goal_intake": "goal_brief",
    }.get(request.get("artifact_kind"))
    if expected_pair is None and request.get("artifact_kind") != "goal_compile_request":
        errors.append(
            "--request artifact_kind must be research_brief, design_brief, goal_intake, or goal_compile_request"
        )
    elif request.get("artifact_kind") == "goal_compile_request":
        if response.get("artifact_kind") not in {
            "goal_compilation",
            "goal_package",
            "goal_prompt",
        }:
            errors.append(
                "response artifact_kind must be goal_compilation, goal_package, or goal_prompt"
            )
    elif response.get("artifact_kind") != expected_pair:
        errors.append("response artifact_kind does not match the request kind")
    for field in (
        "protocol",
        "request_id",
        "invocation_mode",
        "goal_id",
        "formation_id",
        "contract_revision",
        "input_fingerprint",
        "snapshot_refs",
        "permission_ceiling",
    ):
        if response.get(field) != request.get(field):
            errors.append(f"response {field} does not match the request binding")
    if request.get("artifact_id") == response.get("artifact_id"):
        errors.append("response artifact_id must differ from the request artifact_id")
    if request.get("artifact_kind") == "goal_compile_request":
        request_payload = request.get("payload")
        response_payload = response.get("payload")
        if isinstance(request_payload, dict) and isinstance(response_payload, dict):
            for field in ("changed_fields", "user_decision_refs"):
                if response_payload.get(field) != request_payload.get(field):
                    errors.append(
                        f"response payload.{field} does not match the request binding"
                    )
            current_contract = request_payload.get("current_contract")
            response_contract = response_payload.get("contract")
            changed_fields = request_payload.get("changed_fields")
            if (
                isinstance(current_contract, dict)
                and isinstance(response_contract, dict)
                and isinstance(changed_fields, list)
            ):
                actual_changes = {
                    key
                    for key in current_contract.keys() | response_contract.keys()
                    if current_contract.get(key) != response_contract.get(key)
                }
                undeclared = actual_changes - set(changed_fields)
                if undeclared:
                    errors.append(
                        "response contract changes undeclared fields: "
                        + ", ".join(sorted(undeclared))
                    )


def require_fields(
    payload: dict[str, Any], fields: tuple[str, ...], prefix: str, errors: list[str]
) -> None:
    for field in fields:
        if field not in payload:
            errors.append(f"{prefix}.{field} is required")


def validate_snapshots(value: Any, errors: list[str]) -> None:
    if not object_list(value, nonempty_list=True):
        errors.append("snapshot_refs must be a non-empty array of objects")
        return
    for index, snapshot in enumerate(value):
        if not nonempty(snapshot.get("locator")):
            errors.append(f"snapshot_refs[{index}].locator is required")
        if not nonempty(snapshot.get("revision_or_hash")):
            errors.append(f"snapshot_refs[{index}].revision_or_hash is required")


def validate_research_brief(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        (
            "purpose",
            "questions",
            "scope",
            "exclusions",
            "evidence_policy",
            "stop_conditions",
            "known_facts",
            "known_unknowns",
            "workspace_rule",
        ),
        "payload",
        errors,
    )
    questions = payload.get("questions")
    if not object_list(questions, nonempty_list=True):
        errors.append("payload.questions must be a non-empty array of objects")
    else:
        ids: set[str] = set()
        for index, question in enumerate(questions):
            question_id = question.get("id")
            if not nonempty(question_id):
                errors.append(f"payload.questions[{index}].id is required")
            elif question_id in ids:
                errors.append(f"payload.questions[{index}].id is duplicated")
            else:
                ids.add(question_id)
            if not nonempty(question.get("question")):
                errors.append(f"payload.questions[{index}].question is required")
            if not nonempty(question.get("decision_impact")):
                errors.append(f"payload.questions[{index}].decision_impact is required")


def validate_research_report(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        (
            "artifact_class",
            "confirmed_facts",
            "inferences",
            "contradictions",
            "unresolved",
            "coverage",
            "participants",
        ),
        "payload",
        errors,
    )
    for field in ("confirmed_facts", "inferences", "contradictions", "unresolved", "coverage"):
        if not isinstance(payload.get(field), list):
            errors.append(f"payload.{field} must be an array")
    coverage = payload.get("coverage", [])
    if isinstance(coverage, list):
        for index, item in enumerate(coverage):
            if not isinstance(item, dict):
                continue
            if not nonempty(item.get("question_id")):
                errors.append(f"payload.coverage[{index}].question_id is required")
            if item.get("status") not in {"covered", "partial", "missing"}:
                errors.append(f"payload.coverage[{index}].status is invalid")
    if not object_list(payload.get("participants"), nonempty_list=True):
        errors.append("payload.participants must be a non-empty array of objects")


def validate_design_brief(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        (
            "problem",
            "desired_outcome",
            "confirmed_facts",
            "assumptions",
            "known_unknowns",
            "hard_constraints",
            "preferences",
            "non_goals",
            "authorized_scope",
            "evaluation_criteria",
            "research_artifact_refs",
            "decision_record_refs",
            "decisions_reserved_for_user",
            "required_deliverable",
        ),
        "payload",
        errors,
    )
    if not nonempty(payload.get("problem")):
        errors.append("payload.problem must be non-empty")
    if not nonempty(payload.get("desired_outcome")):
        errors.append("payload.desired_outcome must be non-empty")
    if not nonempty(payload.get("required_deliverable")):
        errors.append("payload.required_deliverable must be non-empty")
    if not isinstance(payload.get("evaluation_criteria"), list) or not payload.get("evaluation_criteria"):
        errors.append("payload.evaluation_criteria must be a non-empty array")


def validate_final_design(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        (
            "artifact_class",
            "chosen_architecture",
            "modules_and_interfaces",
            "data_flows",
            "invariants",
            "failure_handling",
            "security_and_performance",
            "migration",
            "tests_and_acceptance",
            "risks",
            "decision_records",
            "rejected_alternatives",
            "unresolved",
            "candidate_coverage",
        ),
        "payload",
        errors,
    )
    if not isinstance(payload.get("chosen_architecture"), dict) or not payload.get("chosen_architecture"):
        errors.append("payload.chosen_architecture must be a non-empty object")
    for field in (
        "modules_and_interfaces",
        "data_flows",
        "invariants",
        "failure_handling",
        "security_and_performance",
        "migration",
        "tests_and_acceptance",
        "risks",
        "decision_records",
        "rejected_alternatives",
        "unresolved",
        "candidate_coverage",
    ):
        if not isinstance(payload.get(field), list):
            errors.append(f"payload.{field} must be an array")
    unresolved = payload.get("unresolved", [])
    if isinstance(unresolved, list):
        for index, item in enumerate(unresolved):
            if isinstance(item, dict) and item.get("kind") not in {
                "missing_evidence",
                "user_decision",
                "implementation_detail",
            }:
                errors.append(f"payload.unresolved[{index}].kind is invalid")


def validate_goal_intake(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        ("user_intent", "scope_hints", "known_facts", "known_unknowns", "requested_mode"),
        "payload",
        errors,
    )
    if not nonempty(payload.get("user_intent")):
        errors.append("payload.user_intent must be non-empty")
    if payload.get("requested_mode") not in {"brief", "compile"}:
        errors.append("payload.requested_mode must be brief or compile")


def validate_goal_brief(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        (
            "artifact_class",
            "objective",
            "outcome",
            "scope",
            "exclusions",
            "constraints",
            "definition_of_done",
            "unknowns",
            "recommended_mode",
            "initialization",
            "unresolved_questions",
            "active_truth_refs",
            "orchestration_policy",
        ),
        "payload",
        errors,
    )
    for field in ("objective", "outcome"):
        if not nonempty(payload.get(field)):
            errors.append(f"payload.{field} must be non-empty")
    if payload.get("recommended_mode") not in {"fast", "deep"}:
        errors.append("payload.recommended_mode must be fast or deep")
    if not isinstance(payload.get("definition_of_done"), list) or not payload.get("definition_of_done"):
        errors.append("payload.definition_of_done must be a non-empty array")
    if not isinstance(payload.get("orchestration_policy"), dict):
        errors.append("payload.orchestration_policy must be an object")


def validate_goal_compile_request(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        (
            "goal_brief",
            "current_contract",
            "changed_fields",
            "research_refs",
            "design_refs",
            "user_decision_refs",
            "target_executor",
            "render_prompt",
        ),
        "payload",
        errors,
    )
    if not isinstance(payload.get("goal_brief"), dict):
        errors.append("payload.goal_brief must be an object")
    if not isinstance(payload.get("current_contract"), dict) or not payload.get(
        "current_contract"
    ):
        errors.append("payload.current_contract must be a non-empty object")
    if not string_list(payload.get("changed_fields"), nonempty_list=True):
        errors.append("payload.changed_fields must be a non-empty string array")
    for field in ("research_refs", "design_refs"):
        if not isinstance(payload.get(field), list):
            errors.append(f"payload.{field} must be an array")
    if not string_list(payload.get("user_decision_refs"), nonempty_list=True):
        errors.append("payload.user_decision_refs must be a non-empty string array")
    if payload.get("target_executor") not in {"native_goal", "external_agent", "both"}:
        errors.append("payload.target_executor is invalid")
    if not isinstance(payload.get("render_prompt"), bool):
        errors.append("payload.render_prompt must be boolean")


def validate_goal_compilation(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        (
            "artifact_class",
            "contract",
            "changed_fields",
            "user_decision_refs",
            "source_refs",
            "compilation_warnings",
            "unresolved",
        ),
        "payload",
        errors,
    )
    if not isinstance(payload.get("contract"), dict) or not payload.get("contract"):
        errors.append("payload.contract must be a non-empty object")
    if not string_list(payload.get("changed_fields"), nonempty_list=True):
        errors.append("payload.changed_fields must be a non-empty string array")
    if not isinstance(payload.get("user_decision_refs"), list) or not all(
        nonempty(item) for item in payload.get("user_decision_refs", [])
    ):
        errors.append("payload.user_decision_refs must be a string array")
    for field in ("source_refs", "compilation_warnings", "unresolved"):
        if not isinstance(payload.get(field), list):
            errors.append(f"payload.{field} must be an array")


def validate_goal_package(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        (
            "artifact_class",
            "formation_id",
            "package_revision",
            "contract",
            "initial_solution",
            "initial_milestones",
            "changed_fields",
            "user_decision_refs",
            "source_refs",
            "participant_ledger",
            "user_confirmation",
            "handoff_intent",
            "unresolved",
        ),
        "payload",
        errors,
    )
    for field in ("formation_id", "package_revision"):
        if not nonempty(payload.get(field)):
            errors.append(f"payload.{field} must be non-empty")
    contract = payload.get("contract")
    if not isinstance(contract, dict) or not contract:
        errors.append("payload.contract must be a non-empty object")
    else:
        for field in ("objective", "outcome"):
            if not nonempty(contract.get(field)):
                errors.append(f"payload.contract.{field} must be non-empty")
        if not object_list(contract.get("definition_of_done"), nonempty_list=True):
            errors.append(
                "payload.contract.definition_of_done must be a non-empty array of objects"
            )
    solution = payload.get("initial_solution")
    if not isinstance(solution, dict):
        errors.append("payload.initial_solution must be an object")
    else:
        if not nonempty(solution.get("summary")):
            errors.append("payload.initial_solution.summary must be non-empty")
        if not isinstance(solution.get("rejected_alternatives"), list):
            errors.append("payload.initial_solution.rejected_alternatives must be an array")
    milestones = payload.get("initial_milestones")
    if not object_list(milestones, nonempty_list=True):
        errors.append("payload.initial_milestones must be a non-empty array of objects")
    else:
        for index, milestone in enumerate(milestones):
            for field in ("id", "outcome", "rollback"):
                if not nonempty(milestone.get(field)):
                    errors.append(
                        f"payload.initial_milestones[{index}].{field} must be non-empty"
                    )
            if not string_list(milestone.get("dod_ids"), nonempty_list=True):
                errors.append(
                    f"payload.initial_milestones[{index}].dod_ids must be a non-empty string array"
                )
            for field in ("dependencies", "assumptions", "write_scope", "risks"):
                if not isinstance(milestone.get(field), list):
                    errors.append(
                        f"payload.initial_milestones[{index}].{field} must be an array"
                    )
            verification = milestone.get("verification")
            if not isinstance(verification, dict):
                errors.append(
                    f"payload.initial_milestones[{index}].verification must be an object"
                )
            else:
                for field in ("method", "expected"):
                    if not nonempty(verification.get(field)):
                        errors.append(
                            f"payload.initial_milestones[{index}].verification.{field} must be non-empty"
                        )
    for field in ("source_refs", "unresolved"):
        if not isinstance(payload.get(field), list):
            errors.append(f"payload.{field} must be an array")
    if not string_list(payload.get("changed_fields"), nonempty_list=True):
        errors.append("payload.changed_fields must be a non-empty string array")
    if not isinstance(payload.get("user_decision_refs"), list) or not all(
        nonempty(item) for item in payload.get("user_decision_refs", [])
    ):
        errors.append("payload.user_decision_refs must be a string array")
    ledger = payload.get("participant_ledger")
    if not object_list(ledger, nonempty_list=True):
        errors.append("payload.participant_ledger must be a non-empty array of objects")
    else:
        for index, participant in enumerate(ledger):
            for field in (
                "participant_id",
                "role",
                "lineage",
                "artifact_ids",
                "history_inherited",
                "started_at",
                "ended_at",
            ):
                if field not in participant:
                    errors.append(
                        f"payload.participant_ledger[{index}].{field} is required"
                    )
            for field in ("participant_id", "role", "lineage"):
                if not nonempty(participant.get(field)):
                    errors.append(
                        f"payload.participant_ledger[{index}].{field} must be non-empty"
                    )
            if not string_list(participant.get("artifact_ids")):
                errors.append(
                    f"payload.participant_ledger[{index}].artifact_ids must be a string array"
                )
            if not isinstance(participant.get("history_inherited"), bool):
                errors.append(
                    f"payload.participant_ledger[{index}].history_inherited must be boolean"
                )
            for field in ("started_at", "ended_at"):
                if not valid_time(participant.get(field)):
                    errors.append(
                        f"payload.participant_ledger[{index}].{field} must be a timezone-aware timestamp"
                    )
    confirmation = payload.get("user_confirmation")
    if not isinstance(confirmation, dict):
        errors.append("payload.user_confirmation must be an object")
    else:
        if confirmation.get("confirmed") is not True:
            errors.append("payload.user_confirmation.confirmed must be true")
        if not nonempty(confirmation.get("statement_or_ref")):
            errors.append("payload.user_confirmation.statement_or_ref must be non-empty")
        if not valid_time(confirmation.get("confirmed_at")):
            errors.append("payload.user_confirmation.confirmed_at must be a timezone-aware ISO 8601 timestamp")
        for field in ("creation_authorized", "start_authorized"):
            if not isinstance(confirmation.get(field), bool):
                errors.append(f"payload.user_confirmation.{field} must be boolean")
    intent = payload.get("handoff_intent")
    if intent not in {"prompt_only", "ready", "activate"}:
        errors.append("payload.handoff_intent is invalid")
    elif isinstance(confirmation, dict):
        creation_authorized = confirmation.get("creation_authorized")
        start_authorized = confirmation.get("start_authorized")
        if intent == "prompt_only" and (
            creation_authorized is not False or start_authorized is not False
        ):
            errors.append("prompt_only requires creation_authorized=false and start_authorized=false")
        elif intent == "ready" and (
            creation_authorized is not True or start_authorized is not False
        ):
            errors.append("ready requires creation_authorized=true and start_authorized=false")
        elif intent == "activate" and (
            creation_authorized is not True or start_authorized is not True
        ):
            errors.append("activate requires creation_authorized=true and start_authorized=true")


def validate_goal_prompt(payload: dict[str, Any], errors: list[str]) -> None:
    require_fields(
        payload,
        ("artifact_class", "prompt", "target_executor", "source_contract_revision"),
        "payload",
        errors,
    )
    if not nonempty(payload.get("prompt")):
        errors.append("payload.prompt must be non-empty")
    if payload.get("target_executor") not in {"native_goal", "external_agent", "both"}:
        errors.append("payload.target_executor is invalid")
    if not nonempty(payload.get("source_contract_revision")):
        errors.append("payload.source_contract_revision must be non-empty")


def validate_other_payload(kind: str, payload: dict[str, Any], errors: list[str]) -> None:
    fields_by_kind = {
        "user_decision_record": (
            "artifact_class",
            "decision_id",
            "user_statement",
            "decision",
            "scope",
            "decided_by",
            "decided_at",
            "applies_to_revision",
        ),
        "integration_record": (
            "artifact_class",
            "source_artifact_id",
            "source_fingerprint",
            "input_revision",
            "disposition",
            "mapped_items",
            "reason",
            "before_revision",
            "after_revision",
            "owner_id",
        ),
        "participant_ledger": ("participants",),
    }
    require_fields(payload, fields_by_kind[kind], "payload", errors)
    if kind == "integration_record" and payload.get("disposition") not in {
        "accepted",
        "partially_accepted",
        "rejected",
        "stale",
    }:
        errors.append("payload.disposition is invalid")
    if kind == "participant_ledger" and not object_list(
        payload.get("participants"), nonempty_list=True
    ):
        errors.append("payload.participants must be a non-empty array of objects")


def validate(data: Any, args: argparse.Namespace) -> list[str]:
    if not isinstance(data, dict):
        return ["artifact root must be an object"]

    errors: list[str] = []
    if data.get("protocol") != PROTOCOL:
        errors.append(f"protocol must be {PROTOCOL}")
    kind = data.get("artifact_kind")
    if kind not in KINDS:
        errors.append("artifact_kind is invalid")
    elif args.expected_kind is not None and kind != args.expected_kind:
        errors.append("artifact_kind does not match --expected-kind")
    for field in ("artifact_id", "request_id"):
        if not nonempty(data.get(field)):
            errors.append(f"{field} is required")
    if data.get("invocation_mode") not in {
        "direct",
        "delegated_by_goal",
        "delegated_by_goal_prompt",
    }:
        errors.append("invocation_mode is invalid")
    if not SHA256_RE.fullmatch(data.get("input_fingerprint", "")):
        errors.append("input_fingerprint must be sha256:<64 lowercase hex characters>")
    if not valid_time(data.get("created_at")):
        errors.append("created_at must be a timezone-aware ISO 8601 timestamp")
    if not isinstance(data.get("payload"), dict):
        errors.append("payload must be an object")

    validate_snapshots(data.get("snapshot_refs"), errors)
    permissions = data.get("permission_ceiling")
    if not isinstance(permissions, dict):
        errors.append("permission_ceiling must be an object")
    else:
        if not string_list(permissions.get("read_paths"), nonempty_list=True):
            errors.append("permission_ceiling.read_paths must be a non-empty string array")
        if permissions.get("write_allowed") is not False:
            errors.append("permission_ceiling.write_allowed must be false")
        if not isinstance(permissions.get("network_allowed"), bool):
            errors.append("permission_ceiling.network_allowed must be boolean")
        elif permissions["network_allowed"] and not args.allow_network:
            errors.append("network access exceeds the allowed ceiling")
        if args.allowed_read_path and isinstance(permissions.get("read_paths"), list):
            for index, path in enumerate(permissions["read_paths"]):
                if isinstance(path, str) and not path_within(path, args.allowed_read_path):
                    errors.append(f"permission_ceiling.read_paths[{index}] exceeds the allowed roots")

    if not string_list(data.get("participant_ids"), nonempty_list=True):
        errors.append("participant_ids must be a non-empty string array")

    mode = data.get("invocation_mode")
    if mode == "delegated_by_goal":
        if not nonempty(data.get("goal_id")):
            errors.append("goal_id is required for delegated_by_goal")
        if not nonempty(data.get("contract_revision")):
            errors.append("contract_revision is required for delegated_by_goal")
        if data.get("goal_phase") not in RUNTIME_PHASES:
            errors.append("goal_phase is invalid for delegated_by_goal")
        if kind in {"goal_intake", "goal_brief"}:
            errors.append(
                f"{kind} is formation-owned and cannot use delegated_by_goal"
            )
        if kind == "goal_compile_request" and data.get("goal_phase") != "active":
            errors.append("goal_compile_request requires goal_phase active")
        if kind == "goal_compile_request" and not nonempty(data.get("formation_id")):
            errors.append("goal_compile_request requires source formation_id")
        if kind == "goal_compile_request" and isinstance(payload := data.get("payload"), dict):
            current_contract = payload.get("current_contract")
            if isinstance(current_contract, dict):
                if current_contract.get("goal_id") != data.get("goal_id"):
                    errors.append(
                        "payload.current_contract.goal_id does not match envelope goal_id"
                    )
                if current_contract.get("contract_revision") != data.get(
                    "contract_revision"
                ):
                    errors.append(
                        "payload.current_contract.contract_revision does not match envelope contract_revision"
                    )
        if (
            kind == "goal_compilation"
            and isinstance(payload := data.get("payload"), dict)
            and not string_list(payload.get("user_decision_refs"), nonempty_list=True)
        ):
            errors.append("runtime goal_compilation requires non-empty user_decision_refs")
        if (
            kind == "goal_package"
            and isinstance(payload := data.get("payload"), dict)
            and not string_list(payload.get("user_decision_refs"), nonempty_list=True)
        ):
            errors.append("runtime goal_package requires non-empty user_decision_refs")
    elif mode == "delegated_by_goal_prompt":
        if not nonempty(data.get("formation_id")):
            errors.append("formation_id is required for delegated_by_goal_prompt")
        if data.get("goal_id") not in {None, ""}:
            errors.append("goal_id must be null for delegated_by_goal_prompt")
        if not nonempty(data.get("contract_revision")):
            errors.append("contract_revision is required for delegated_by_goal_prompt")
        if data.get("goal_phase") != "formation":
            errors.append("goal_phase must be formation for delegated_by_goal_prompt")
        if kind not in {
            "research_brief",
            "research_report",
            "design_brief",
            "final_design",
            "integration_record",
            "participant_ledger",
        }:
            errors.append(
                f"{kind} cannot use delegated_by_goal_prompt"
            )
    if kind == "goal_compile_request" and mode != "delegated_by_goal":
        errors.append("goal_compile_request requires delegated_by_goal")
    if kind == "goal_package" and mode == "direct" and data.get("goal_id") not in {
        None,
        "",
    }:
        errors.append("goal_id must be null for a direct goal_package")

    expected_pairs = (
        ("goal_id", args.expected_goal_id),
        ("formation_id", args.expected_formation_id),
        ("contract_revision", args.expected_contract_revision),
        ("input_fingerprint", args.expected_input_fingerprint),
    )
    for field, expected in expected_pairs:
        if expected is not None and data.get(field) != expected:
            errors.append(f"{field} does not match the expected value")

    payload = data.get("payload")
    if (
        isinstance(payload, dict)
        and kind in ARTIFACT_CLASSES
        and payload.get("artifact_class") != ARTIFACT_CLASSES[kind]
    ):
        errors.append(f"payload.artifact_class is invalid for {kind}")
    if isinstance(payload, dict):
        if kind == "research_brief":
            validate_research_brief(payload, errors)
        elif kind == "research_report":
            validate_research_report(payload, errors)
        elif kind == "design_brief":
            validate_design_brief(payload, errors)
        elif kind == "final_design":
            validate_final_design(payload, errors)
        elif kind == "goal_intake":
            validate_goal_intake(payload, errors)
        elif kind == "goal_brief":
            validate_goal_brief(payload, errors)
        elif kind == "goal_compile_request":
            validate_goal_compile_request(payload, errors)
        elif kind == "goal_compilation":
            validate_goal_compilation(payload, errors)
        elif kind == "goal_package":
            validate_goal_package(payload, errors)
        elif kind == "goal_prompt":
            validate_goal_prompt(payload, errors)
        elif kind in {"user_decision_record", "integration_record", "participant_ledger"}:
            validate_other_payload(kind, payload, errors)
        if kind == "research_report" and isinstance(payload.get("participants"), list):
            payload_participants = {
                item.get("participant_id")
                for item in payload["participants"]
                if isinstance(item, dict) and nonempty(item.get("participant_id"))
            }
            if payload_participants != set(data.get("participant_ids", [])):
                errors.append("payload participant IDs must match envelope participant_ids")
        if kind == "goal_package":
            if payload.get("formation_id") != data.get("formation_id"):
                errors.append("payload.formation_id does not match envelope formation_id")
            if payload.get("package_revision") != data.get("contract_revision"):
                errors.append(
                    "payload.package_revision does not match envelope contract_revision"
                )
            ledger_ids = {
                item.get("participant_id")
                for item in payload.get("participant_ledger", [])
                if isinstance(item, dict) and nonempty(item.get("participant_id"))
            }
            if ledger_ids != set(data.get("participant_ids", [])):
                errors.append(
                    "payload participant ledger IDs must match envelope participant_ids"
                )
    return errors


def main() -> int:
    args = parse_args()
    try:
        data = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 2

    errors = validate(data, args)
    if args.request is not None:
        try:
            request = json.loads(args.request.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"INVALID: {exc}")
            return 2
        request_args = copy.copy(args)
        request_args.expected_kind = None
        request_errors = validate(request, request_args)
        errors.extend(f"invalid request: {error}" for error in request_errors)
        validate_binding(request, data, errors)
    if errors:
        for error in errors:
            print(f"INVALID: {error}")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
