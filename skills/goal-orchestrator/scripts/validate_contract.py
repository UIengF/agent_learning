#!/usr/bin/env python3
"""Validate the structure and activation readiness of a goal contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
STATUSES = {
    "draft",
    "ready",
    "active",
    "waiting_for_decision",
    "paused",
    "blocked",
    "verifying",
    "complete",
    "cancelled",
    "superseded",
}
UNKNOWN_CLASSES = {"known_unknown", "unknown_known", "unknown_unknown"}
UNKNOWN_RESOLUTIONS = {"inspection", "retrieval", "prototype", "user_decision"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--allow-draft", action="store_true", help="Validate shape without requiring activation-ready content")
    return parser.parse_args()


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def string_list(value: Any) -> bool:
    return isinstance(value, list) and all(nonempty_string(item) for item in value)


def validate(data: Any, allow_draft: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["contract root must be an object"]

    if data.get("version") != 1:
        errors.append("version must be 1")
    if not nonempty_string(data.get("contract_revision")):
        errors.append("contract_revision must be a non-empty string")
    if data.get("final_review_required") is not True:
        errors.append("final_review_required must be true")
    goal_id = data.get("goal_id")
    if not isinstance(goal_id, str) or not SLUG_RE.fullmatch(goal_id):
        errors.append("goal_id must be lowercase hyphen-case and at most 64 characters")
    if data.get("status") not in STATUSES:
        errors.append(f"status must be one of: {', '.join(sorted(STATUSES))}")
    if not nonempty_string(data.get("objective")):
        errors.append("objective must be a non-empty string")
    if not allow_draft and not nonempty_string(data.get("outcome")):
        errors.append("outcome must be non-empty before activation")

    criteria = data.get("definition_of_done")
    if not isinstance(criteria, list):
        errors.append("definition_of_done must be an array")
        criteria = []
    if not allow_draft and not criteria:
        errors.append("definition_of_done must contain at least one criterion before activation")
    seen_ids: set[str] = set()
    for index, item in enumerate(criteria):
        prefix = f"definition_of_done[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        criterion_id = item.get("id")
        if not nonempty_string(criterion_id):
            errors.append(f"{prefix}.id must be non-empty")
        elif criterion_id in seen_ids:
            errors.append(f"{prefix}.id is duplicated: {criterion_id}")
        else:
            seen_ids.add(criterion_id)
        if not nonempty_string(item.get("criterion")):
            errors.append(f"{prefix}.criterion must be non-empty")
        verification = item.get("verification")
        if not isinstance(verification, dict):
            errors.append(f"{prefix}.verification must be an object")
        else:
            for field in ("method", "expected"):
                if not nonempty_string(verification.get(field)):
                    errors.append(f"{prefix}.verification.{field} must be non-empty")
            if "command" in verification and not nonempty_string(verification.get("command")):
                errors.append(f"{prefix}.verification.command must be non-empty when provided")

    for field in ("constraints", "non_goals", "stop_rules"):
        if not string_list(data.get(field)):
            errors.append(f"{field} must be an array of non-empty strings")
    if not allow_draft and not data.get("stop_rules"):
        errors.append("stop_rules must contain at least one rule before activation")

    permissions = data.get("permissions")
    if not isinstance(permissions, dict):
        errors.append("permissions must be an object")
    else:
        permission_count = 0
        for field in ("allowed_without_confirmation", "requires_confirmation", "forbidden"):
            if not string_list(permissions.get(field)):
                errors.append(f"permissions.{field} must be an array of non-empty strings")
            else:
                permission_count += len(permissions[field])
        if not allow_draft and permission_count == 0:
            errors.append("permissions must define at least one boundary before activation")

    unknowns = data.get("unknowns")
    if not isinstance(unknowns, list):
        errors.append("unknowns must be an array")
    else:
        required_unknown_fields = ("id", "class", "question", "impact", "resolution", "status")
        seen_unknown_ids: set[str] = set()
        for index, unknown in enumerate(unknowns):
            prefix = f"unknowns[{index}]"
            if not isinstance(unknown, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field in required_unknown_fields:
                if not nonempty_string(unknown.get(field)):
                    errors.append(f"{prefix}.{field} must be a non-empty string")
            unknown_id = unknown.get("id")
            if nonempty_string(unknown_id):
                if unknown_id in seen_unknown_ids:
                    errors.append(f"{prefix}.id is duplicated: {unknown_id}")
                seen_unknown_ids.add(unknown_id)
            if unknown.get("class") not in UNKNOWN_CLASSES:
                errors.append(f"{prefix}.class is invalid")
            if unknown.get("resolution") not in UNKNOWN_RESOLUTIONS:
                errors.append(f"{prefix}.resolution is invalid")
            if unknown.get("status") not in {"open", "resolved", "accepted_risk"}:
                errors.append(f"{prefix}.status must be open, resolved, or accepted_risk")
            elif unknown.get("status") != "open" and not nonempty_string(unknown.get("evidence")):
                errors.append(f"{prefix}.evidence must be non-empty when resolved or accepted_risk")
    changes = data.get("contract_changes")
    if not isinstance(changes, list) or not changes:
        errors.append("contract_changes must be a non-empty array")
    else:
        for index, change in enumerate(changes):
            prefix = f"contract_changes[{index}]"
            if not isinstance(change, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for field in ("revision", "changed_at", "changed_by", "reason"):
                if not nonempty_string(change.get(field)):
                    errors.append(f"{prefix}.{field} must be non-empty")
        if isinstance(changes[-1], dict) and changes[-1].get("revision") != data.get("contract_revision"):
            errors.append("the latest contract_changes revision must match contract_revision")

    if isinstance(unknowns, list):
        for index, unknown in enumerate(unknowns):
            if not isinstance(unknown, dict) or unknown.get("status") != "accepted_risk":
                continue
            acceptance = unknown.get("acceptance")
            if not isinstance(acceptance, dict):
                errors.append(f"unknowns[{index}].acceptance must be an object for accepted_risk")
                continue
            for field in ("accepted_by", "accepted_at", "scope", "review_condition"):
                if not nonempty_string(acceptance.get(field)):
                    errors.append(f"unknowns[{index}].acceptance.{field} must be non-empty")
    return errors


def main() -> int:
    args = parse_args()
    try:
        data = json.loads(args.contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 2
    errors = validate(data, args.allow_draft)
    if errors:
        for error in errors:
            print(f"INVALID: {error}")
        return 1
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
