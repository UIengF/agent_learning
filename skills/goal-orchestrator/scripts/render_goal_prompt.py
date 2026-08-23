#!/usr/bin/env python3
"""Render a validated Goal contract as a compact /goal export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_contract import validate as validate_contract

RENDERABLE_STATUSES = {"ready", "active", "verifying", "complete"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Emit plain /goal text or a goal_prompt payload",
    )
    parser.add_argument(
        "--target-executor",
        choices=("native_goal", "external_agent", "both"),
        default="external_agent",
    )
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help="Render a non-activatable preview and label it as such",
    )
    return parser.parse_args()


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if nonempty(item)]


def bullet_section(title: str, values: list[str]) -> list[str]:
    if not values:
        return []
    return [f"{title}:", *(f"- {value}" for value in values), ""]


def render(contract: dict[str, Any], *, preview: bool = False) -> str:
    objective = contract["objective"].strip().rstrip("。.")
    lines = [f"/goal {objective}。", ""]
    if preview:
        lines.extend(["Preview: This draft is not approved for activation.", ""])

    outcome = contract.get("outcome", "").strip()
    if outcome:
        lines.extend(["Outcome:", outcome, ""])

    lines.extend(bullet_section("Scope", as_string_list(contract.get("scope"))))
    lines.extend(bullet_section("Excludes", as_string_list(contract.get("non_goals"))))
    lines.extend(bullet_section("Constraints", as_string_list(contract.get("constraints"))))

    active_truth = as_string_list(contract.get("active_truth_refs"))
    if active_truth:
        lines.extend(bullet_section("Active truth", active_truth))

    criteria = contract.get("definition_of_done", [])
    lines.append("Done when:")
    for index, criterion in enumerate(criteria, start=1):
        verification = criterion.get("verification", {})
        evidence = [
            verification.get("method", "").strip(),
            verification.get("command", "").strip(),
            verification.get("expected", "").strip(),
        ]
        evidence_text = "; ".join(item for item in evidence if item)
        suffix = f" Evidence: {evidence_text}." if evidence_text else ""
        lines.append(f"{index}. {criterion['criterion'].strip()}.{suffix}")
    lines.append("")

    permissions = contract.get("permissions", {})
    lines.extend(
        bullet_section(
            "Allowed without confirmation",
            as_string_list(permissions.get("allowed_without_confirmation")),
        )
    )
    lines.extend(
        bullet_section(
            "Requires confirmation",
            as_string_list(permissions.get("requires_confirmation")),
        )
    )
    lines.extend(bullet_section("Forbidden", as_string_list(permissions.get("forbidden"))))

    execution_rules = as_string_list(contract.get("execution_rules"))
    if not execution_rules:
        execution_rules = [
            "Keep the contract stable and change only the implementation plan when evidence requires it",
            "Continue independent work around recoverable waits or permission gaps",
            "Do not report completion until every Done-when item has fresh evidence for the current revision",
        ]
    lines.extend(bullet_section("Execution", execution_rules))
    lines.extend(bullet_section("Stop or ask when", as_string_list(contract.get("stop_rules"))))

    token_budget = contract.get("token_budget")
    if isinstance(token_budget, int) and token_budget > 0:
        lines.append(f"Use a token budget of {token_budget} tokens for this goal.")

    return "\n".join(lines).strip() + "\n"


def check_renderable(contract: Any, *, allow_draft: bool) -> list[str]:
    if not isinstance(contract, dict):
        return ["contract root must be an object"]
    errors = validate_contract(contract, allow_draft=allow_draft)
    if not allow_draft and contract.get("status") not in RENDERABLE_STATUSES:
        errors.append("contract status must be ready, active, verifying, or complete")
    if not allow_draft:
        unknowns = contract.get("unknowns", [])
        if isinstance(unknowns, list):
            for unknown in unknowns:
                if isinstance(unknown, dict) and unknown.get("status") == "open":
                    errors.append(f"open unknown remains: {unknown.get('id', '<unknown>')}")
    return errors


def main() -> int:
    args = parse_args()
    try:
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 2

    errors = check_renderable(contract, allow_draft=args.allow_draft)
    if errors:
        for error in errors:
            print(f"INVALID: {error}")
        return 1

    prompt = render(contract, preview=args.allow_draft)
    if args.format == "text":
        print(prompt, end="")
    else:
        payload = {
            "artifact_class": "rendered_goal_prompt_not_goal_state",
            "prompt": prompt,
            "target_executor": args.target_executor,
            "source_contract_revision": contract["contract_revision"],
            "source_goal_id": contract["goal_id"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
