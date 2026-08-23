#!/usr/bin/env python3
"""Import a confirmed GoalPackage into a new file-backed runtime Goal."""

from __future__ import annotations

import argparse
import copy
import json
import re
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_contract import validate as validate_contract
from validate_handoff import validate as validate_handoff

SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--goal-id", required=True)
    return parser.parse_args()


def validation_args(formation_id: str, package_revision: str) -> Namespace:
    return Namespace(
        expected_kind="goal_package",
        expected_goal_id=None,
        expected_formation_id=formation_id,
        expected_contract_revision=package_revision,
        expected_input_fingerprint=None,
        allowed_read_path=[],
        allow_network=True,
    )


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.write_text(content, encoding="utf-8")


def render_plan(milestones: list[dict[str, Any]], status: str) -> str:
    current = "M1" if status == "active" else "无"
    lines = ["# 计划", "", f"当前里程碑：{current}", ""]
    for index, milestone in enumerate(milestones, start=1):
        milestone_id = milestone.get("id") or f"M{index}"
        milestone_status = "in_progress" if status == "active" and index == 1 else "pending"
        lines.extend(
            [
                f"## {milestone_id}: {milestone.get('outcome', '').strip()}",
                "",
                f"Status: {milestone_status}",
                "",
                f"DOD: {', '.join(milestone.get('dod_ids', []))}",
                f"Dependencies: {json.dumps(milestone.get('dependencies', []), ensure_ascii=False)}",
                f"Risks: {json.dumps(milestone.get('risks', []), ensure_ascii=False)}",
                f"Write Scope: {json.dumps(milestone.get('write_scope', []), ensure_ascii=False)}",
                f"Verification: {json.dumps(milestone.get('verification', {}), ensure_ascii=False)}",
                f"Rollback: {milestone.get('rollback', '')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def evidence_document(
    goal_id: str,
    revision: str,
    participant_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "goal_id": goal_id,
        "contract_revision": revision,
        "capability_protocol": "goal-capability/v1",
        "participant_ledger": participant_ledger,
        "revision": "",
        "evidence": [],
        "final_review": {
            "ocr": {
                "verdict": "pending",
                "status": "pending",
                "artifact_revision": "",
                "source": "",
                "warnings": [],
                "comments": [],
                "unresolved_findings": [],
            },
            "subagent": {
                "verdict": "pending",
                "artifact_revision": "",
                "source": "",
                "summary": "",
                "unresolved_findings": [],
            },
        },
        "findings": [],
        "pending_approvals": [],
        "remaining_required_work": [],
    }


def import_package(document: dict[str, Any], *, root: Path, goal_id: str) -> Path:
    if not SLUG_RE.fullmatch(goal_id):
        raise ValueError(
            "goal-id must be lowercase hyphen-case and at most 64 characters"
        )
    if document.get("invocation_mode") != "direct":
        raise ValueError("only direct formation GoalPackages can create a new runtime Goal")
    if document.get("goal_id") not in {None, ""}:
        raise ValueError("initial GoalPackage must not already be bound to a runtime goal_id")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise TypeError("GoalPackage payload must be an object")
    formation_id = payload.get("formation_id")
    package_revision = payload.get("package_revision")
    if not isinstance(formation_id, str) or not isinstance(package_revision, str):
        raise TypeError("GoalPackage formation_id and package_revision are required")
    handoff_errors = validate_handoff(
        document,
        validation_args(formation_id, package_revision),
    )
    if handoff_errors:
        raise ValueError("; ".join(handoff_errors))

    intent = payload["handoff_intent"]
    if intent == "prompt_only":
        raise ValueError("prompt_only GoalPackage cannot create a runtime Goal")
    status = "active" if intent == "activate" else "ready"
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    revision = "contract:1"
    participant_ledger = copy.deepcopy(payload["participant_ledger"])
    for participant in participant_ledger:
        participant["source_package_revision"] = payload["package_revision"]
        participant["contract_revision"] = revision

    contract = copy.deepcopy(payload["contract"])
    contract.update(
        {
            "version": 1,
            "goal_id": goal_id,
            "status": status,
            "contract_revision": revision,
            "final_review_required": True,
            "contract_changes": [
                {
                    "revision": revision,
                    "changed_at": now,
                    "changed_by": "goal-orchestrator",
                    "reason": (
                        f"imported confirmed GoalPackage "
                        f"{payload['formation_id']}@{payload['package_revision']}"
                    ),
                }
            ],
            "source_goal_package": {
                "artifact_id": document["artifact_id"],
                "formation_id": payload["formation_id"],
                "package_revision": payload["package_revision"],
                "input_fingerprint": document["input_fingerprint"],
                "snapshot_refs": document["snapshot_refs"],
                "source_refs": payload["source_refs"],
                "participant_ledger": payload["participant_ledger"],
                "user_confirmation": payload["user_confirmation"],
            },
        }
    )
    contract_errors = validate_contract(contract, allow_draft=False)
    if contract_errors:
        raise ValueError(
            "package contract is not runtime-ready: " + "; ".join(contract_errors)
        )

    goal_dir = root.resolve() / ".goals" / goal_id
    goal_dir.mkdir(parents=True, exist_ok=False)
    try:
        write_new(
            goal_dir / "goal.json",
            json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        )
        write_new(
            goal_dir / "plan.md",
            render_plan(payload["initial_milestones"], status),
        )
        write_new(
            goal_dir / "findings.md",
            "# 发现\n\n"
            f"Source GoalPackage: {payload['formation_id']}@{payload['package_revision']}\n",
        )
        write_new(
            goal_dir / "progress.md",
            f"# 进展\n\n{now} imported confirmed GoalPackage with intent `{intent}`.\n",
        )
        write_new(
            goal_dir / "evidence.json",
            json.dumps(
                evidence_document(goal_id, revision, participant_ledger),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    except Exception:
        for child in goal_dir.iterdir():
            child.unlink()
        goal_dir.rmdir()
        raise
    return goal_dir


def main() -> int:
    args = parse_args()
    try:
        document = json.loads(args.package.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TypeError("GoalPackage root must be an object")
        goal_dir = import_package(document, root=args.root, goal_id=args.goal_id)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"INVALID: {exc}")
        return 1
    print(goal_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
