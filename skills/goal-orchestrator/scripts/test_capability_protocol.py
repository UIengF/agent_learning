#!/usr/bin/env python3
"""Focused regression tests for goal-capability/v1 validation and audit gates."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from audit_completion import audit_capability_protocol
from render_goal_prompt import check_renderable, render
from validate_handoff import validate, validate_binding

FINGERPRINT = "sha256:" + "a" * 64
REVISION = "contract:2"
ARTIFACT_REVISION = "git:abc123"
FORMATION_ID = "formation-example"


def args(**overrides: object) -> Namespace:
    values = {
        "expected_kind": None,
        "expected_goal_id": None,
        "expected_formation_id": None,
        "expected_contract_revision": None,
        "expected_input_fingerprint": None,
        "allowed_read_path": ["/Users/uleng/Code"],
        "allow_network": False,
    }
    values.update(overrides)
    return Namespace(**values)


def research_brief() -> dict[str, object]:
    return {
        "protocol": "goal-capability/v1",
        "artifact_kind": "research_brief",
        "artifact_id": "research-brief-1",
        "request_id": "request-1",
        "invocation_mode": "delegated_by_goal",
        "goal_id": "example-goal",
        "contract_revision": REVISION,
        "goal_phase": "active",
        "input_fingerprint": FINGERPRINT,
        "snapshot_refs": [
            {"locator": "/Users/uleng/Code/example", "revision_or_hash": "git:abc123"}
        ],
        "permission_ceiling": {
            "read_paths": ["/Users/uleng/Code/example"],
            "write_allowed": False,
            "network_allowed": False,
        },
        "participant_ids": ["/root/researcher-1"],
        "created_at": "2026-07-17T08:00:00Z",
        "payload": {
            "purpose": "Establish the factual baseline for the Goal.",
            "questions": [
                {
                    "id": "RQ-1",
                    "question": "What is the current behavior?",
                    "decision_impact": "Determines whether design work is needed.",
                }
            ],
            "scope": ["/Users/uleng/Code/example"],
            "exclusions": [],
            "evidence_policy": {"authoritative_preferred": True},
            "stop_conditions": [],
            "known_facts": [],
            "known_unknowns": ["current behavior"],
            "workspace_rule": {
                "cwd": "/Users/uleng/Code",
                "explicit_paths_required": True,
            },
        },
    }


def design_brief() -> dict[str, object]:
    artifact = research_brief()
    artifact.update(
        {
            "artifact_kind": "design_brief",
            "artifact_id": "design-brief-1",
            "request_id": "design-request-1",
            "participant_ids": ["/root/design-lead"],
            "payload": {
                "problem": "Choose a stable module boundary.",
                "desired_outcome": "One implementation-ready architecture.",
                "confirmed_facts": [],
                "assumptions": [],
                "known_unknowns": [],
                "hard_constraints": ["read-only design"],
                "preferences": [],
                "non_goals": [],
                "authorized_scope": ["/Users/uleng/Code/example"],
                "evaluation_criteria": ["maintainability", "migration risk"],
                "research_artifact_refs": [],
                "decision_record_refs": [],
                "decisions_reserved_for_user": ["product_intent"],
                "required_deliverable": "one implementation-ready design",
            },
        }
    )
    return artifact


def formation_research_brief() -> dict[str, object]:
    artifact = research_brief()
    artifact.update(
        {
            "invocation_mode": "delegated_by_goal_prompt",
            "goal_id": None,
            "formation_id": FORMATION_ID,
            "contract_revision": "package:1",
            "goal_phase": "formation",
        }
    )
    return artifact


def final_design() -> dict[str, object]:
    artifact = design_brief()
    artifact.update(
        {
            "artifact_kind": "final_design",
            "artifact_id": "final-design-1",
            "participant_ids": [
                "/root/design-lead",
                "/root/design-lead/architect-a",
                "/root/design-lead/architect-b",
                "/root/design-lead/synthesizer",
            ],
            "payload": {
                "artifact_class": "adjudicated_design_not_implementation_evidence",
                "chosen_architecture": {"summary": "Selected module boundary."},
                "modules_and_interfaces": [],
                "data_flows": [],
                "invariants": [],
                "failure_handling": [],
                "security_and_performance": [],
                "migration": [],
                "tests_and_acceptance": [],
                "risks": [],
                "decision_records": [],
                "rejected_alternatives": [],
                "unresolved": [],
                "candidate_coverage": [
                    {"candidate_id": "A", "disposition": "selected", "reason": "best fit"},
                    {"candidate_id": "B", "disposition": "rejected", "reason": "higher risk"},
                ],
            },
        }
    )
    return artifact


def goal_intake() -> dict[str, object]:
    artifact = research_brief()
    artifact.update(
        {
            "artifact_kind": "goal_intake",
            "artifact_id": "goal-intake-1",
            "request_id": "goal-brief-request-1",
            "invocation_mode": "direct",
            "goal_id": None,
            "formation_id": FORMATION_ID,
            "contract_revision": "package:1",
            "participant_ids": ["/root/goal-prompt"],
            "payload": {
                "user_intent": "Add a verified example capability.",
                "scope_hints": ["/Users/uleng/Code/example"],
                "known_facts": [],
                "known_unknowns": ["current behavior"],
                "requested_mode": "brief",
            },
        }
    )
    return artifact


def goal_brief() -> dict[str, object]:
    artifact = goal_intake()
    artifact.update(
        {
            "artifact_kind": "goal_brief",
            "artifact_id": "goal-brief-1",
            "payload": {
                "artifact_class": "advisory_goal_brief_not_goal_state",
                "objective": "Produce a verified example outcome.",
                "outcome": "The example outcome is observable and verified.",
                "scope": ["/Users/uleng/Code/example"],
                "exclusions": [],
                "constraints": [],
                "definition_of_done": [
                    {
                        "id": "DOD-1",
                        "criterion": "The example check passes.",
                        "verification": {"method": "test", "expected": "exit 0"},
                    }
                ],
                "unknowns": [],
                "recommended_mode": "fast",
                "initialization": [],
                "unresolved_questions": [],
                "active_truth_refs": [],
                "orchestration_policy": {
                    "mode": "auto",
                    "allow": ["research", "design"],
                    "require": [],
                    "ask_before_delegate": False,
                },
            },
        }
    )
    return artifact


def goal_compile_request() -> dict[str, object]:
    artifact = goal_intake()
    artifact.update(
        {
            "artifact_kind": "goal_compile_request",
            "artifact_id": "goal-compile-request-1",
            "request_id": "goal-compile-request-1",
            "invocation_mode": "delegated_by_goal",
            "goal_id": "example-goal",
            "formation_id": FORMATION_ID,
            "contract_revision": REVISION,
            "goal_phase": "active",
            "payload": {
                "goal_brief": goal_brief()["payload"],
                "current_contract": valid_contract(),
                "changed_fields": ["scope"],
                "research_refs": [],
                "design_refs": [],
                "user_decision_refs": ["decision-scope-1"],
                "target_executor": "native_goal",
                "render_prompt": False,
            },
        }
    )
    return artifact


def goal_compilation() -> dict[str, object]:
    artifact = goal_compile_request()
    artifact.update(
        {
            "artifact_kind": "goal_compilation",
            "artifact_id": "goal-compilation-1",
            "payload": {
                "artifact_class": "candidate_goal_contract_not_goal_state",
                "contract": valid_contract(),
                "changed_fields": ["scope"],
                "user_decision_refs": ["decision-scope-1"],
                "source_refs": ["goal-brief-1", "decision-scope-1"],
                "compilation_warnings": [],
                "unresolved": [],
            },
        }
    )
    return artifact


def goal_package() -> dict[str, object]:
    artifact = formation_research_brief()
    artifact.update(
        {
            "artifact_kind": "goal_package",
            "artifact_id": "goal-package-1",
            "request_id": "goal-package-1",
            "invocation_mode": "direct",
            "contract_revision": "package:2",
            "participant_ids": ["/root/goal-prompt"],
            "payload": {
                "artifact_class": "approved_goal_package_not_goal_state",
                "formation_id": FORMATION_ID,
                "package_revision": "package:2",
                "contract": {
                    "objective": "Produce a verified example outcome.",
                    "outcome": "The example outcome is observable and verified.",
                    "definition_of_done": [
                        {
                            "id": "DOD-1",
                            "criterion": "The example check passes.",
                            "verification": {"method": "test", "expected": "exit 0"},
                        }
                    ],
                },
                "initial_solution": {
                    "summary": "Use the existing example validation path.",
                    "rejected_alternatives": [],
                },
                "initial_milestones": [
                    {
                        "id": "M1",
                        "outcome": "The example check is implemented and verified.",
                        "dod_ids": ["DOD-1"],
                        "dependencies": [],
                        "assumptions": [],
                        "write_scope": ["/Users/uleng/Code/example"],
                        "verification": {"method": "test", "expected": "exit 0"},
                        "risks": [],
                        "rollback": "restore the prior verified revision",
                    }
                ],
                "changed_fields": [
                    "objective",
                    "outcome",
                    "definition_of_done",
                    "initial_solution",
                    "initial_milestones",
                ],
                "user_decision_refs": [],
                "source_refs": ["goal-brief-1"],
                "participant_ledger": [
                    {
                        "participant_id": "/root/goal-prompt",
                        "role": "formation_owner",
                        "lineage": "/root/goal-prompt",
                        "artifact_ids": ["goal-package-1"],
                        "history_inherited": True,
                        "started_at": "2026-08-10T07:00:00Z",
                        "ended_at": "2026-08-10T08:00:00Z",
                    }
                ],
                "user_confirmation": {
                    "confirmed": True,
                    "statement_or_ref": "user confirmed the package",
                    "confirmed_at": "2026-08-10T08:00:00Z",
                    "creation_authorized": True,
                    "start_authorized": True,
                },
                "handoff_intent": "activate",
                "unresolved": [],
            },
        }
    )
    return artifact


def runtime_ready_goal_package(*, intent: str = "activate") -> dict[str, object]:
    artifact = goal_package()
    artifact["payload"]["contract"] = valid_contract()  # type: ignore[index]
    artifact["payload"]["handoff_intent"] = intent  # type: ignore[index]
    confirmation = artifact["payload"]["user_confirmation"]  # type: ignore[index]
    confirmation["creation_authorized"] = intent in {"ready", "activate"}  # type: ignore[index]
    confirmation["start_authorized"] = intent == "activate"  # type: ignore[index]
    artifact["payload"]["initial_milestones"] = [  # type: ignore[index]
        {
            "id": "M1",
            "outcome": "The example check is implemented and verified.",
            "dod_ids": ["DOD-1"],
            "dependencies": [],
            "assumptions": [],
            "write_scope": ["/Users/uleng/Code/example"],
            "verification": {"method": "test", "expected": "exit 0"},
            "risks": [],
            "rollback": "restore the prior verified revision",
        }
    ]
    return artifact


def protocol_evidence() -> dict[str, object]:
    return {
        "capability_protocol": "goal-capability/v1",
        "evidence": [{"evidence_kind": "implementation_evidence"}],
        "final_review": {
            "subagent": {
                "participant_id": "final-reviewer",
                "lineage": "/root/final-reviewer",
            }
        },
        "participant_ledger": [
            {
                "participant_id": "implementer",
                "role": "implementer",
                "lineage": "/root/implementer",
                "contract_revision": REVISION,
                "artifact_ids": ["implementation-1"],
                "history_inherited": True,
                "started_at": "2026-07-17T08:00:00Z",
                "ended_at": "2026-07-17T08:05:00Z",
            },
            {
                "participant_id": "final-reviewer",
                "role": "final_reviewer",
                "lineage": "/root/final-reviewer",
                "contract_revision": REVISION,
                "artifact_ids": ["review-1"],
                "history_inherited": False,
                "started_at": "2026-07-17T08:06:00Z",
                "ended_at": "2026-07-17T08:10:00Z",
            },
        ],
    }


def valid_contract() -> dict[str, object]:
    return {
        "version": 1,
        "goal_id": "example-goal",
        "status": "verifying",
        "contract_revision": REVISION,
        "objective": "Produce a verified example outcome.",
        "outcome": "The example outcome is observable and verified.",
        "definition_of_done": [
            {
                "id": "DOD-1",
                "criterion": "The example check passes.",
                "verification": {"method": "test", "expected": "exit 0"},
            }
        ],
        "constraints": [],
        "non_goals": [],
        "permissions": {
            "allowed_without_confirmation": ["read local files"],
            "requires_confirmation": [],
            "forbidden": [],
        },
        "unknowns": [],
        "stop_rules": ["complete only with fresh evidence"],
        "final_review_required": True,
        "contract_changes": [
            {
                "revision": REVISION,
                "changed_at": "2026-07-17T08:00:00Z",
                "changed_by": "user",
                "reason": "test fixture",
            }
        ],
    }


def valid_evidence(*, protocol: bool) -> dict[str, object]:
    checked_at = "2026-07-17T08:05:30Z"
    document: dict[str, object] = {
        "goal_id": "example-goal",
        "contract_revision": REVISION,
        "revision": ARTIFACT_REVISION,
        "evidence": [
            {
                "criterion_id": "DOD-1",
                "status": "pass",
                "checked_at": checked_at,
                "contract_revision": REVISION,
                "artifact_revision": ARTIFACT_REVISION,
                "source": "test fixture",
                "summary": "check passed",
            }
        ],
        "final_review": {
            "ocr": {
                "verdict": "not_applicable",
                "status": "skipped",
                "reason": "non-code fixture",
                "source": "fixture applicability check",
            },
            "subagent": {
                "verdict": "pass",
                "artifact_revision": ARTIFACT_REVISION,
                "source": "independent fixture review",
                "summary": "no unresolved issue",
                "unresolved_findings": [],
            },
        },
        "findings": [],
        "pending_approvals": [],
        "remaining_required_work": [],
    }
    if protocol:
        protocol_fields = protocol_evidence()
        document["capability_protocol"] = protocol_fields["capability_protocol"]
        document["participant_ledger"] = protocol_fields["participant_ledger"]
        document["evidence"][0]["evidence_kind"] = "implementation_evidence"  # type: ignore[index]
        document["final_review"]["subagent"].update(  # type: ignore[index]
            protocol_fields["final_review"]["subagent"]  # type: ignore[index]
        )
    return document


class HandoffValidationTests(unittest.TestCase):
    def test_valid_delegated_research_brief(self) -> None:
        self.assertEqual(validate(research_brief(), args()), [])

    def test_runtime_delegation_rejects_formation_phase(self) -> None:
        artifact = research_brief()
        artifact["goal_phase"] = "formation"
        errors = validate(artifact, args())
        self.assertIn("goal_phase is invalid for delegated_by_goal", errors)

    def test_valid_formation_research_brief(self) -> None:
        self.assertEqual(
            validate(
                formation_research_brief(),
                args(expected_formation_id=FORMATION_ID),
            ),
            [],
        )

    def test_formation_delegation_rejects_runtime_goal_id(self) -> None:
        artifact = formation_research_brief()
        artifact["goal_id"] = "example-goal"
        errors = validate(artifact, args())
        self.assertIn("goal_id must be null for delegated_by_goal_prompt", errors)

    def test_formation_delegation_requires_formation_id(self) -> None:
        artifact = formation_research_brief()
        artifact["formation_id"] = None
        errors = validate(artifact, args())
        self.assertIn("formation_id is required for delegated_by_goal_prompt", errors)

    def test_runtime_owner_cannot_delegate_goal_intake(self) -> None:
        artifact = goal_intake()
        artifact.update(
            {
                "invocation_mode": "delegated_by_goal",
                "goal_id": "example-goal",
                "formation_id": None,
                "contract_revision": REVISION,
                "goal_phase": "active",
            }
        )
        errors = validate(artifact, args())
        self.assertIn(
            "goal_intake is formation-owned and cannot use delegated_by_goal",
            errors,
        )

    def test_rejects_write_permission(self) -> None:
        artifact = research_brief()
        artifact["permission_ceiling"]["write_allowed"] = True  # type: ignore[index]
        errors = validate(artifact, args())
        self.assertIn("permission_ceiling.write_allowed must be false", errors)

    def test_rejects_revision_mismatch(self) -> None:
        errors = validate(
            research_brief(),
            args(expected_contract_revision="contract:3"),
        )
        self.assertIn("contract_revision does not match the expected value", errors)

    def test_valid_delegated_design_brief(self) -> None:
        self.assertEqual(validate(design_brief(), args()), [])

    def test_valid_final_design(self) -> None:
        self.assertEqual(validate(final_design(), args()), [])

    def test_final_design_binding_matches_brief(self) -> None:
        errors: list[str] = []
        validate_binding(design_brief(), final_design(), errors)
        self.assertEqual(errors, [])

    def test_rejects_stale_response_binding(self) -> None:
        response = final_design()
        response["contract_revision"] = "contract:3"
        errors: list[str] = []
        validate_binding(design_brief(), response, errors)
        self.assertIn("response contract_revision does not match the request binding", errors)

    def test_goal_brief_and_compilation_bindings(self) -> None:
        self.assertEqual(validate(goal_intake(), args()), [])
        self.assertEqual(validate(goal_brief(), args()), [])
        self.assertEqual(validate(goal_compile_request(), args()), [])
        self.assertEqual(validate(goal_compilation(), args()), [])
        brief_errors: list[str] = []
        validate_binding(goal_intake(), goal_brief(), brief_errors)
        self.assertEqual(brief_errors, [])
        compilation_errors: list[str] = []
        validate_binding(goal_compile_request(), goal_compilation(), compilation_errors)
        self.assertEqual(compilation_errors, [])

    def test_runtime_compile_requires_user_decision_and_changed_fields(self) -> None:
        artifact = goal_compile_request()
        artifact["payload"]["user_decision_refs"] = []  # type: ignore[index]
        artifact["payload"]["changed_fields"] = []  # type: ignore[index]
        errors = validate(artifact, args())
        self.assertIn(
            "payload.user_decision_refs must be a non-empty string array",
            errors,
        )
        self.assertIn("payload.changed_fields must be a non-empty string array", errors)

    def test_runtime_compile_binds_current_contract_identity(self) -> None:
        artifact = goal_compile_request()
        artifact["payload"]["current_contract"]["goal_id"] = "other-goal"  # type: ignore[index]
        artifact["payload"]["current_contract"]["contract_revision"] = "contract:1"  # type: ignore[index]
        errors = validate(artifact, args())
        self.assertIn(
            "payload.current_contract.goal_id does not match envelope goal_id",
            errors,
        )
        self.assertIn(
            "payload.current_contract.contract_revision does not match envelope contract_revision",
            errors,
        )

    def test_runtime_compilation_binds_changed_fields_and_decisions(self) -> None:
        response = goal_compilation()
        response["payload"]["changed_fields"] = ["outcome"]  # type: ignore[index]
        errors: list[str] = []
        validate_binding(goal_compile_request(), response, errors)
        self.assertIn(
            "response payload.changed_fields does not match the request binding",
            errors,
        )

    def test_runtime_compilation_rejects_undeclared_contract_change(self) -> None:
        response = goal_compilation()
        response["payload"]["contract"]["outcome"] = "A different outcome."  # type: ignore[index]
        errors: list[str] = []
        validate_binding(goal_compile_request(), response, errors)
        self.assertIn("response contract changes undeclared fields: outcome", errors)

    def test_valid_confirmed_goal_package(self) -> None:
        self.assertEqual(validate(goal_package(), args()), [])

    def test_direct_goal_package_rejects_runtime_goal_id(self) -> None:
        artifact = goal_package()
        artifact["goal_id"] = "example-goal"
        errors = validate(artifact, args())
        self.assertIn("goal_id must be null for a direct goal_package", errors)

    def test_runtime_goal_package_requires_user_decision(self) -> None:
        artifact = goal_package()
        artifact.update(
            {
                "invocation_mode": "delegated_by_goal",
                "goal_id": "example-goal",
                "goal_phase": "active",
            }
        )
        artifact["payload"]["user_decision_refs"] = []  # type: ignore[index]
        errors = validate(artifact, args())
        self.assertIn(
            "runtime goal_package requires non-empty user_decision_refs",
            errors,
        )

    def test_goal_package_binds_envelope_and_payload_identity(self) -> None:
        artifact = goal_package()
        artifact["formation_id"] = "other-formation"
        artifact["contract_revision"] = "package:999"
        errors = validate(artifact, args())
        self.assertIn("payload.formation_id does not match envelope formation_id", errors)
        self.assertIn(
            "payload.package_revision does not match envelope contract_revision",
            errors,
        )

    def test_goal_package_requires_confirmed_user_approval(self) -> None:
        artifact = goal_package()
        artifact["payload"]["user_confirmation"]["confirmed"] = False  # type: ignore[index]
        errors = validate(artifact, args())
        self.assertIn("payload.user_confirmation.confirmed must be true", errors)

    def test_activate_goal_package_requires_start_authorization(self) -> None:
        artifact = goal_package()
        artifact["payload"]["user_confirmation"]["start_authorized"] = False  # type: ignore[index]
        errors = validate(artifact, args())
        self.assertIn(
            "activate requires creation_authorized=true and start_authorized=true",
            errors,
        )

    def test_goal_package_renderer_uses_confirmed_package(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "goal-prompt"
            / "scripts"
            / "render_goal_package.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "goal-package.json"
            package_path.write_text(json.dumps(goal_package()), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script), str(package_path), "--format", "json"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["artifact_class"], "rendered_goal_prompt_not_goal_state")
            self.assertEqual(payload["source_formation_id"], FORMATION_ID)
            self.assertEqual(payload["source_package_revision"], "package:2")
            self.assertIn("Initial milestones:", payload["prompt"])

    def test_importer_creates_runtime_goal_from_activate_package(self) -> None:
        script = Path(__file__).with_name("import_goal_package.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "goal-package.json"
            package_path.write_text(
                json.dumps(runtime_ready_goal_package()),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(package_path),
                    "--root",
                    temp_dir,
                    "--goal-id",
                    "imported-example",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            contract = json.loads(
                (
                    Path(temp_dir)
                    / ".goals"
                    / "imported-example"
                    / "goal.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(contract["status"], "active")
            self.assertEqual(contract["source_goal_package"]["formation_id"], FORMATION_ID)
            evidence = json.loads(
                (
                    Path(temp_dir)
                    / ".goals"
                    / "imported-example"
                    / "evidence.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(evidence["capability_protocol"], "goal-capability/v1")
            self.assertEqual(evidence["participant_ledger"][0]["role"], "formation_owner")

    def test_importer_keeps_ready_milestones_pending(self) -> None:
        script = Path(__file__).with_name("import_goal_package.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "goal-package.json"
            package_path.write_text(
                json.dumps(runtime_ready_goal_package(intent="ready")),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(package_path),
                    "--root",
                    temp_dir,
                    "--goal-id",
                    "ready-example",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = (
                Path(temp_dir) / ".goals" / "ready-example" / "plan.md"
            ).read_text(encoding="utf-8")
            self.assertIn("当前里程碑：无", plan)
            self.assertNotIn("Status: in_progress", plan)

    def test_importer_rejects_runtime_goal_package_revision(self) -> None:
        script = Path(__file__).with_name("import_goal_package.py")
        artifact = runtime_ready_goal_package()
        artifact.update(
            {
                "invocation_mode": "delegated_by_goal",
                "goal_id": "existing-goal",
                "goal_phase": "active",
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "goal-package.json"
            package_path.write_text(json.dumps(artifact), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(package_path),
                    "--root",
                    temp_dir,
                    "--goal-id",
                    "new-goal-from-revision",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("only direct formation GoalPackages", result.stdout)

    def test_importer_rejects_prompt_only_package(self) -> None:
        script = Path(__file__).with_name("import_goal_package.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "goal-package.json"
            package_path.write_text(
                json.dumps(runtime_ready_goal_package(intent="prompt_only")),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(package_path),
                    "--root",
                    temp_dir,
                    "--goal-id",
                    "prompt-only-example",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt_only GoalPackage cannot create", result.stdout)

    def test_goal_package_renderer_requires_confirmation_reference(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "goal-prompt"
            / "scripts"
            / "render_goal_package.py"
        )
        artifact = goal_package()
        artifact["payload"]["user_confirmation"]["statement_or_ref"] = ""  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "goal-package.json"
            package_path.write_text(json.dumps(artifact), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script), str(package_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("statement_or_ref", result.stdout)

    def test_goal_package_renderer_rejects_raw_payload(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / "goal-prompt"
            / "scripts"
            / "render_goal_package.py"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "goal-package.json"
            package_path.write_text(
                json.dumps(goal_package()["payload"]),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(script), str(package_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("canonical goal_package envelope", result.stdout)

    def test_goal_prompt_renderer_uses_validated_contract(self) -> None:
        contract = valid_contract()
        self.assertEqual(check_renderable(contract, allow_draft=False), [])
        prompt = render(contract)
        self.assertIn("/goal Produce a verified example outcome。", prompt)
        self.assertIn("Done when:", prompt)
        self.assertIn("Evidence: test; exit 0.", prompt)

    def test_goal_prompt_renderer_rejects_draft_by_default(self) -> None:
        contract = valid_contract()
        contract["status"] = "draft"
        errors = check_renderable(contract, allow_draft=False)
        self.assertIn("contract status must be ready, active, verifying, or complete", errors)

    def test_handoff_cli_validates_goal_brief_binding(self) -> None:
        script = Path(__file__).with_name("validate_handoff.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            request_path = Path(temp_dir) / "goal-intake.json"
            response_path = Path(temp_dir) / "goal-brief.json"
            request_path.write_text(json.dumps(goal_intake()), encoding="utf-8")
            response_path.write_text(json.dumps(goal_brief()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(response_path),
                    "--request",
                    str(request_path),
                    "--expected-kind",
                    "goal_brief",
                    "--allowed-read-path",
                    "/Users/uleng/Code",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_renderer_cli_emits_goal_prompt_payload(self) -> None:
        script = Path(__file__).with_name("render_goal_prompt.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            contract_path = Path(temp_dir) / "goal.json"
            contract_path.write_text(json.dumps(valid_contract()), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(script), str(contract_path), "--format", "json"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["artifact_class"], "rendered_goal_prompt_not_goal_state")
            self.assertEqual(payload["source_contract_revision"], REVISION)


class CompletionProtocolTests(unittest.TestCase):
    def test_valid_independent_final_reviewer(self) -> None:
        failures: list[str] = []
        audit_capability_protocol(protocol_evidence(), REVISION, failures)
        self.assertEqual(failures, [])

    def test_rejects_prior_participant_as_final_reviewer(self) -> None:
        document = protocol_evidence()
        document["participant_ledger"][1]["lineage"] = "/root/implementer"  # type: ignore[index]
        document["final_review"]["subagent"]["lineage"] = "/root/implementer"  # type: ignore[index]
        failures: list[str] = []
        audit_capability_protocol(document, REVISION, failures)
        self.assertIn("final_reviewer lineage overlaps a prior Goal participant", failures)

    def test_rejects_advisory_artifact_as_dod_evidence(self) -> None:
        document = copy.deepcopy(protocol_evidence())
        document["evidence"][0]["evidence_kind"] = "final_design"  # type: ignore[index]
        failures: list[str] = []
        audit_capability_protocol(document, REVISION, failures)
        self.assertTrue(any("cannot satisfy a DOD" in item for item in failures))

    def test_rejects_descendant_lineage_as_final_reviewer(self) -> None:
        document = protocol_evidence()
        document["participant_ledger"][1]["lineage"] = "/root/implementer/final"  # type: ignore[index]
        document["final_review"]["subagent"]["lineage"] = "/root/implementer/final"  # type: ignore[index]
        failures: list[str] = []
        audit_capability_protocol(document, REVISION, failures)
        self.assertIn("final_reviewer lineage overlaps a prior Goal participant", failures)

    def test_rejects_noncanonical_final_reviewer_lineage(self) -> None:
        for lineage in (
            "/root//implementer",
            " /root/implementer",
            "/root/implementer ",
            "/root/implementer/",
        ):
            with self.subTest(lineage=lineage):
                document = protocol_evidence()
                document["participant_ledger"][1]["lineage"] = lineage  # type: ignore[index]
                document["final_review"]["subagent"]["lineage"] = lineage  # type: ignore[index]
                failures: list[str] = []
                audit_capability_protocol(document, REVISION, failures)
                self.assertTrue(any("lineage must be canonical" in item for item in failures))

    def test_final_reviewer_starts_after_final_dod_evidence(self) -> None:
        failures: list[str] = []
        audit_capability_protocol(
            protocol_evidence(),
            REVISION,
            failures,
            final_evidence_at=datetime.fromisoformat("2026-07-17T08:07:00+00:00"),
        )
        self.assertIn(
            "final_reviewer did not start after final DOD evidence was recorded",
            failures,
        )

    def test_final_reviewer_cannot_start_at_final_dod_timestamp(self) -> None:
        script = Path(__file__).with_name("audit_completion.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            contract_path = Path(temp_dir) / "goal.json"
            evidence_path = Path(temp_dir) / "evidence.json"
            document = valid_evidence(protocol=True)
            checked_at = document["evidence"][0]["checked_at"]  # type: ignore[index]
            document["participant_ledger"][1]["started_at"] = checked_at  # type: ignore[index]
            contract_path.write_text(json.dumps(valid_contract()), encoding="utf-8")
            evidence_path.write_text(json.dumps(document), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(contract_path),
                    str(evidence_path),
                    "--expected-revision",
                    ARTIFACT_REVISION,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "final_reviewer did not start after final DOD evidence was recorded",
            result.stdout,
        )

    def test_final_reviewer_requires_artifact_reference(self) -> None:
        document = protocol_evidence()
        document["participant_ledger"][1]["artifact_ids"] = []  # type: ignore[index]
        failures: list[str] = []
        audit_capability_protocol(document, REVISION, failures)
        self.assertIn(
            "participant_ledger[1].artifact_ids must be a non-empty string array",
            failures,
        )

    def test_final_reviewer_requires_explicit_artifact_revision(self) -> None:
        script = Path(__file__).with_name("audit_completion.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            contract_path = Path(temp_dir) / "goal.json"
            evidence_path = Path(temp_dir) / "evidence.json"
            document = valid_evidence(protocol=True)
            del document["final_review"]["subagent"]["artifact_revision"]  # type: ignore[index]
            contract_path.write_text(json.dumps(valid_contract()), encoding="utf-8")
            evidence_path.write_text(json.dumps(document), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(contract_path),
                    str(evidence_path),
                    "--expected-revision",
                    ARTIFACT_REVISION,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("final_review.subagent.artifact_revision is missing", result.stdout)

    def test_rejects_completion_review_as_dod_evidence(self) -> None:
        document = copy.deepcopy(protocol_evidence())
        document["evidence"][0]["evidence_kind"] = "completion_review"  # type: ignore[index]
        failures: list[str] = []
        audit_capability_protocol(document, REVISION, failures)
        self.assertTrue(any("cannot satisfy a DOD" in item for item in failures))

    def test_missing_capability_protocol_is_rejected(self) -> None:
        failures: list[str] = []
        audit_capability_protocol({"evidence": []}, REVISION, failures)
        self.assertIn("capability_protocol must be goal-capability/v1", failures)

    def test_full_audit_requires_protocol_v1(self) -> None:
        script = Path(__file__).with_name("audit_completion.py")
        for protocol in (False, True):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as temp_dir:
                contract_path = Path(temp_dir) / "goal.json"
                evidence_path = Path(temp_dir) / "evidence.json"
                contract_path.write_text(json.dumps(valid_contract()), encoding="utf-8")
                evidence_path.write_text(json.dumps(valid_evidence(protocol=protocol)), encoding="utf-8")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        str(contract_path),
                        str(evidence_path),
                        "--expected-revision",
                        ARTIFACT_REVISION,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if protocol:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("capability_protocol must be", result.stdout)

    def test_direction_reviewer_must_be_independent(self) -> None:
        document = protocol_evidence()
        document["participant_ledger"].insert(  # type: ignore[index]
            1,
            {
                "participant_id": "direction-reviewer",
                "role": "direction_reviewer",
                "lineage": "/root/implementer",
                "contract_revision": REVISION,
                "artifact_ids": ["direction-review-1"],
                "history_inherited": False,
                "started_at": "2026-07-17T08:05:00Z",
                "ended_at": "2026-07-17T08:06:00Z",
            },
        )
        failures: list[str] = []
        audit_capability_protocol(document, REVISION, failures)
        self.assertIn(
            "direction_reviewer direction-reviewer lineage overlaps another Goal participant",
            failures,
        )

    def test_direction_reviewer_lineage_must_be_canonical(self) -> None:
        document = protocol_evidence()
        document["participant_ledger"].insert(  # type: ignore[index]
            1,
            {
                "participant_id": "direction-reviewer",
                "role": "direction_reviewer",
                "lineage": "/root//implementer",
                "contract_revision": REVISION,
                "artifact_ids": ["direction-review-1"],
                "history_inherited": False,
                "started_at": "2026-07-17T08:05:00Z",
                "ended_at": "2026-07-17T08:05:30Z",
            },
        )
        failures: list[str] = []
        audit_capability_protocol(document, REVISION, failures)
        self.assertIn(
            "participant_ledger[1].lineage must be canonical",
            failures,
        )


if __name__ == "__main__":
    unittest.main()
