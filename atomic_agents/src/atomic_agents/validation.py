"""JSON Schema validation helpers for atomic-agents contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"
SCHEMA_FILENAMES = (
    "atom_contract.schema.json",
    "skeleton.schema.json",
    "lockfile_event.schema.json",
    "approval_summary.schema.json",
    "stop_report.schema.json",
)

SCHEMAS: dict[str, dict[str, Any]] = {}
_VALIDATORS: dict[str, Draft202012Validator] = {}


def _load_schemas() -> None:
    for filename in SCHEMA_FILENAMES:
        schema_path = SCHEMA_DIR / filename
        with schema_path.open("r", encoding="utf-8") as schema_file:
            schema = json.load(schema_file)
        Draft202012Validator.check_schema(schema)
        SCHEMAS[filename] = schema
        _VALIDATORS[filename] = Draft202012Validator(schema)


def _validate(filename: str, data: Any) -> None:
    _VALIDATORS[filename].validate(data)


def validate_atom_contract(data: Any) -> None:
    _validate("atom_contract.schema.json", data)


def validate_skeleton(data: Any) -> None:
    _validate("skeleton.schema.json", data)


def validate_lockfile_event(data: Any) -> None:
    _validate("lockfile_event.schema.json", data)


def validate_approval_summary(data: Any) -> None:
    _validate("approval_summary.schema.json", data)


def validate_stop_report(data: Any) -> None:
    _validate("stop_report.schema.json", data)


def tolerant_read_event(line: str) -> dict[str, Any] | None:
    if not line.strip():
        return None

    event = json.loads(line)
    validate_lockfile_event(event)
    return event


_load_schemas()


__all__ = [
    "SCHEMAS",
    "SCHEMA_DIR",
    "ValidationError",
    "tolerant_read_event",
    "validate_approval_summary",
    "validate_atom_contract",
    "validate_lockfile_event",
    "validate_skeleton",
    "validate_stop_report",
]
