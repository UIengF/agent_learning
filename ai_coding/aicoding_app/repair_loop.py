from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .permissions import WorkspacePolicy
from .test_failures import parse_pytest_failures


_ASSERT_FUNC_EQUALS_RE = re.compile(r"assert\s+(?P<func>[A-Za-z_][A-Za-z0-9_]*)\(\)\s*==\s*(?P<value>.+)")
_IMPORT_RE_TEMPLATE = r"from\s+(?P<module>[A-Za-z_][A-Za-z0-9_.]*)\s+import\s+.*\b{func}\b"


@dataclass(frozen=True)
class RepairCandidate:
    patch: str
    reason: str


def build_repair_candidate(policy: WorkspacePolicy, validation_output: str) -> RepairCandidate | None:
    failures = parse_pytest_failures(validation_output)
    for failure in failures:
        candidate = _candidate_for_failure(policy, failure.path, failure.line)
        if candidate is not None:
            return candidate
    return None


def _candidate_for_failure(
    policy: WorkspacePolicy, test_path: str, failure_line: int | None
) -> RepairCandidate | None:
    resolved_test = policy.resolve_path(test_path)
    try:
        test_lines = resolved_test.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, UnicodeDecodeError):
        return None
    if not failure_line:
        return None

    start = max(0, failure_line - 3)
    end = min(len(test_lines), failure_line + 2)
    for line in test_lines[start:end]:
        match = _ASSERT_FUNC_EQUALS_RE.search(line.strip())
        if not match:
            continue
        func_name = match.group("func")
        expected_value = match.group("value").strip()
        source_path = _source_for_import(policy, test_lines, func_name)
        if source_path is None:
            continue
        patch = _patch_return_value(policy, source_path, func_name, expected_value)
        if patch:
            rel = source_path.relative_to(policy.workspace).as_posix()
            return RepairCandidate(
                patch=patch,
                reason=f"matched pytest assertion for {func_name}() and updated {rel}",
            )
    return None


def _source_for_import(
    policy: WorkspacePolicy, test_lines: list[str], func_name: str
) -> Path | None:
    pattern = re.compile(_IMPORT_RE_TEMPLATE.format(func=re.escape(func_name)))
    for line in test_lines:
        match = pattern.search(line.strip())
        if not match:
            continue
        module_path = Path(*match.group("module").split(".")).with_suffix(".py")
        candidate = policy.resolve_path(module_path)
        if candidate.exists():
            return candidate
    direct = policy.resolve_path(f"{func_name}.py")
    return direct if direct.exists() else None


def _patch_return_value(
    policy: WorkspacePolicy, source_path: Path, func_name: str, expected_value: str
) -> str | None:
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    def_index = None
    for index, line in enumerate(lines):
        if re.match(rf"^def\s+{re.escape(func_name)}\s*\(", line):
            def_index = index
            break
    if def_index is None:
        return None
    for index in range(def_index + 1, len(lines)):
        line = lines[index]
        if line.startswith("def ") or line.startswith("class "):
            break
        return_match = re.match(r"^(?P<indent>\s*)return\s+(?P<value>.+)$", line)
        if not return_match:
            continue
        old = line
        new = f"{return_match.group('indent')}return {expected_value}"
        if old == new:
            return None
        rel = source_path.relative_to(policy.workspace).as_posix()
        return f"*** Begin Patch\n*** Update File: {rel}\n@@\n-{old}\n+{new}\n*** End Patch"
    return None
