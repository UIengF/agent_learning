from __future__ import annotations

from pathlib import Path

from .repo_map import build_repo_map
from .test_failures import parse_pytest_failures


def explain_context(workspace: str | Path, query: str) -> str:
    repo_map = build_repo_map(workspace)
    failures = parse_pytest_failures(query)
    if failures:
        lines = ["Pytest failure context:"]
        for failure in failures:
            lines.append(f"- {failure.format()}")
            lines.append(repo_map.explain(failure.path))
        return "\n\n".join(lines)
    return repo_map.explain(query)
