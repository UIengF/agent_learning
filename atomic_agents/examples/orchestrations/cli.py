"""Unified CLI for example orchestration templates."""

from __future__ import annotations

import argparse
import json

from . import arena as arena_mod
from . import design as design_mod
from . import execute_plan as execute_plan_mod
from . import pipeline as pipeline_mod
from . import review as review_mod
from . import scatter_gather as sg_mod
from .common import DEFAULT_ROLE_RUNNERS, OrchestrationResult

_TEMPLATES = (
    "design", "pipeline", "scatter-gather", "scatter_gather", "review", "arena",
    "execute-plan", "execute_plan",
)

_COUNT_ROLES = {
    "design": {"designer": (3, 2), "researcher": (3, 1)},
    "review": {"analyst": (2, 2)},
    "arena": {"designer": (3, 2)},
    "scatter-gather": {"explorer": (3, 2)},
}
_PRIMARY_COUNT_ROLE = {
    "design": "designer",
    "review": "analyst",
    "arena": "designer",
    "scatter-gather": "explorer",
}
_ALIASES = {"scatter_gather": "scatter-gather", "execute_plan": "execute-plan"}
_RUNNERS = {"codex", "ducc"}


def run(
    template: str,
    request: str,
    *,
    role_runners: dict | None = None,
    doc_path: str | None = None,
    plan_path: str | None = None,
    n: int | None = None,
    counts: dict[str, int] | None = None,
    min_score: int = 7,
    workspace: str | None = None,
) -> OrchestrationResult:
    name = _ALIASES.get(template.strip().lower(), template.strip().lower())
    resolved_counts = _resolve_counts(name, n, counts)
    if name in {"execute-plan", "execute_plan"}:
        if not plan_path:
            raise ValueError("plan_path is required for execute_plan template")
        return execute_plan_mod.execute_plan(
            request,
            plan_path,
            role_runners=role_runners,
            workspace=workspace,
        )
    if name == "design":
        return design_mod.design(
            request,
            n_stances=resolved_counts["designer"],
            n_researchers=resolved_counts["researcher"],
            role_runners=role_runners,
            workspace=workspace,
        )
    if name == "pipeline":
        return pipeline_mod.pipeline(request, role_runners=role_runners, workspace=workspace)
    if name in {"scatter-gather", "scatter_gather"}:
        return sg_mod.scatter_gather(
            request,
            n_explorers=resolved_counts["explorer"],
            role_runners=role_runners,
            workspace=workspace,
        )
    if name == "review":
        if not doc_path:
            raise ValueError("doc_path is required for review template")
        return review_mod.review(
            request,
            doc_path,
            n_reviewers=resolved_counts["analyst"],
            min_score=min_score,
            role_runners=role_runners,
            workspace=workspace,
        )
    if name == "arena":
        return arena_mod.arena(
            request,
            n_solutions=resolved_counts["designer"],
            min_score=min_score,
            role_runners=role_runners,
            workspace=workspace,
        )
    raise ValueError(f"unknown template {template!r}; available: {', '.join(_TEMPLATES)}")


def _parse_assignments(values: list[str], option: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects ROLE=VALUE, got {value!r}")
        role, assigned = (part.strip().lower() for part in value.split("=", 1))
        if not role or not assigned:
            raise ValueError(f"{option} expects non-empty ROLE=VALUE, got {value!r}")
        if role in assignments and assignments[role] != assigned:
            raise ValueError(
                f"conflicting {option} values for role {role!r}: "
                f"{assignments[role]!r} and {assigned!r}"
            )
        assignments[role] = assigned
    return assignments


def parse_count_overrides(values: list[str] | None) -> dict[str, int]:
    raw = _parse_assignments(values or [], "--count")
    counts: dict[str, int] = {}
    for role, value in raw.items():
        try:
            count = int(value)
        except ValueError as exc:
            raise ValueError(f"--count for role {role!r} must be an integer, got {value!r}") from exc
        if count < 1:
            raise ValueError(f"--count for role {role!r} must be >= 1, got {count}")
        counts[role] = count
    return counts


def parse_role_overrides(values: list[str] | None) -> dict[str, object]:
    raw = _parse_assignments(values or [], "--role")
    overrides: dict[str, object] = {}
    for role, value in raw.items():
        if role not in DEFAULT_ROLE_RUNNERS:
            known = ", ".join(sorted(DEFAULT_ROLE_RUNNERS))
            raise ValueError(f"unknown role {role!r} for --role; known roles: {known}")
        runners = [runner.strip().lower() for runner in value.split(",")]
        if any(not runner for runner in runners):
            raise ValueError(f"--role for {role!r} contains an empty runner")
        invalid = [runner for runner in runners if runner not in _RUNNERS]
        if invalid:
            raise ValueError(
                f"unknown runner {invalid[0]!r} for role {role!r}; expected codex or ducc"
            )
        overrides[role] = runners[0] if len(runners) == 1 else runners
    return overrides


def _runner_values_equal(left: object, right: object) -> bool:
    def normalized(value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)  # type: ignore[arg-type]
    return normalized(left) == normalized(right)


def _role_runners_from_args(args: argparse.Namespace) -> dict[str, object]:
    role_runners = parse_role_overrides(args.role)
    for role in ("designer", "reviewer", "implementer"):
        runner = getattr(args, role)
        if runner:
            if role in role_runners and not _runner_values_equal(role_runners[role], runner):
                raise ValueError(
                    f"conflicting runner overrides for role {role!r}: "
                    f"--role specifies {role_runners[role]!r}, --{role} specifies {runner!r}"
                )
            role_runners[role] = runner
    return role_runners


def _resolve_counts(template: str, n: int | None, counts: dict[str, int] | None) -> dict[str, int]:
    supported = _COUNT_ROLES.get(template, {})
    supplied = dict(counts or {})
    unknown = sorted(set(supplied) - set(supported))
    if unknown:
        if supported:
            raise ValueError(
                f"template {template!r} does not support --count for role {unknown[0]!r}; "
                f"supported roles: {', '.join(supported)}"
            )
        raise ValueError(f"template {template!r} does not support --count")
    if n is not None:
        primary = _PRIMARY_COUNT_ROLE.get(template)
        if primary is None:
            raise ValueError(f"template {template!r} does not support --n")
        if primary in supplied and supplied[primary] != n:
            raise ValueError(
                f"conflicting counts for role {primary!r}: --n={n} and --count {primary}={supplied[primary]}"
            )
        supplied[primary] = n
    resolved: dict[str, int] = {}
    for role, (default, minimum) in supported.items():
        value = supplied.get(role, default)
        if value < minimum:
            raise ValueError(
                f"--count for role {role!r} in template {template!r} must be >= {minimum}, got {value}"
            )
        resolved[role] = value
    return resolved


def _print_result(result: OrchestrationResult) -> None:
    payload = {
        "succeeded": result.succeeded,
        "artifacts": result.artifacts,
        "final_path": result.final_path,
        "atom_runner_plan": result.atom_runner_plan,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> bool:
    parser = argparse.ArgumentParser(description="Run an atomic-agents orchestration template.")
    parser.add_argument("template", help=f"Template name: {', '.join(_TEMPLATES)}")
    parser.add_argument("request", help="Natural-language request for the orchestration.")
    parser.add_argument("--doc", dest="doc_path", help="Document path for the review template.")
    parser.add_argument("--plan", dest="plan_path", help="Plan document path for the execute_plan template.")
    parser.add_argument("--n", type=int, help="Parallel count for templates that support it.")
    parser.add_argument("--count", action="append", default=[], metavar="ROLE=N",
                        help="Override a repeatable role count; may be repeated.")
    parser.add_argument("--role", action="append", default=[], metavar="ROLE=RUNNER[,RUNNER]",
                        help="Override or round-robin a role's codex/ducc runner; may be repeated.")
    parser.add_argument("--min-score", type=int, default=7, help="Minimum score for judge gates.")
    parser.add_argument("--designer", choices=("codex", "ducc"), help="Override designer runner.")
    parser.add_argument("--reviewer", choices=("codex", "ducc"), help="Override reviewer runner.")
    parser.add_argument("--implementer", choices=("codex", "ducc"), help="Override implementer runner.")
    parser.add_argument("--workspace", help="Workspace directory to use.")
    args = parser.parse_args()

    result = run(
        args.template,
        args.request,
        role_runners=_role_runners_from_args(args),
        doc_path=args.doc_path,
        plan_path=args.plan_path,
        n=args.n,
        counts=parse_count_overrides(args.count),
        min_score=args.min_score,
        workspace=args.workspace,
    )
    _print_result(result)
    return result.succeeded


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)


__all__ = ["run", "main", "parse_count_overrides", "parse_role_overrides"]
