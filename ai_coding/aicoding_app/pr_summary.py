from __future__ import annotations

from .evidence_cache import compact_text


def build_commit_preview(changed_files: list[str], diff: str) -> str:
    scope = ", ".join(changed_files[:3]) if changed_files else "workspace"
    if len(changed_files) > 3:
        scope += f", +{len(changed_files) - 3} more"
    summary = "Update " + scope
    return "\n".join(
        [
            "Commit preview:",
            f"message: {summary}",
            "changed files:",
            "\n".join(f"- {item}" for item in changed_files) or "- none",
            "diff summary:",
            compact_text(diff, 1000) or "No diff.",
        ]
    )


def build_pr_summary(
    *,
    change_summary: str,
    changed_files: list[str],
    validation_result: str,
    risks: list[str],
) -> str:
    return "\n".join(
        [
            "PR-ready summary:",
            "",
            "Summary:",
            f"- {change_summary or 'No change summary provided.'}",
            "",
            "Changed files:",
            "\n".join(f"- {item}" for item in changed_files) or "- none",
            "",
            "Validation:",
            compact_text(validation_result, 1200) or "- Not run.",
            "",
            "Risks:",
            "\n".join(f"- {item}" for item in risks) or "- none",
        ]
    )
