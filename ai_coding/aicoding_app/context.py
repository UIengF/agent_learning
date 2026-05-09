from __future__ import annotations

from .evidence_cache import compact_text
from .plan import CodingPlan, format_plan


def compress_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_context(
    *,
    history: list[dict[str, str]],
    plan: CodingPlan,
    evidence_context: str,
    latest_diff: str,
    max_chars: int,
    project_instructions: str = "",
    repo_map: str = "",
    documentation_summary: str = "",
    memory_summary: str = "",
) -> str:
    recent_history = history[-6:]
    history_text = "\n".join(
        f"{item.get('role', 'unknown')}: {compact_text(item.get('content', ''), 600)}"
        for item in recent_history
    )
    blocks = [
        "Project instructions:",
        project_instructions or "No project instructions found.",
        "Repository map:",
        repo_map or "No repository map captured yet.",
        "Documentation files:",
        documentation_summary or "No documentation summary captured yet.",
        "Memory summary:",
        memory_summary or "Memory: none",
        "Current coding plan:",
        format_plan(plan),
        "Evidence cache:",
        evidence_context or "No cached evidence.",
        "Latest diff:",
        compress_text(latest_diff, 3000) if latest_diff else "No diff captured yet.",
        "Recent conversation:",
        history_text or "No previous turns.",
    ]
    return compress_text("\n\n".join(blocks), max_chars)
