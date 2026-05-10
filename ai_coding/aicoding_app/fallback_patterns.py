from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import re
from re import Match, Pattern

from .permissions import WorkspacePolicy


@dataclass(frozen=True)
class FallbackPatternContext:
    policy: WorkspacePolicy

    @staticmethod
    def added_lines(content: str) -> str:
        return "\n".join(f"+{line}" for line in content.split("\\n"))

    @staticmethod
    def removed_lines(content: str) -> str:
        return "\n".join(f"-{line}" for line in content.splitlines())


PatchBuilder = Callable[[Match[str], FallbackPatternContext], str | None]


@dataclass(frozen=True)
class FallbackPattern:
    name: str
    expression: str
    builder: PatchBuilder
    flags: int = re.IGNORECASE | re.DOTALL

    def compile(self) -> Pattern[str]:
        return re.compile(self.expression, self.flags)


class FallbackPatternRegistry:
    def __init__(self, patterns: Iterable[FallbackPattern] = ()) -> None:
        self._patterns = list(patterns)

    @classmethod
    def defaults(cls) -> FallbackPatternRegistry:
        return cls(DEFAULT_FALLBACK_PATTERNS)

    def register(self, pattern: FallbackPattern) -> None:
        self._patterns.append(pattern)

    def build_patch(self, task: str, *, policy: WorkspacePolicy) -> str | None:
        context = FallbackPatternContext(policy=policy)
        for pattern in self._patterns:
            match = pattern.compile().search(task)
            if not match:
                continue
            patch = pattern.builder(match, context)
            if patch:
                return patch
        return None

    def names(self) -> list[str]:
        return [pattern.name for pattern in self._patterns]


def _build_create_patch(match: Match[str], context: FallbackPatternContext) -> str:
    path, content = match.groups()
    return (
        f"*** Begin Patch\n"
        f"*** Add File: {path}\n"
        f"{context.added_lines(content)}\n"
        f"*** End Patch"
    )


def _build_set_patch(match: Match[str], context: FallbackPatternContext) -> str:
    path, content = match.groups()
    target = context.policy.resolve_path(path)
    if target.exists():
        current = target.read_text(encoding="utf-8")
        return (
            f"*** Begin Patch\n"
            f"*** Update File: {path}\n"
            f"@@\n"
            f"{context.removed_lines(current)}\n"
            f"{context.added_lines(content)}\n"
            f"*** End Patch"
        )
    return (
        f"*** Begin Patch\n"
        f"*** Add File: {path}\n"
        f"{context.added_lines(content)}\n"
        f"*** End Patch"
    )


def _build_append_patch(match: Match[str], _context: FallbackPatternContext) -> str:
    text, path = match.groups()
    return f"*** Begin Patch\n*** Update File: {path}\n@@\n+{text}\n*** End Patch"


def _build_replace_patch(match: Match[str], _context: FallbackPatternContext) -> str:
    old, new, path = match.groups()
    return f"*** Begin Patch\n*** Update File: {path}\n@@\n-{old}\n+{new}\n*** End Patch"


DEFAULT_FALLBACK_PATTERNS: tuple[FallbackPattern, ...] = (
    FallbackPattern(
        name="create-file-with-content",
        expression=r"create\s+['\"]?([A-Za-z0-9_./\\-]+)['\"]?\s+with\s+['\"](.+?)['\"]",
        builder=_build_create_patch,
    ),
    FallbackPattern(
        name="set-file-to-content",
        expression=r"set\s+['\"]?([A-Za-z0-9_./\\-]+)['\"]?\s+to\s+['\"](.+?)['\"]",
        builder=_build_set_patch,
    ),
    FallbackPattern(
        name="append-text-to-file",
        expression=r"append\s+['\"](.+?)['\"]\s+to\s+([A-Za-z0-9_./\\-]+)",
        builder=_build_append_patch,
    ),
    FallbackPattern(
        name="replace-text-in-file",
        expression=r"replace\s+['\"](.+?)['\"]\s+with\s+['\"](.+?)['\"]\s+in\s+([A-Za-z0-9_./\\-]+)",
        builder=_build_replace_patch,
    ),
)
