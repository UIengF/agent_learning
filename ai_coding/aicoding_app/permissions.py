from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex


class PermissionDenied(ValueError):
    pass


class CommandDenied(PermissionDenied):
    pass


_BLOCKED_PREFIXES = (
    "rm",
    "del",
    "erase",
    "rmdir",
    "remove-item",
    "git reset --hard",
    "git clean",
    "git checkout --",
    "shutdown",
    "format",
)


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().lower().split())


def _matches_prefix(command: str, prefix: str) -> bool:
    normalized = _normalize_command(command)
    normalized_prefix = _normalize_command(prefix)
    return normalized == normalized_prefix or normalized.startswith(f"{normalized_prefix} ")


def _has_shell_control_operator(command: str) -> bool:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            index += 1
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            index += 1
            continue
        if not in_single_quote and not in_double_quote:
            if char in {";", "|"}:
                return True
            if command[index : index + 2] in {"&&", "||"}:
                return True
        index += 1
    return False


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=False)
    except ValueError as exc:
        raise CommandDenied(f"could not parse command: {exc}") from exc


@dataclass(frozen=True)
class WorkspacePolicy:
    workspace: Path
    allowed_commands: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", self.workspace.resolve())

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        if resolved == self.workspace or self.workspace in resolved.parents:
            return resolved
        raise PermissionDenied(f"path is outside workspace: {resolved}")

    def relative_path(self, path: str | Path) -> str:
        return self.resolve_path(path).relative_to(self.workspace).as_posix()

    def validate_command(self, command: str) -> str:
        stripped = command.strip()
        if not stripped:
            raise CommandDenied("empty command is not allowed")
        if _has_shell_control_operator(stripped):
            raise CommandDenied("shell control operators are not allowed")
        normalized = _normalize_command(stripped)
        for blocked in _BLOCKED_PREFIXES:
            if _matches_prefix(normalized, blocked):
                raise CommandDenied(f"destructive command is not allowed: {blocked}")
        if any(_matches_prefix(stripped, allowed) for allowed in self.allowed_commands):
            return stripped
        if self._is_workspace_python_script_command(stripped):
            return stripped
        allowed_text = "; ".join(self.allowed_commands)
        raise CommandDenied(
            f"command is outside whitelist: {stripped}. "
            f"Allowed: {allowed_text}; python <workspace-relative .py> [args]"
        )

    def _is_workspace_python_script_command(self, command: str) -> bool:
        parts = _split_command(command)
        if len(parts) < 2 or parts[0].lower() not in {"python", "python3"}:
            return False
        script_token = parts[1].strip("\"'")
        if script_token.startswith("-"):
            return False
        script = Path(script_token)
        if script.is_absolute() or script.suffix.lower() != ".py":
            return False
        if script.name.lower() == "setup.py" and any(
            part.strip("\"'").lower() in {"install", "develop", "bdist_wheel", "sdist"}
            for part in parts[2:]
        ):
            return False
        try:
            resolved = self.resolve_path(script)
        except PermissionDenied:
            return False
        return resolved.is_file()
