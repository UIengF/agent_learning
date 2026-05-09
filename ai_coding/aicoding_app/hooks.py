from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .permissions import WorkspacePolicy


@dataclass(frozen=True)
class HookPreview:
    name: str
    commands: tuple[str, ...]
    allowed: tuple[str, ...]
    denied: tuple[str, ...]

    def format(self) -> str:
        if not self.commands:
            return f"Hook {self.name}: not configured."
        lines = [f"Hook {self.name}: preview only; no hook commands were executed."]
        if self.allowed:
            lines.extend(["Allowed by policy:", *[f"- {item}" for item in self.allowed]])
        if self.denied:
            lines.extend(["Denied by policy:", *[f"- {item}" for item in self.denied]])
        return "\n".join(lines)


def load_hook_commands(workspace: str | Path, hook_name: str) -> tuple[str, ...]:
    config_path = Path(workspace).resolve() / ".aicoding" / "config.toml"
    if not config_path.exists():
        return ()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    hooks = data.get("hooks", {})
    raw = hooks.get(hook_name)
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(item for item in raw if isinstance(item, str))
    return ()


def preview_hook(policy: WorkspacePolicy, hook_name: str) -> HookPreview:
    commands = load_hook_commands(policy.workspace, hook_name)
    allowed: list[str] = []
    denied: list[str] = []
    for command in commands:
        try:
            allowed.append(policy.validate_command(command))
        except Exception as exc:
            denied.append(f"{command} ({exc})")
    return HookPreview(
        name=hook_name,
        commands=commands,
        allowed=tuple(allowed),
        denied=tuple(denied),
    )
