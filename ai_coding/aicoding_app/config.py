from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ALLOWED_COMMANDS = (
    "pytest",
    "python -m pytest",
    "ruff check",
    "python -m ruff check",
    "pyright",
    "python -m pyright",
    "git status",
    "git diff",
    "git branch",
    "rg",
)


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _as_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, *, default: int, minimum: int) -> int:
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def _split_semicolon(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    items = tuple(item.strip() for item in value.split(";") if item.strip())
    return items or default


def _strip_optional_angle_brackets(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.startswith("<") and stripped.endswith(">") and len(stripped) > 2:
        return stripped[1:-1].strip()
    return stripped


@dataclass(frozen=True)
class ModelConfig:
    api_key: str | None
    api_base: str
    model_name: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model_name)

    def masked_api_key(self) -> str:
        if not self.api_key:
            return "<missing>"
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"


@dataclass(frozen=True)
class HarnessConfig:
    allowed_commands: tuple[str, ...]
    runtime_dir: Path
    max_context_chars: int
    trace_enabled: bool
    command_timeout_seconds: int
    max_tool_rounds: int


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    model: ModelConfig
    harness: HarnessConfig

    def public_dict(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "model": {
                "api_key": self.model.masked_api_key(),
                "api_base": self.model.api_base,
                "model_name": self.model.model_name,
                "configured": self.model.configured,
            },
            "harness": {
                "allowed_commands": list(self.harness.allowed_commands),
                "runtime_dir": str(self.harness.runtime_dir),
                "max_context_chars": self.harness.max_context_chars,
                "trace_enabled": self.harness.trace_enabled,
                "command_timeout_seconds": self.harness.command_timeout_seconds,
                "max_tool_rounds": self.harness.max_tool_rounds,
            },
        }


def build_app_config(
    *,
    env_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> AppConfig:
    file_values = _read_env_file(Path(env_file or project_root / ".env"))
    merged = dict(file_values)
    merged.update(dict(os.environ if env is None else env))

    runtime_raw = merged.get("AICODING_RUNTIME_DIR", "runtime")
    runtime_dir = Path(runtime_raw)
    if not runtime_dir.is_absolute():
        runtime_dir = project_root / runtime_dir

    return AppConfig(
        project_root=project_root,
        model=ModelConfig(
            api_key=_strip_optional_angle_brackets(
                merged.get("AICODING_MODEL_API_KEY")
                or merged.get("AICODING_API_KEY")
                or merged.get("DEEPSEEK_API_KEY")
                or merged.get("OPENAI_API_KEY")
            ),
            api_base=_strip_optional_angle_brackets(
                merged.get("AICODING_MODEL_API_BASE")
                or merged.get("AICODING_BASE_URL")
                or merged.get("DEEPSEEK_BASE_URL")
                or merged.get("OPENAI_BASE_URL")
            )
            or "https://api.deepseek.com",
            model_name=_strip_optional_angle_brackets(
                merged.get("AICODING_MODEL_NAME")
                or merged.get("AICODING_MODEL")
                or merged.get("DEEPSEEK_MODEL")
                or merged.get("OPENAI_MODEL")
            )
            or "deepseek-chat",
        ),
        harness=HarnessConfig(
            allowed_commands=_split_semicolon(
                merged.get("AICODING_ALLOWED_COMMANDS"), DEFAULT_ALLOWED_COMMANDS
            ),
            runtime_dir=runtime_dir,
            max_context_chars=_as_int(
                merged.get("AICODING_MAX_CONTEXT_CHARS"), default=16000, minimum=2000
            ),
            trace_enabled=_as_bool(merged.get("AICODING_TRACE_ENABLED"), default=True),
            command_timeout_seconds=_as_int(
                merged.get("AICODING_COMMAND_TIMEOUT_SECONDS"), default=60, minimum=5
            ),
            max_tool_rounds=_as_int(
                merged.get("AICODING_MAX_TOOL_ROUNDS"), default=60, minimum=5
            ),
        ),
    )
