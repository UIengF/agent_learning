from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    data_dir: Path
    runs_dir: Path
    artifacts_dir: Path
    cache_dir: Path
    db_path: Path
    profile: str
    api_key: str
    base_url: str
    model: str
    default_max_rounds: int = 2
    default_max_parallel_agents: int = 3
    default_budget_tokens: int = 4000
    default_budget_cost_usd: float = 5.0
    default_search_limit: int = 2

    @classmethod
    def default(cls, project_root: Path | None = None) -> "AppConfig":
        root = Path(os.getenv("RESEARCH_SWARM_PROJECT_ROOT", project_root or Path.cwd())).resolve()
        data_dir = root / "results"
        profile = os.getenv("RESEARCH_SWARM_PROFILE", "auto").strip().lower() or "auto"
        api_key = os.getenv("RESEARCH_SWARM_API_KEY", "").strip()
        if profile == "auto":
            profile = "cloud" if api_key else "deterministic"
        return cls(
            project_root=root,
            data_dir=data_dir,
            runs_dir=data_dir / "runs",
            artifacts_dir=data_dir / "artifacts",
            cache_dir=data_dir / "cache",
            db_path=data_dir / "runs" / "swarm.sqlite3",
            profile=profile,
            api_key=api_key,
            base_url=os.getenv("RESEARCH_SWARM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.getenv("RESEARCH_SWARM_MODEL", "gpt-4o-mini"),
        )

    def ensure_directories(self) -> None:
        self.project_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
