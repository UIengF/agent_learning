"""atomic-agents — 原子化多 agent 协作运行时。

独立项目，运行时不依赖 ~/.claude/skills/teammate。
"""

from atomic_agents.run import RunResult, run_meta_plan, run_skeleton

__version__ = "0.1.0"

__all__ = ["RunResult", "__version__", "run_meta_plan", "run_skeleton"]
