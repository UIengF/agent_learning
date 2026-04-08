from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import typer

from local_research_swarm.config import AppConfig
from local_research_swarm.runtime import SwarmService


app = typer.Typer(no_args_is_help=True)


def _config() -> AppConfig:
    config = AppConfig.default(project_root=Path.cwd())
    config.ensure_directories()
    return config


def _service() -> SwarmService:
    return SwarmService.from_config(_config())


def _jsonify(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonify(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


@app.command("doctor")
def doctor() -> None:
    config = _config()
    payload = {
        "project_root": str(config.project_root),
        "profile": config.profile,
        "db_path": str(config.db_path),
    }
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command("submit")
def submit(
    goal: str,
    mode: str = typer.Option("swarm", "--mode"),
    max_rounds: int = typer.Option(2, min=0),
    max_parallel_agents: int = typer.Option(3, min=1),
    budget_tokens: int = typer.Option(4000, min=1),
    budget_cost_usd: float = typer.Option(5.0, min=0.0),
    defer: bool = typer.Option(False, "--defer"),
) -> None:
    run = _service().submit(
        goal,
        mode=mode,
        max_rounds=max_rounds,
        max_parallel_agents=max_parallel_agents,
        budget_tokens=budget_tokens,
        budget_cost_usd=budget_cost_usd,
        defer=defer,
    )
    typer.echo(
        json.dumps(
            {"run_id": run.id, "status": run.status.value, "runtime_mode": run.runtime_mode, "result_path": run.result_path},
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("status")
def status(run_id: str) -> None:
    run = _service().status(run_id)
    typer.echo(json.dumps(_jsonify(run), indent=2, ensure_ascii=False))


@app.command("trace")
def trace(run_id: str) -> None:
    events = _service().trace(run_id)
    typer.echo(json.dumps(_jsonify(events), indent=2, ensure_ascii=False))


@app.command("result")
def result(run_id: str) -> None:
    typer.echo(_service().result(run_id))


@app.command("resume")
def resume(run_id: str) -> None:
    run = _service().resume(run_id)
    typer.echo(
        json.dumps(
            {"run_id": run.id, "status": run.status.value, "runtime_mode": run.runtime_mode, "result_path": run.result_path},
            indent=2,
            ensure_ascii=False,
        )
    )


@app.command("abort")
def abort(run_id: str) -> None:
    run = _service().abort(run_id)
    typer.echo(
        json.dumps(
            {"run_id": run.id, "status": run.status.value, "runtime_mode": run.runtime_mode},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    app()
