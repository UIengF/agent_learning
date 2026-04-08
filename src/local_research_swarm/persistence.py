from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from local_research_swarm.config import AppConfig
from local_research_swarm.models import (
    AgentAction,
    AgentRole,
    ArtifactRecord,
    CheckpointRecord,
    RunRecord,
    RunStatus,
    TaskRecord,
    TaskStatus,
    TraceEvent,
    new_id,
    utc_now,
)


class SQLiteRunStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.config.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    runtime_mode TEXT NOT NULL DEFAULT 'swarm',
                    status TEXT NOT NULL,
                    current_agent TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    max_rounds INTEGER NOT NULL,
                    max_parallel_agents INTEGER NOT NULL,
                    budget_tokens INTEGER NOT NULL,
                    budget_cost_usd REAL NOT NULL,
                    spent_tokens INTEGER NOT NULL,
                    spent_cost_usd REAL NOT NULL,
                    auto_execute INTEGER NOT NULL,
                    result_artifact_id TEXT NOT NULL,
                    result_path TEXT NOT NULL,
                    last_error TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    last_error TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    citation_label TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trace_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent_role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    message TEXT NOT NULL,
                    token_usage INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    duration_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_path TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
            if "runtime_mode" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN runtime_mode TEXT NOT NULL DEFAULT 'swarm'")

    def create_run(
        self,
        *,
        goal: str,
        runtime_mode: str = "swarm",
        max_rounds: int,
        max_parallel_agents: int,
        budget_tokens: int,
        budget_cost_usd: float,
        auto_execute: bool,
    ) -> RunRecord:
        record = RunRecord(
            id=new_id("run"),
            goal=goal,
            runtime_mode=runtime_mode,
            status=RunStatus.QUEUED,
            current_agent="",
            created_at=utc_now(),
            updated_at=utc_now(),
            round_index=0,
            max_rounds=max_rounds,
            max_parallel_agents=max_parallel_agents,
            budget_tokens=budget_tokens,
            budget_cost_usd=budget_cost_usd,
            auto_execute=auto_execute,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, goal, runtime_mode, status, current_agent, created_at, updated_at, round_index,
                    max_rounds, max_parallel_agents, budget_tokens, budget_cost_usd,
                    spent_tokens, spent_cost_usd, auto_execute, result_artifact_id, result_path, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.goal,
                    record.runtime_mode,
                    record.status.value,
                    record.current_agent,
                    record.created_at,
                    record.updated_at,
                    record.round_index,
                    record.max_rounds,
                    record.max_parallel_agents,
                    record.budget_tokens,
                    record.budget_cost_usd,
                    record.spent_tokens,
                    record.spent_cost_usd,
                    1 if record.auto_execute else 0,
                    record.result_artifact_id,
                    record.result_path,
                    record.last_error,
                ),
            )
        return record

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run id: {run_id}")
        return self._row_to_run(row)

    def update_run(self, run: RunRecord) -> RunRecord:
        run.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET runtime_mode = ?, status = ?, current_agent = ?, updated_at = ?, round_index = ?,
                    max_rounds = ?, max_parallel_agents = ?, budget_tokens = ?, budget_cost_usd = ?,
                    spent_tokens = ?, spent_cost_usd = ?, auto_execute = ?, result_artifact_id = ?,
                    result_path = ?, last_error = ?
                WHERE id = ?
                """,
                (
                    run.runtime_mode,
                    run.status.value,
                    run.current_agent,
                    run.updated_at,
                    run.round_index,
                    run.max_rounds,
                    run.max_parallel_agents,
                    run.budget_tokens,
                    run.budget_cost_usd,
                    run.spent_tokens,
                    run.spent_cost_usd,
                    1 if run.auto_execute else 0,
                    run.result_artifact_id,
                    run.result_path,
                    run.last_error,
                    run.id,
                ),
            )
        return run

    def set_run_status(self, run_id: str, status: RunStatus, *, current_agent: str = "", last_error: str = "") -> RunRecord:
        run = self.get_run(run_id)
        run.status = status
        run.current_agent = current_agent
        run.last_error = last_error
        return self.update_run(run)

    def increment_usage(self, run_id: str, token_usage: int, cost_usd: float) -> RunRecord:
        run = self.get_run(run_id)
        run.spent_tokens += int(token_usage)
        run.spent_cost_usd += float(cost_usd)
        return self.update_run(run)

    def upsert_task(self, task: TaskRecord) -> TaskRecord:
        task.updated_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (
                    id, run_id, role, title, description, status, round_index, payload_json,
                    created_at, updated_at, started_at, completed_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    role = excluded.role,
                    title = excluded.title,
                    description = excluded.description,
                    status = excluded.status,
                    round_index = excluded.round_index,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at,
                    last_error = excluded.last_error
                """,
                (
                    task.id,
                    task.run_id,
                    task.role.value,
                    task.title,
                    task.description,
                    task.status.value,
                    task.round_index,
                    json.dumps(task.payload, ensure_ascii=False),
                    task.created_at,
                    task.updated_at,
                    task.started_at,
                    task.completed_at,
                    task.last_error,
                ),
            )
        return task

    def list_tasks(self, run_id: str, status: TaskStatus | None = None) -> list[TaskRecord]:
        query = "SELECT * FROM tasks WHERE run_id = ?"
        params: tuple[object, ...] = (run_id,)
        if status is not None:
            query += " AND status = ?"
            params += (status.value,)
        query += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_task(row) for row in rows]

    def get_task(self, task_id: str) -> TaskRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown task id: {task_id}")
        return self._row_to_task(row)

    def mark_task_status(self, task_id: str, status: TaskStatus, *, last_error: str = "") -> TaskRecord:
        task = self.get_task(task_id)
        task.status = status
        task.last_error = last_error
        if status is TaskStatus.IN_PROGRESS:
            task.started_at = utc_now()
        if status in {TaskStatus.DONE, TaskStatus.DROPPED, TaskStatus.BLOCKED}:
            task.completed_at = utc_now()
        return self.upsert_task(task)

    def reset_in_progress_tasks(self, run_id: str) -> None:
        for task in self.list_tasks(run_id):
            if task.status is TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.PENDING
                task.last_error = ""
                self.upsert_task(task)

    def save_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        artifact.file_path = self._write_artifact_file(artifact)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts (
                    id, run_id, task_id, kind, title, content, source_url, citation_label,
                    citations_json, metadata_json, created_at, file_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    artifact.run_id,
                    artifact.task_id,
                    artifact.kind,
                    artifact.title,
                    artifact.content,
                    artifact.source_url,
                    artifact.citation_label,
                    json.dumps(artifact.citations, ensure_ascii=False),
                    json.dumps(artifact.metadata, ensure_ascii=False),
                    artifact.created_at,
                    artifact.file_path,
                ),
            )
        if artifact.kind == "report":
            run = self.get_run(artifact.run_id)
            run.result_artifact_id = artifact.id
            run.result_path = artifact.file_path
            self.update_run(run)
        return artifact

    def list_artifacts(self, run_id: str, kind: str | None = None) -> list[ArtifactRecord]:
        query = "SELECT * FROM artifacts WHERE run_id = ?"
        params: tuple[object, ...] = (run_id,)
        if kind is not None:
            query += " AND kind = ?"
            params += (kind,)
        query += " ORDER BY created_at ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_artifact(row) for row in rows]

    def get_result_artifact(self, run_id: str) -> ArtifactRecord:
        run = self.get_run(run_id)
        if run.result_artifact_id:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (run.result_artifact_id,)).fetchone()
            if row is not None:
                return self._row_to_artifact(row)
        reports = self.list_artifacts(run_id, kind="report")
        if not reports:
            raise KeyError(f"No report artifact for run {run_id}")
        return reports[-1]

    def append_trace(self, event: TraceEvent) -> TraceEvent:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trace_events (
                    id, run_id, task_id, agent_role, action, message, token_usage, cost_usd, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.run_id,
                    event.task_id,
                    event.agent_role.value,
                    event.action.value,
                    event.message,
                    event.token_usage,
                    event.cost_usd,
                    event.duration_ms,
                    event.created_at,
                ),
            )
        return event

    def list_trace(self, run_id: str) -> list[TraceEvent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trace_events WHERE run_id = ? ORDER BY created_at ASC", (run_id,)).fetchall()
        return [self._row_to_trace(row) for row in rows]

    def save_checkpoint(self, run_id: str, *, phase: str, payload: dict) -> CheckpointRecord:
        record = CheckpointRecord(id=new_id("checkpoint"), run_id=run_id, phase=phase, payload=payload)
        record.file_path = self._write_checkpoint_file(record)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints (id, run_id, phase, payload_json, created_at, file_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.run_id,
                    record.phase,
                    json.dumps(record.payload, ensure_ascii=False),
                    record.created_at,
                    record.file_path,
                ),
            )
        return record

    def latest_checkpoint(self, run_id: str) -> CheckpointRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"No checkpoints for run {run_id}")
        return self._row_to_checkpoint(row)

    def _run_dir(self, run_id: str) -> Path:
        path = self.config.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _artifact_dir(self, run_id: str) -> Path:
        path = self.config.artifacts_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_artifact_file(self, artifact: ArtifactRecord) -> str:
        suffix = ".md" if artifact.kind == "report" else ".txt"
        path = self._artifact_dir(artifact.run_id) / f"{artifact.id}-{artifact.kind}{suffix}"
        path.write_text(artifact.content, encoding="utf-8")
        return str(path)

    def _write_checkpoint_file(self, checkpoint: CheckpointRecord) -> str:
        checkpoint_dir = self._run_dir(checkpoint.run_id) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{checkpoint.created_at.replace(':', '-')}-{checkpoint.phase}.json"
        path.write_text(json.dumps(checkpoint.payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            goal=row["goal"],
            runtime_mode=row["runtime_mode"],
            status=RunStatus(row["status"]),
            current_agent=row["current_agent"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            round_index=int(row["round_index"]),
            max_rounds=int(row["max_rounds"]),
            max_parallel_agents=int(row["max_parallel_agents"]),
            budget_tokens=int(row["budget_tokens"]),
            budget_cost_usd=float(row["budget_cost_usd"]),
            spent_tokens=int(row["spent_tokens"]),
            spent_cost_usd=float(row["spent_cost_usd"]),
            auto_execute=bool(row["auto_execute"]),
            result_artifact_id=row["result_artifact_id"],
            result_path=row["result_path"],
            last_error=row["last_error"],
        )

    def _row_to_task(self, row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            id=row["id"],
            run_id=row["run_id"],
            role=AgentRole(row["role"]),
            title=row["title"],
            description=row["description"],
            status=TaskStatus(row["status"]),
            round_index=int(row["round_index"]),
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            last_error=row["last_error"],
        )

    def _row_to_artifact(self, row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            id=row["id"],
            run_id=row["run_id"],
            task_id=row["task_id"],
            kind=row["kind"],
            title=row["title"],
            content=row["content"],
            source_url=row["source_url"],
            citation_label=row["citation_label"],
            citations=json.loads(row["citations_json"]),
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            file_path=row["file_path"],
        )

    def _row_to_trace(self, row: sqlite3.Row) -> TraceEvent:
        return TraceEvent(
            id=row["id"],
            run_id=row["run_id"],
            task_id=row["task_id"],
            agent_role=AgentRole(row["agent_role"]),
            action=AgentAction(row["action"]),
            message=row["message"],
            token_usage=int(row["token_usage"]),
            cost_usd=float(row["cost_usd"]),
            duration_ms=int(row["duration_ms"]),
            created_at=row["created_at"],
        )

    def _row_to_checkpoint(self, row: sqlite3.Row) -> CheckpointRecord:
        return CheckpointRecord(
            id=row["id"],
            run_id=row["run_id"],
            phase=row["phase"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            file_path=row["file_path"],
        )
