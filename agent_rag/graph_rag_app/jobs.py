from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import traceback
from typing import Any, Callable
from uuid import uuid4

from .agent_evaluation import run_evaluation_dataset
from .config import AppConfig, IndexBuildConfig, build_app_config
from .eval_datasets import load_evaluation_dataset
from .eval_reporting import load_evaluation_report, save_evaluation_report
from .indexing import build_index


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    kind: str
    status: str
    created_at: str
    updated_at: str
    log_path: str
    result: dict[str, Any] | None = None
    error: str = ""


class JobNotFound(KeyError):
    pass


class BackgroundJobManager:
    def __init__(self, runtime_dir: str | Path = "runtime/jobs", max_workers: int = 2):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _set_record(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = record
        (self.runtime_dir / f"{record.job_id}.json").write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _update(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        record = self.get(job_id)
        self._set_record(
            JobRecord(
                job_id=record.job_id,
                kind=record.kind,
                status=status,
                created_at=record.created_at,
                updated_at=self._now(),
                log_path=record.log_path,
                result=result,
                error=error,
            )
        )

    def submit(self, kind: str, target: Callable[[Path], dict[str, Any]]) -> JobRecord:
        job_id = uuid4().hex
        log_path = self.runtime_dir / f"{job_id}.log"
        record = JobRecord(
            job_id=job_id,
            kind=kind,
            status="queued",
            created_at=self._now(),
            updated_at=self._now(),
            log_path=str(log_path),
        )
        self._set_record(record)

        def run() -> None:
            self._update(job_id, status="running")
            try:
                log_path.write_text(
                    f"job_id={job_id}\nkind={kind}\nstatus=running\n",
                    encoding="utf-8",
                )
                result = target(log_path)
            except Exception as exc:
                details = "".join(traceback.format_exception(exc))
                with log_path.open("a", encoding="utf-8") as file:
                    file.write("\nstatus=failed\n")
                    file.write(details)
                self._update(job_id, status="failed", error=str(exc))
                return

            with log_path.open("a", encoding="utf-8") as file:
                file.write("\nstatus=completed\n")
                file.write(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                file.write("\n")
            self._update(job_id, status="completed", result=result)

        self._executor.submit(run)
        return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
        if record is not None:
            return record

        record_path = self.runtime_dir / f"{job_id}.json"
        if not record_path.exists():
            raise JobNotFound(job_id)
        data = json.loads(record_path.read_text(encoding="utf-8"))
        record = JobRecord(
            job_id=str(data["job_id"]),
            kind=str(data["kind"]),
            status=str(data["status"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            log_path=str(data["log_path"]),
            result=data.get("result") if isinstance(data.get("result"), dict) else None,
            error=str(data.get("error", "")),
        )
        with self._lock:
            self._jobs[job_id] = record
        return record

    def read_log(self, job_id: str, *, max_chars: int = 12000) -> str:
        record = self.get(job_id)
        path = Path(record.log_path)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]


def submit_index_build_job(
    manager: BackgroundJobManager,
    *,
    kb_path: str,
    output_dir: str,
    config: IndexBuildConfig | None = None,
) -> JobRecord:
    index_config = config or IndexBuildConfig()

    def target(log_path: Path) -> dict[str, Any]:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"kb_path={kb_path}\noutput_dir={output_dir}\n")
        report = build_index(kb_path=kb_path, output_dir=output_dir, config=index_config)
        return {
            "index_dir": report.index_dir,
            "chunk_count": report.chunk_count,
            "document_count": report.document_count,
            "built_at": report.built_at,
            "index_db_path": report.index_db_path,
            "manifest_path": report.manifest_path,
        }

    return manager.submit("index_build", target)


def submit_eval_run_job(
    manager: BackgroundJobManager,
    *,
    dataset: str,
    index_dir: str,
    output_dir: str,
    app_config: AppConfig | None = None,
    baseline_run: str | None = None,
    tags: list[str] | None = None,
) -> JobRecord:
    def target(log_path: Path) -> dict[str, Any]:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"dataset={dataset}\nindex_dir={index_dir}\n")
        include_tags = set(tags) if tags else None
        loaded_dataset = load_evaluation_dataset(Path(dataset), include_tags=include_tags)
        config = app_config or build_app_config(index_dir)
        report = run_evaluation_dataset(loaded_dataset, index_dir=index_dir, app_config=config)
        baseline_report = load_evaluation_report(Path(baseline_run)) if baseline_run else None
        artifact = save_evaluation_report(
            report,
            output_root=Path(output_dir),
            baseline_report=baseline_report,
        )
        return {
            "run_dir": str(artifact.run_dir),
            "summary_path": str(artifact.summary_path),
            "results_path": str(artifact.results_path),
            "dataset_name": report.dataset_name,
            "case_count": report.case_count,
            "pass_rate": report.pass_rate,
        }

    return manager.submit("eval_run", target)


def job_to_dict(record: JobRecord) -> dict[str, Any]:
    return asdict(record)
