from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest import TestCase

from graph_rag_app.config import PermissionConfig
from graph_rag_app.jobs import BackgroundJobManager, JobNotFound
from graph_rag_app.permissions import PermissionDenied, ToolPermissionPolicy


class PermissionPolicyTests(TestCase):
    def test_validate_index_dir_allows_project_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy = ToolPermissionPolicy(
                config=PermissionConfig(allowed_index_roots="agent"),
                project_root=root,
            )

            self.assertEqual(policy.validate_index_dir("agent/index-a"), "agent/index-a")

    def test_validate_index_dir_rejects_path_outside_allowed_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy = ToolPermissionPolicy(
                config=PermissionConfig(allowed_index_roots="agent"),
                project_root=root,
            )

            with self.assertRaises(PermissionDenied):
                policy.validate_index_dir("../outside")

    def test_validate_web_fetch_rejects_localhost_when_disabled(self) -> None:
        policy = ToolPermissionPolicy(
            config=PermissionConfig(
                allow_local_web_fetch=False,
                allow_private_web_fetch=False,
            )
        )

        with self.assertRaises(PermissionDenied):
            policy.validate_web_fetch_url("http://localhost:8000")

    def test_validate_web_fetch_rejects_private_ip_when_disabled(self) -> None:
        policy = ToolPermissionPolicy(
            config=PermissionConfig(
                allow_local_web_fetch=True,
                allow_private_web_fetch=False,
            )
        )

        with self.assertRaises(PermissionDenied):
            policy.validate_web_fetch_url("http://127.0.0.1:8000")


class BackgroundJobManagerTests(TestCase):
    def test_job_manager_runs_target_and_persists_record_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BackgroundJobManager(tmpdir)

            record = manager.submit(
                "demo",
                lambda log_path: {"log_path": str(log_path), "value": 42},
            )
            for _ in range(100):
                current = manager.get(record.job_id)
                if current.status == "completed":
                    break
                time.sleep(0.01)

            current = manager.get(record.job_id)
            reloaded = BackgroundJobManager(tmpdir).get(record.job_id)
            log = manager.read_log(record.job_id)

        self.assertEqual(current.status, "completed")
        self.assertEqual(current.result["value"], 42)
        self.assertEqual(reloaded.status, "completed")
        self.assertIn("status=completed", log)

    def test_job_manager_reports_missing_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BackgroundJobManager(tmpdir)

            with self.assertRaises(JobNotFound):
                manager.get("missing")
