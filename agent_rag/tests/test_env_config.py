from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.config import build_app_config, load_env_file


class EnvConfigTests(TestCase):
    def test_load_env_file_sets_missing_values_without_overriding_existing_environment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "DASHSCOPE_API_KEY=from-file",
                        "RAG_WEB_ENABLED=false",
                        "IGNORED_LINE_WITHOUT_EQUALS",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "from-env"}, clear=False):
                load_env_file(env_path)

                self.assertEqual(os.environ["DASHSCOPE_API_KEY"], "from-env")
                self.assertEqual(os.environ["RAG_WEB_ENABLED"], "false")

    def test_build_app_config_loads_project_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("DASHSCOPE_API_KEY=from-project-env\n", encoding="utf-8")

            with patch("graph_rag_app.config.PROJECT_ENV_PATH", env_path):
                with patch.dict(os.environ, {}, clear=True):
                    config = build_app_config(".")

        self.assertEqual(config.model.api_key, "from-project-env")

    def test_build_app_config_sets_model_provider_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            with patch("graph_rag_app.config.PROJECT_ENV_PATH", env_path):
                with patch.dict(
                    os.environ,
                    {
                        "RAG_MODEL_API_KEY": "model-key",
                        "RAG_MODEL_API_BASE": "https://api.deepseek.com",
                        "RAG_MODEL_NAME": "deepseek-v4-flash",
                    },
                    clear=True,
                ):
                    config = build_app_config(".")

        self.assertEqual(config.model.api_key, "model-key")
        self.assertEqual(config.model.api_base, "https://api.deepseek.com")
        self.assertEqual(config.model.model_name, "deepseek-v4-flash")

    def test_build_app_config_sets_eval_judge_defaults_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            with patch("graph_rag_app.config.PROJECT_ENV_PATH", env_path):
                with patch.dict(
                    os.environ,
                    {
                        "DASHSCOPE_API_KEY": "judge-key",
                        "RAG_EVAL_JUDGE_ENABLED": "true",
                        "RAG_EVAL_JUDGE_MODEL": "qwen3.6-plus",
                        "RAG_EVAL_JUDGE_API_BASE": (
                            "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        ),
                    },
                    clear=True,
                ):
                    config = build_app_config(".")

        self.assertTrue(config.eval_judge.enabled)
        self.assertEqual(config.eval_judge.api_key, "judge-key")
        self.assertEqual(config.eval_judge.model_name, "qwen3.6-plus")
        self.assertEqual(
            config.eval_judge.api_base,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def test_build_app_config_sets_langsmith_defaults_and_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAG_LANGSMITH_ENABLED": "true",
                "LANGCHAIN_TRACING_V2": "true",
                "LANGCHAIN_API_KEY": "ls-key",
                "LANGCHAIN_PROJECT": "agent-rag-evals",
                "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
            },
            clear=True,
        ):
            config = build_app_config(".")

        self.assertTrue(config.langsmith.enabled)
        self.assertTrue(config.langsmith.tracing_v2)
        self.assertEqual(config.langsmith.api_key, "ls-key")
        self.assertEqual(config.langsmith.project_name, "agent-rag-evals")
        self.assertEqual(config.langsmith.endpoint, "https://api.smith.langchain.com")

    def test_build_app_config_sets_harness_defaults_and_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAG_SKILLS_DIR": "custom-skills",
                "RAG_STRUCTURED_TRACE_ENABLED": "false",
                "RAG_STRUCTURED_TRACE_DIR": "custom-traces",
            },
            clear=True,
        ):
            config = build_app_config(".")

        self.assertEqual(config.harness.skills_dir, "custom-skills")
        self.assertFalse(config.harness.structured_trace_enabled)
        self.assertEqual(config.harness.structured_trace_dir, "custom-traces")

    def test_build_app_config_sets_permission_and_job_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAG_ALLOWED_INDEX_ROOTS": "agent;runtime",
                "RAG_ALLOW_LOCAL_WEB_FETCH": "false",
                "RAG_ALLOW_PRIVATE_WEB_FETCH": "false",
                "RAG_JOB_RUNTIME_DIR": "custom-jobs",
                "RAG_JOB_MAX_LOG_CHARS": "2500",
            },
            clear=True,
        ):
            config = build_app_config(".")

        self.assertEqual(config.permissions.allowed_index_roots, "agent;runtime")
        self.assertFalse(config.permissions.allow_local_web_fetch)
        self.assertFalse(config.permissions.allow_private_web_fetch)
        self.assertEqual(config.jobs.runtime_dir, "custom-jobs")
        self.assertEqual(config.jobs.max_log_chars, 2500)
