from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.config import build_app_config, load_env_file


class EnvConfigTests(TestCase):
    def test_load_env_file_sets_missing_values_without_overriding_existing_environment(self) -> None:
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
