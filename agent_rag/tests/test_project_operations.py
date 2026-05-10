from __future__ import annotations

import json
import subprocess
import tomllib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectOperationsTests(unittest.TestCase):
    def test_readme_documents_langgraph_environment_and_service_scripts(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("python -m pip install -e .", readme)
        self.assertIn("scripts\\start.ps1", readme)
        self.assertIn("scripts\\stop.ps1", readme)
        self.assertIn("scripts\\status.ps1", readme)

    def test_requirements_do_not_list_unused_beautifulsoup_dependency(self) -> None:
        pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertNotIn("beautifulsoup4", pyproject_text)
        self.assertNotIn("bs4", pyproject_text)

    def test_quality_entrypoint_is_documented_for_local_and_ci_runs(self) -> None:
        pyproject_path = PROJECT_ROOT / "pyproject.toml"
        dev_requirements_path = PROJECT_ROOT / "requirements-dev.txt"
        workflow_path = PROJECT_ROOT.parent / ".github" / "workflows" / "agent-rag-quality.yml"

        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        dev_requirements = dev_requirements_path.read_text(encoding="utf-8")
        workflow = workflow_path.read_text(encoding="utf-8")
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("ruff", pyproject["tool"])
        self.assertIn("pyright", pyproject["tool"])
        self.assertIn("coverage", pyproject["tool"])
        self.assertIn("ruff", dev_requirements)
        self.assertIn("pyright", dev_requirements)
        self.assertIn("coverage", dev_requirements)
        self.assertIn("runs-on: ubuntu-latest", workflow)
        for command in (
            "python -m ruff format --check .",
            "python -m ruff check .",
            "python -m pyright",
            "python -m coverage run -m pytest tests/",
            "python -m coverage report",
        ):
            self.assertIn(command, workflow)
            self.assertIn(command, readme)

    def test_smoke_dataset_covers_extended_case_groups(self) -> None:
        records = [
            json.loads(line)
            for line in (PROJECT_ROOT / "evals" / "datasets" / "agent-smoke.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        metadata = records[0]["dataset"]
        cases = [record for record in records[1:] if "id" in record]

        self.assertIn("zh", metadata["groups"])
        self.assertIn("multi-hop", metadata["groups"])
        self.assertIn("scholar", metadata["groups"])
        self.assertIn("no-evidence", metadata["groups"])
        self.assertGreaterEqual(len(cases), 7)

    def test_service_scripts_use_langgraph_environment_and_project_paths(self) -> None:
        script_dir = PROJECT_ROOT / "scripts"
        for script_name in ("start.ps1", "stop.ps1", "status.ps1"):
            script = (script_dir / script_name).read_text(encoding="utf-8")
            self.assertIn("langgraph", script)
            self.assertIn("ProjectRoot", script)

        start_script = (script_dir / "start.ps1").read_text(encoding="utf-8")
        self.assertIn("conda run -n langgraph", start_script)
        self.assertIn("graph_rag.py", start_script)
        self.assertIn("8765", start_script)

    def test_status_script_runs_without_parse_errors(self) -> None:
        result = subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(PROJECT_ROOT / "scripts" / "status.ps1"),
                "-Port",
                "1",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Expected Conda environment: langgraph", result.stdout)


if __name__ == "__main__":
    unittest.main()
