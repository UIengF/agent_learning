from __future__ import annotations

from pathlib import Path

from aicoding_app.cli import main
from aicoding_app.context_explain import explain_context
from aicoding_app.repo_map import build_repo_map
from aicoding_app.test_failures import parse_pytest_failures


def _write_demo_repo(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "app.py").write_text(
        "class App:\n    def run(self):\n        return 'ok'\n",
        encoding="utf-8",
    )
    (root / "src" / "consumer.py").write_text(
        "from src.app import App\n\n\ndef build_app():\n    return App()\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_app.py").write_text(
        "from src.app import App\n\n\ndef test_app():\n    assert App().run() == 'ok'\n",
        encoding="utf-8",
    )


def test_repo_map_explains_symbol_definition_and_related_files(tmp_path: Path) -> None:
    _write_demo_repo(tmp_path)

    explanation = build_repo_map(tmp_path).explain("App")

    assert "Symbol definitions:" in explanation
    assert "src/app.py:1: class App" in explanation
    assert "tests/test_app.py" in explanation


def test_repo_map_relates_source_and_test_files(tmp_path: Path) -> None:
    _write_demo_repo(tmp_path)
    repo_map = build_repo_map(tmp_path)

    assert "tests/test_app.py" in repo_map.related_files_for_path("src/app.py")
    assert "src/app.py" in repo_map.related_files_for_path("tests/test_app.py")


def test_repo_map_reports_imports_and_reverse_imports(tmp_path: Path) -> None:
    _write_demo_repo(tmp_path)
    repo_map = build_repo_map(tmp_path)

    assert "src.app" in repo_map.imports_for_path("src/consumer.py")
    assert "src/consumer.py" in repo_map.reverse_imports_for_path("src/app.py")
    assert "tests/test_app.py" in repo_map.reverse_imports_for_path("src/app.py")


def test_parse_pytest_failures_extracts_nodeid_location_and_summary() -> None:
    output = "\n".join(
        [
            "tests/test_app.py:4: AssertionError",
            "FAILED tests/test_app.py::test_app - AssertionError: expected ok",
        ]
    )

    failures = parse_pytest_failures(output)

    assert len(failures) == 1
    assert failures[0].path == "tests/test_app.py"
    assert failures[0].line == 4
    assert failures[0].test_name == "test_app"
    assert "expected ok" in failures[0].summary


def test_explain_context_links_pytest_failure_to_related_source(tmp_path: Path) -> None:
    _write_demo_repo(tmp_path)
    output = "FAILED tests/test_app.py::test_app - AssertionError: expected ok"

    explanation = explain_context(tmp_path, output)

    assert "Pytest failure context:" in explanation
    assert "tests/test_app.py" in explanation
    assert "src/app.py" in explanation


def test_context_explain_cli_prints_symbol_context(tmp_path: Path, capsys) -> None:
    _write_demo_repo(tmp_path)

    assert main(["dev", "context", "explain", "--workspace", str(tmp_path), "--query", "App"]) == 0

    output = capsys.readouterr().out
    assert "Symbol definitions:" in output
    assert "src/app.py:1: class App" in output
