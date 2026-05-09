from __future__ import annotations

from pathlib import Path

from aicoding_app.cli import main
from aicoding_app.text_hygiene import clean_text_hygiene, format_text_hygiene_report


def test_text_hygiene_check_reports_non_ascii_without_modifying(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text('"""bad тАФ dash"""\nVALUE = "ok"\n', encoding="utf-8")

    report = format_text_hygiene_report(tmp_path)

    assert "Text hygiene: non-ASCII characters found." in report
    assert "module.py:1" in report
    assert path.read_text(encoding="utf-8") == '"""bad тАФ dash"""\nVALUE = "ok"\n'


def test_text_hygiene_clean_applies_safe_replacements(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text('"""bad тАФ dash бк arrow тЬУ тЧЛ тЖТ ✓ ○ →"""\n', encoding="utf-8")

    result = clean_text_hygiene(tmp_path)

    assert "module.py" in result
    assert (
        path.read_text(encoding="utf-8")
        == '"""bad - dash -> arrow done open -> done open ->"""\n'
    )


def test_text_hygiene_cli_check_and_clean(tmp_path: Path, capsys) -> None:
    path = tmp_path / "README.md"
    path.write_text("hello — world\n", encoding="utf-8")

    assert main(["text", "check", "--workspace", str(tmp_path)]) == 0
    check_output = capsys.readouterr().out
    assert "README.md:1" in check_output
    assert "hello — world" in path.read_text(encoding="utf-8")

    assert main(["text", "clean", "--workspace", str(tmp_path)]) == 0
    clean_output = capsys.readouterr().out
    assert "README.md" in clean_output
    assert path.read_text(encoding="utf-8") == "hello - world\n"
