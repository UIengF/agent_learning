from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .agent_evaluation import EvaluationRunReport


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


class EvaluationReportArtifact:
    def __init__(self, run_dir: Path, summary_path: Path, results_path: Path) -> None:
        self.run_dir = run_dir
        self.summary_path = summary_path
        self.results_path = results_path


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_evaluation_reports(
    current: EvaluationRunReport | dict[str, Any],
    baseline: EvaluationRunReport | dict[str, Any],
) -> dict[str, Any]:
    current_data = asdict(current) if is_dataclass(current) else current
    baseline_data = asdict(baseline) if is_dataclass(baseline) else baseline
    current_results = {item["case_id"]: item for item in current_data.get("results", [])}
    baseline_results = {item["case_id"]: item for item in baseline_data.get("results", [])}
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []

    for case_id in sorted(set(current_results) & set(baseline_results)):
        current_case = current_results[case_id]
        baseline_case = baseline_results[case_id]
        current_passed = bool(current_case.get("passed"))
        baseline_passed = bool(baseline_case.get("passed"))
        if baseline_passed and not current_passed:
            regressions.append(
                {
                    "case_id": case_id,
                    "baseline_passed": baseline_passed,
                    "current_passed": current_passed,
                }
            )
        elif not baseline_passed and current_passed:
            improvements.append(
                {
                    "case_id": case_id,
                    "baseline_passed": baseline_passed,
                    "current_passed": current_passed,
                }
            )
        else:
            unchanged.append(
                {
                    "case_id": case_id,
                    "baseline_passed": baseline_passed,
                    "current_passed": current_passed,
                }
            )

    return {
        "summary": {
            "baseline_dataset_version": baseline_data.get("dataset_version"),
            "current_dataset_version": current_data.get("dataset_version"),
            "regression_count": len(regressions),
            "improvement_count": len(improvements),
            "unchanged_count": len(unchanged),
        },
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
    }


def save_evaluation_report(
    report: EvaluationRunReport,
    *,
    output_root: str | Path = Path("runtime") / "evals",
    now: datetime | None = None,
    baseline_report: EvaluationRunReport | dict[str, Any] | None = None,
) -> EvaluationReportArtifact:
    current = now or datetime.now().astimezone()
    slug = report.dataset_name.replace(" ", "-").replace("_", "-").lower()
    run_dir = Path(output_root) / f"{current.strftime('%Y%m%d-%H%M%S')}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    results_path = run_dir / "results.json"
    comparison_path = run_dir / "comparison.json"

    summary_payload = {
        "dataset_name": report.dataset_name,
        "dataset_version": report.dataset_version,
        "case_count": report.case_count,
        "passed_case_count": report.passed_case_count,
        "failed_case_count": report.failed_case_count,
        "pass_rate": report.pass_rate,
        "layer_pass_rates": report.layer_pass_rates,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    results_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_case_artifacts(report, run_dir)
    if baseline_report is not None:
        comparison_path.write_text(
            json.dumps(
                compare_evaluation_reports(report, baseline_report),
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
    return EvaluationReportArtifact(
        run_dir=run_dir,
        summary_path=summary_path,
        results_path=results_path,
    )


def _safe_case_id(case_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in case_id)
    return safe.strip("-") or "case"


def _write_case_artifacts(report: EvaluationRunReport, run_dir: Path) -> None:
    cases_dir = run_dir / "cases"
    for result in report.results:
        if not result.judge_artifacts:
            continue
        case_dir = cases_dir / _safe_case_id(result.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        for layer_name, artifact in result.judge_artifacts.items():
            prompt = artifact.get("judge_prompt")
            response = artifact.get("judge_response")
            if prompt is not None:
                (case_dir / f"judge-{layer_name}-prompt.txt").write_text(
                    str(prompt),
                    encoding="utf-8",
                )
            if response is not None:
                (case_dir / f"judge-{layer_name}-response.json").write_text(
                    json.dumps(response, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8",
                )
