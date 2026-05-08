from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from graph_rag_app.agent_evaluation import (
    AdaptedTrace,
    CaseEvaluationResult,
    EvaluationRunReport,
    LayerScore,
)
from graph_rag_app.question_frame import build_question_frame


class EvalReportingTests(TestCase):
    def test_save_evaluation_report_writes_summary_and_results_files(self) -> None:
        from graph_rag_app.eval_reporting import save_evaluation_report

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            report = EvaluationRunReport(
                dataset_name="agent-smoke",
                dataset_version="2026-04-19",
                case_count=1,
                passed_case_count=1,
                failed_case_count=0,
                pass_rate=1.0,
                layer_pass_rates={"question": 1.0, "retrieval": 1.0},
                results=[
                    CaseEvaluationResult(
                        case_id="case-1",
                        question="OpenAI agent docs?",
                        answer="answer",
                        passed=True,
                        layer_scores={
                            "question": LayerScore(
                                name="question",
                                status="passed",
                                score=1.0,
                                max_score=1.0,
                            )
                        },
                        trace=AdaptedTrace(
                            answer="answer",
                            tool_names=["local_rag_retrieve"],
                            sources=[
                                {
                                    "source_type": "local",
                                    "source_path": "OpenAI/agents.md",
                                    "section_title": "Overview",
                                }
                            ],
                            local_results=[],
                            web_searches=[],
                            question_frame=build_question_frame("OpenAI agent docs?"),
                        ),
                        judge_artifacts={
                            "answer": {
                                "judge_prompt": "prompt text",
                                "judge_response": {"passed": True, "score": 1.0},
                            }
                        },
                    )
                ],
            )

            artifact = save_evaluation_report(report, output_root=output_root)

            self.assertTrue(artifact.run_dir.exists())
            self.assertTrue(artifact.summary_path.exists())
            self.assertTrue(artifact.results_path.exists())
            summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["dataset_name"], "agent-smoke")
            self.assertEqual(summary["case_count"], 1)
            results = json.loads(artifact.results_path.read_text(encoding="utf-8"))
            self.assertEqual(len(results["results"]), 1)
            self.assertEqual(results["results"][0]["case_id"], "case-1")
            prompt_path = artifact.run_dir / "cases" / "case-1" / "judge-answer-prompt.txt"
            response_path = artifact.run_dir / "cases" / "case-1" / "judge-answer-response.json"
            self.assertTrue(prompt_path.exists())
            self.assertTrue(response_path.exists())

    def test_compare_reports_identifies_regressions_and_improvements(self) -> None:
        from graph_rag_app.eval_reporting import compare_evaluation_reports

        baseline = EvaluationRunReport(
            dataset_name="agent-smoke",
            dataset_version="2026-04-19",
            case_count=2,
            passed_case_count=2,
            failed_case_count=0,
            pass_rate=1.0,
            layer_pass_rates={"question": 1.0, "sources": 1.0},
            results=[
                CaseEvaluationResult(
                    case_id="case-a",
                    question="q1",
                    answer="a1",
                    passed=True,
                    layer_scores={
                        "sources": LayerScore(
                            name="sources", status="passed", score=1.0, max_score=1.0
                        )
                    },
                    trace=AdaptedTrace(
                        answer="a1",
                        tool_names=[],
                        sources=[],
                        local_results=[],
                        web_searches=[],
                        question_frame=build_question_frame("q1"),
                    ),
                ),
                CaseEvaluationResult(
                    case_id="case-b",
                    question="q2",
                    answer="a2",
                    passed=False,
                    layer_scores={
                        "sources": LayerScore(
                            name="sources", status="failed", score=0.0, max_score=1.0
                        )
                    },
                    trace=AdaptedTrace(
                        answer="a2",
                        tool_names=[],
                        sources=[],
                        local_results=[],
                        web_searches=[],
                        question_frame=build_question_frame("q2"),
                    ),
                ),
            ],
        )
        current = EvaluationRunReport(
            dataset_name="agent-smoke",
            dataset_version="2026-04-20",
            case_count=2,
            passed_case_count=1,
            failed_case_count=1,
            pass_rate=0.5,
            layer_pass_rates={"question": 1.0, "sources": 0.5},
            results=[
                CaseEvaluationResult(
                    case_id="case-a",
                    question="q1",
                    answer="a1",
                    passed=False,
                    layer_scores={
                        "sources": LayerScore(
                            name="sources", status="failed", score=0.0, max_score=1.0
                        )
                    },
                    trace=AdaptedTrace(
                        answer="a1",
                        tool_names=[],
                        sources=[],
                        local_results=[],
                        web_searches=[],
                        question_frame=build_question_frame("q1"),
                    ),
                ),
                CaseEvaluationResult(
                    case_id="case-b",
                    question="q2",
                    answer="a2",
                    passed=True,
                    layer_scores={
                        "sources": LayerScore(
                            name="sources", status="passed", score=1.0, max_score=1.0
                        )
                    },
                    trace=AdaptedTrace(
                        answer="a2",
                        tool_names=[],
                        sources=[],
                        local_results=[],
                        web_searches=[],
                        question_frame=build_question_frame("q2"),
                    ),
                ),
            ],
        )

        diff = compare_evaluation_reports(current, baseline)

        self.assertEqual(diff["summary"]["regression_count"], 1)
        self.assertEqual(diff["summary"]["improvement_count"], 1)
        self.assertEqual(diff["regressions"][0]["case_id"], "case-a")
        self.assertEqual(diff["improvements"][0]["case_id"], "case-b")
