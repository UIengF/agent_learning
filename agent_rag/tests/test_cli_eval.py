from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


class CliEvalCommandTests(TestCase):
    def test_parse_args_supports_eval_run_command(self) -> None:
        from graph_rag_app.cli import parse_args

        args = parse_args(
            [
                "eval",
                "run",
                "--dataset",
                "evals/datasets/agent-smoke.jsonl",
                "--index-dir",
                "agent",
                "--output-dir",
                "runtime/evals",
                "--baseline-run",
                "runtime/evals/20260420-agent-smoke/results.json",
                "--judge-enabled",
                "--judge-model",
                "qwen3.6-plus",
                "--judge-api-base",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ]
        )

        self.assertEqual(args.command, "eval")
        self.assertEqual(args.eval_command, "run")
        self.assertEqual(args.dataset, "evals/datasets/agent-smoke.jsonl")
        self.assertEqual(args.index_dir, "agent")
        self.assertEqual(args.output_dir, "runtime/evals")
        self.assertEqual(args.baseline_run, "runtime/evals/20260420-agent-smoke/results.json")
        self.assertTrue(args.judge_enabled)
        self.assertEqual(args.judge_model, "qwen3.6-plus")
        self.assertEqual(args.judge_api_base, "https://dashscope.aliyuncs.com/compatible-mode/v1")

    def test_main_runs_evaluation_and_saves_report(self) -> None:
        from graph_rag_app.agent_evaluation import EvaluationRunReport
        from graph_rag_app.cli import main

        fake_report = EvaluationRunReport(
            dataset_name="agent-smoke",
            dataset_version="2026-04-19",
            case_count=3,
            passed_case_count=2,
            failed_case_count=1,
            pass_rate=2 / 3,
            layer_pass_rates={"question": 1.0, "sources": 0.5},
            results=[],
        )

        with patch(
            "graph_rag_app.cli.load_evaluation_dataset", return_value="dataset"
        ) as load_dataset:
            with patch(
                "graph_rag_app.cli.run_evaluation_dataset", return_value=fake_report
            ) as run_dataset:
                with patch("graph_rag_app.cli.save_evaluation_report") as save_report:
                    save_report.return_value.run_dir = Path("runtime/evals/20260420-agent-smoke")
                    with patch("graph_rag_app.cli.build_app_config") as build_app_config:
                        build_app_config.return_value = "app-config"
                        with patch(
                            "graph_rag_app.cli.override_eval_judge_config",
                            return_value="overridden-config",
                        ) as override_config:
                            with patch("builtins.print") as print_output:
                                exit_code = main(
                                    [
                                        "eval",
                                        "run",
                                        "--dataset",
                                        "evals/datasets/agent-smoke.jsonl",
                                        "--index-dir",
                                        "agent",
                                        "--output-dir",
                                        "runtime/evals",
                                        "--judge-enabled",
                                        "--judge-model",
                                        "qwen3.6-plus",
                                    ]
                                )

        self.assertEqual(exit_code, 0)
        load_dataset.assert_called_once_with(
            Path("evals/datasets/agent-smoke.jsonl"), include_tags=None
        )
        build_app_config.assert_called_once_with("agent")
        override_config.assert_called_once()
        run_dataset.assert_called_once_with(
            "dataset", index_dir="agent", app_config="overridden-config"
        )
        save_report.assert_called_once()
        print_output.assert_called()
