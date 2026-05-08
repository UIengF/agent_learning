from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.eval_datasets import EvaluationCase
from graph_rag_app.config import build_app_config
from graph_rag_app.runtime import AgentRunTrace


class AgentEvaluationTests(TestCase):
    def test_evaluate_case_returns_layered_scores_from_trace(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "comparison-openai-gemini-agent",
                "question": "OpenAI 和 Gemini 在 agent 实现中有哪些异同点？",
                "expected_entities": ["OpenAI", "Gemini"],
                "assertions": {
                    "question": {
                        "expected_intent": "comparison",
                    },
                    "trajectory": {
                        "must_use_tools": ["web_search"],
                    },
                    "sources": {
                        "min_source_count": 2,
                        "required_source_types": ["web"],
                    },
                },
            }
        )
        trace = AgentRunTrace(
            answer="OpenAI and Gemini both support tool-using agents, but they differ in model ecosystem and platform integration.",
            messages=[
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": json.dumps(
                        {
                            "results": [
                                {
                                    "title": "OpenAI Agents",
                                    "url": "https://platform.openai.com/docs/guides/agents",
                                    "snippet": "OpenAI agent guide",
                                }
                            ]
                        }
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": json.dumps(
                        {
                            "url": "https://platform.openai.com/docs/guides/agents",
                            "final_url": "https://platform.openai.com/docs/guides/agents",
                            "title": "OpenAI Agents",
                            "text": "OpenAI agent implementation details.",
                        }
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": json.dumps(
                        {
                            "url": "https://ai.google.dev/gemini-api/docs/function-calling",
                            "final_url": "https://ai.google.dev/gemini-api/docs/function-calling",
                            "title": "Gemini Function Calling",
                            "text": "Gemini tool use and function calling details.",
                        }
                    ),
                },
            ],
        )

        result = evaluate_case(case, index_dir="agent", agent_runner=lambda **_: trace)

        self.assertEqual(result.case_id, "comparison-openai-gemini-agent")
        self.assertTrue(result.passed)
        self.assertEqual(result.layer_scores["question"].status, "passed")
        self.assertEqual(result.layer_scores["trajectory"].status, "passed")
        self.assertEqual(result.layer_scores["sources"].status, "passed")
        self.assertEqual(result.layer_scores["answer"].status, "not_applicable")
        self.assertIn("web_search", result.trace.tool_names)
        self.assertEqual(len(result.trace.sources), 2)

    def test_evaluate_case_reports_failed_layers_and_reasons(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "web-needed",
                "question": "Gemini agent 官方文档里怎么描述 tool use？",
                "expected_entities": ["Gemini"],
                "assertions": {
                    "trajectory": {
                        "must_use_tools": ["web_search"],
                        "must_not_use_tools": ["scholar_search"],
                    },
                    "sources": {
                        "min_source_count": 1,
                        "required_source_types": ["web"],
                        "required_domains": ["google.dev"],
                    },
                },
            }
        )
        trace = AgentRunTrace(
            answer="This answer is too generic.",
            messages=[
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": json.dumps(
                        {
                            "results": [
                                {
                                    "chunk_id": 1,
                                    "score": 1.12,
                                    "text": "Anthropic agent notes.",
                                    "document_id": "doc-1",
                                    "source_path": "Anthropic/agents.md",
                                    "section_title": "Overview",
                                    "strategy": "hybrid",
                                }
                            ]
                        }
                    ),
                }
            ],
        )

        result = evaluate_case(case, index_dir="agent", agent_runner=lambda **_: trace)

        self.assertFalse(result.passed)
        self.assertEqual(result.layer_scores["trajectory"].status, "failed")
        self.assertEqual(result.layer_scores["sources"].status, "failed")
        self.assertIn(
            "Missing required tools: web_search", result.layer_scores["trajectory"].reasons
        )
        self.assertIn("Missing required source types: web", result.layer_scores["sources"].reasons)

    def test_evaluate_case_scores_local_retrieval_entity_consistency(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "local-openai-overview",
                "question": "本地知识库有哪些关于 OpenAI 的资料？",
                "expected_entities": ["OpenAI"],
                "assertions": {
                    "retrieval": {
                        "min_result_count": 1,
                        "expected_entities": ["OpenAI"],
                    }
                },
            }
        )
        trace = AgentRunTrace(
            answer="OpenAI local overview.",
            messages=[
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": json.dumps(
                        {
                            "results": [
                                {
                                    "chunk_id": 1,
                                    "score": 1.12,
                                    "text": "OpenAI agent architecture notes.",
                                    "document_id": "doc-1",
                                    "source_path": "OpenAI/agents.md",
                                    "section_title": "Overview",
                                    "strategy": "hybrid",
                                }
                            ]
                        }
                    ),
                }
            ],
        )

        result = evaluate_case(case, index_dir="agent", agent_runner=lambda **_: trace)

        self.assertEqual(result.layer_scores["retrieval"].status, "passed")
        self.assertGreaterEqual(result.layer_scores["retrieval"].score, 1.0)

    def test_run_evaluation_dataset_returns_aggregate_metrics(self) -> None:
        from graph_rag_app.agent_evaluation import run_evaluation_dataset
        from graph_rag_app.eval_datasets import load_evaluation_dataset
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "suite.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps({"dataset": {"name": "agent-suite", "version": "2026-04-19"}}),
                        json.dumps(
                            {
                                "id": "local-openai",
                                "question": "OpenAI local docs?",
                                "expected_entities": ["OpenAI"],
                                "assertions": {
                                    "retrieval": {
                                        "min_result_count": 1,
                                        "expected_entities": ["OpenAI"],
                                    }
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "id": "gemini-web",
                                "question": "Gemini docs tool use",
                                "expected_entities": ["Gemini"],
                                "assertions": {
                                    "trajectory": {"must_use_tools": ["web_search"]},
                                    "sources": {
                                        "min_source_count": 1,
                                        "required_source_types": ["web"],
                                    },
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            dataset = load_evaluation_dataset(dataset_path)

            def fake_runner(*, question: str, index_dir: str, resume: bool = False):
                if "Gemini" in question:
                    return AgentRunTrace(
                        answer="Gemini docs answer.",
                        messages=[
                            {
                                "role": "tool",
                                "name": "web_search",
                                "content": json.dumps({"results": []}),
                            },
                            {
                                "role": "tool",
                                "name": "web_fetch",
                                "content": json.dumps(
                                    {
                                        "url": "https://google.dev/gemini/tool-use",
                                        "final_url": "https://google.dev/gemini/tool-use",
                                        "title": "Gemini Tool Use",
                                        "text": "Gemini tool use docs.",
                                    }
                                ),
                            },
                        ],
                    )
                return AgentRunTrace(
                    answer="Local OpenAI answer.",
                    messages=[
                        {
                            "role": "tool",
                            "name": "local_rag_retrieve",
                            "content": json.dumps(
                                {
                                    "results": [
                                        {
                                            "chunk_id": 1,
                                            "score": 1.12,
                                            "text": "OpenAI local notes.",
                                            "document_id": "doc-1",
                                            "source_path": "OpenAI/agents.md",
                                            "section_title": "Overview",
                                            "strategy": "hybrid",
                                        }
                                    ]
                                }
                            ),
                        }
                    ],
                )

            report = run_evaluation_dataset(dataset, index_dir="agent", agent_runner=fake_runner)

        self.assertEqual(report.dataset_name, "agent-suite")
        self.assertEqual(report.case_count, 2)
        self.assertEqual(report.passed_case_count, 2)
        self.assertIn("question", report.layer_pass_rates)
        self.assertIn("sources", report.layer_pass_rates)
        self.assertIsNone(report.layer_pass_rates["answer"])

    def test_evaluate_case_uses_qwen_judge_for_answer_layer_when_enabled(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "judge-answer",
                "question": "Compare OpenAI and Gemini agent implementations.",
                "expected_entities": ["OpenAI", "Gemini"],
                "assertions": {
                    "answer": {
                        "must_cover_points": ["OpenAI", "Gemini"],
                        "reference_answer": "Should compare both ecosystems and tool use.",
                    }
                },
            }
        )
        trace = AgentRunTrace(
            answer="OpenAI and Gemini both support tool use, but differ in ecosystem integration.",
            messages=[
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": json.dumps(
                        {
                            "url": "https://platform.openai.com/docs/guides/agents",
                            "final_url": "https://platform.openai.com/docs/guides/agents",
                            "title": "OpenAI Agents",
                            "text": "OpenAI agent docs.",
                        }
                    ),
                }
            ],
        )

        class FakeJudge:
            def invoke(self, prompt):
                class Response:
                    content = json.dumps(
                        {
                            "passed": True,
                            "score": 0.92,
                            "reasons": ["The answer is grounded and sufficiently complete."],
                            "grounded": True,
                            "complete": True,
                        }
                    )

                return Response()

        with patch.dict(
            "os.environ",
            {
                "DASHSCOPE_API_KEY": "judge-key",
                "RAG_EVAL_JUDGE_ENABLED": "true",
                "RAG_EVAL_JUDGE_MODEL": "qwen3.6-plus",
            },
            clear=False,
        ):
            app_config = build_app_config(".")

        result = evaluate_case(
            case,
            index_dir="agent",
            agent_runner=lambda **_: trace,
            app_config=app_config,
            judge_runner=FakeJudge(),
        )

        self.assertEqual(result.layer_scores["answer"].status, "passed")
        self.assertGreaterEqual(result.layer_scores["answer"].score, 0.9)
        self.assertEqual(result.layer_scores["answer"].details["judge_model"], "qwen3.6-plus")
        self.assertIn("judge_prompt", result.judge_artifacts["answer"])
        self.assertIn("judge_response", result.judge_artifacts["answer"])

    def test_evaluate_case_scores_web_search_and_source_quality(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "web-quality",
                "question": "Compare OpenAI and Gemini agent implementations.",
                "expected_entities": ["OpenAI", "Gemini"],
                "assertions": {
                    "web_search": {
                        "required_query_terms": ["OpenAI", "Gemini"],
                    },
                    "source_quality": {
                        "require_entity_coverage": True,
                    },
                },
            }
        )
        trace = AgentRunTrace(
            answer="answer",
            messages=[
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": json.dumps(
                        {
                            "query": "OpenAI Gemini agent implementation comparison",
                            "results": [
                                {
                                    "title": "OpenAI Agents from a mirror",
                                    "url": "https://csdn.net/openai-agent",
                                    "snippet": "OpenAI Agents SDK.",
                                },
                                {
                                    "title": "Gemini Function Calling from a blog",
                                    "url": "https://juejin.cn/gemini-agent",
                                    "snippet": "Gemini function calling.",
                                },
                            ],
                        }
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": json.dumps(
                        {
                            "url": "https://platform.openai.com/docs/guides/agents",
                            "final_url": "https://platform.openai.com/docs/guides/agents",
                            "title": "OpenAI Agents",
                            "text": "OpenAI Agents SDK.",
                        }
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": json.dumps(
                        {
                            "url": "https://ai.google.dev/gemini-api/docs/function-calling",
                            "final_url": "https://ai.google.dev/gemini-api/docs/function-calling",
                            "title": "Gemini Function Calling",
                            "text": "Gemini function calling.",
                        }
                    ),
                },
            ],
        )

        result = evaluate_case(case, index_dir="agent", agent_runner=lambda **_: trace)

        self.assertEqual(result.layer_scores["web_search_quality"].status, "passed")
        self.assertEqual(result.layer_scores["source_quality"].status, "passed")

    def test_evaluate_case_treats_scholar_results_as_sources(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "scholar-graph-rag-survey",
                "question": "Find recent papers or surveys related to Graph RAG agent evaluation.",
                "expected_entities": ["RAG"],
                "assertions": {
                    "trajectory": {"must_use_tools": ["scholar_search"]},
                    "sources": {
                        "min_source_count": 1,
                        "required_source_types": ["scholar"],
                    },
                },
            }
        )
        trace = AgentRunTrace(
            answer="A grounded summary of one Graph RAG survey.",
            messages=[
                {
                    "role": "tool",
                    "name": "scholar_search",
                    "content": json.dumps(
                        {
                            "topic": "graph rag agent evaluation",
                            "planned_queries": ["graph rag agent evaluation survey"],
                            "result_count": 1,
                            "results": [
                                {
                                    "title": "Graph retrieval-augmented generation: A survey",
                                    "url": "https://arxiv.org/abs/2501.12345",
                                    "snippet": "A survey of GraphRAG methods and evaluation settings.",
                                    "publication_summary": "arXiv preprint 2025",
                                    "year": 2025,
                                    "cited_by_count": 128,
                                    "resources": [],
                                    "source_query": "graph rag agent evaluation survey",
                                    "rank": 1,
                                    "source": "google_scholar_serpapi",
                                }
                            ],
                        }
                    ),
                }
            ],
        )

        result = evaluate_case(case, index_dir="agent", agent_runner=lambda **_: trace)

        self.assertEqual(result.layer_scores["trajectory"].status, "passed")
        self.assertEqual(result.layer_scores["sources"].status, "passed")
        self.assertEqual(result.trace.sources[0]["source_type"], "scholar")

    def test_web_search_and_source_quality_use_local_and_web_union_for_entities(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "union-coverage",
                "question": "Compare OpenAI and Gemini agent implementations.",
                "expected_entities": ["OpenAI", "Gemini"],
                "assertions": {
                    "web_search": {"required_query_terms": ["OpenAI", "Gemini"]},
                    "source_quality": {"require_entity_coverage": True},
                },
            }
        )
        trace = AgentRunTrace(
            answer="answer",
            messages=[
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": json.dumps(
                        {
                            "results": [
                                {
                                    "chunk_id": 1,
                                    "score": 1.0,
                                    "text": "OpenAI Agents SDK local notes.",
                                    "document_id": "doc-1",
                                    "source_path": "OpenAI/agents.md",
                                    "section_title": "OpenAI Agents",
                                    "strategy": "hybrid",
                                }
                            ]
                        }
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": json.dumps(
                        {
                            "query": "Gemini agent tool use",
                            "results": [
                                {
                                    "title": "Gemini tool use",
                                    "url": "https://example.com/gemini",
                                    "snippet": "Gemini agent tools.",
                                }
                            ],
                        }
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": json.dumps(
                        {
                            "url": "https://example.com/gemini",
                            "final_url": "https://example.com/gemini",
                            "title": "Gemini tool use",
                            "text": "Gemini agent tools.",
                        }
                    ),
                },
            ],
        )

        result = evaluate_case(case, index_dir="agent", agent_runner=lambda **_: trace)

        self.assertEqual(result.layer_scores["web_search_quality"].status, "passed")
        self.assertEqual(result.layer_scores["source_quality"].status, "passed")

    def test_evaluate_case_forwards_eval_metadata_to_runner(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "local-openai-overview",
                "question": "本地知识库有哪些关于 OpenAI 的资料？",
                "tags": ["smoke", "local"],
                "group": "local",
                "expected_entities": ["OpenAI"],
                "assertions": {
                    "retrieval": {
                        "min_result_count": 1,
                        "expected_entities": ["OpenAI"],
                    }
                },
            }
        )
        captured: dict[str, object] = {}

        def fake_runner(**kwargs):
            captured.update(kwargs)
            return AgentRunTrace(
                answer="OpenAI local overview.",
                messages=[
                    {
                        "role": "tool",
                        "name": "local_rag_retrieve",
                        "content": json.dumps(
                            {
                                "results": [
                                    {
                                        "chunk_id": 1,
                                        "score": 1.12,
                                        "text": "OpenAI agent architecture notes.",
                                        "document_id": "doc-1",
                                        "source_path": "OpenAI/agents.md",
                                        "section_title": "Overview",
                                        "strategy": "hybrid",
                                    }
                                ]
                            }
                        ),
                    }
                ],
            )

        evaluate_case(case, index_dir="agent", agent_runner=fake_runner)

        self.assertEqual(captured["run_metadata"]["mode"], "eval")
        self.assertEqual(captured["run_metadata"]["case_id"], "local-openai-overview")
        self.assertEqual(captured["run_metadata"]["group"], "local")
        self.assertEqual(captured["run_metadata"]["tags"], ["smoke", "local"])

    def test_evaluate_case_syncs_layer_feedback_to_langsmith(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "local-openai-overview",
                "question": "本地知识库有哪些关于 OpenAI 的资料？",
                "tags": ["smoke", "local"],
                "group": "local",
                "expected_entities": ["OpenAI"],
                "assertions": {
                    "retrieval": {
                        "min_result_count": 1,
                        "expected_entities": ["OpenAI"],
                    }
                },
            }
        )
        trace = AgentRunTrace(
            answer="OpenAI local overview.",
            messages=[
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": json.dumps(
                        {
                            "results": [
                                {
                                    "chunk_id": 1,
                                    "score": 1.12,
                                    "text": "OpenAI agent architecture notes.",
                                    "document_id": "doc-1",
                                    "source_path": "OpenAI/agents.md",
                                    "section_title": "Overview",
                                    "strategy": "hybrid",
                                }
                            ]
                        }
                    ),
                }
            ],
            langsmith_run_id="run-123",
        )
        feedback_calls: list[dict[str, object]] = []

        with patch("graph_rag_app.agent_evaluation.write_langsmith_feedback") as write_feedback:
            write_feedback.side_effect = lambda **kwargs: feedback_calls.append(kwargs)
            evaluate_case(
                case,
                index_dir="agent",
                agent_runner=lambda **_: trace,
                app_config=build_app_config("."),
                dataset_name="agent-smoke",
            )

        self.assertTrue(any(call["key"] == "case_passed" for call in feedback_calls))
        self.assertTrue(any(call["key"] == "layer_question" for call in feedback_calls))
        self.assertTrue(any(call["key"] == "layer_retrieval" for call in feedback_calls))

    def test_evaluate_case_syncs_answer_and_trajectory_feedback_to_langsmith(self) -> None:
        from graph_rag_app.agent_evaluation import evaluate_case

        case = EvaluationCase.model_validate(
            {
                "id": "judge-answer",
                "question": "Compare OpenAI and Gemini agent implementations.",
                "tags": ["smoke", "web"],
                "group": "comparison",
                "expected_entities": ["OpenAI", "Gemini"],
                "assertions": {
                    "answer": {
                        "must_cover_points": ["OpenAI", "Gemini"],
                        "reference_answer": "Should compare both ecosystems and tool use.",
                    }
                },
            }
        )
        trace = AgentRunTrace(
            answer="OpenAI and Gemini both support tool use, but differ in ecosystem integration.",
            messages=[
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": json.dumps(
                        {
                            "query": "OpenAI Gemini agent implementation comparison",
                            "results": [
                                {
                                    "title": "OpenAI Agents",
                                    "url": "https://platform.openai.com/docs/guides/agents",
                                    "snippet": "OpenAI Agents SDK.",
                                }
                            ],
                        }
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": json.dumps(
                        {
                            "url": "https://platform.openai.com/docs/guides/agents",
                            "final_url": "https://platform.openai.com/docs/guides/agents",
                            "title": "OpenAI Agents",
                            "text": "OpenAI agent docs.",
                        }
                    ),
                },
            ],
            langsmith_run_id="run-judge-123",
        )

        class FakeJudge:
            def invoke(self, prompt):
                class Response:
                    content = json.dumps(
                        {
                            "passed": True,
                            "score": 0.92,
                            "reasons": ["The answer is grounded and sufficiently complete."],
                            "grounded": True,
                            "complete": True,
                        }
                    )

                return Response()

        feedback_calls: list[dict[str, object]] = []
        with patch.dict(
            "os.environ",
            {
                "DASHSCOPE_API_KEY": "judge-key",
                "RAG_EVAL_JUDGE_ENABLED": "true",
                "RAG_EVAL_JUDGE_MODEL": "qwen3.6-plus",
            },
            clear=False,
        ):
            app_config = build_app_config(".")
        with patch("graph_rag_app.agent_evaluation.write_langsmith_feedback") as write_feedback:
            write_feedback.side_effect = lambda **kwargs: feedback_calls.append(kwargs)
            evaluate_case(
                case,
                index_dir="agent",
                agent_runner=lambda **_: trace,
                app_config=app_config,
                judge_runner=FakeJudge(),
            )

        self.assertTrue(any(call["key"] == "trajectory_summary" for call in feedback_calls))
        self.assertTrue(any(call["key"] == "answer_judge_score" for call in feedback_calls))
        self.assertTrue(any(call["key"] == "answer_grounded" for call in feedback_calls))
        self.assertTrue(any(call["key"] == "answer_complete" for call in feedback_calls))

    def test_run_evaluation_dataset_syncs_dataset_feedback_to_langsmith(self) -> None:
        from graph_rag_app.agent_evaluation import run_evaluation_dataset
        from graph_rag_app.eval_datasets import load_evaluation_dataset
        import tempfile
        from contextlib import contextmanager

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "suite.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "dataset": {
                                    "name": "agent-suite",
                                    "version": "2026-04-24",
                                    "tags": ["smoke", "local"],
                                }
                            }
                        ),
                        json.dumps(
                            {
                                "id": "local-openai",
                                "question": "OpenAI local docs?",
                                "expected_entities": ["OpenAI"],
                                "assertions": {
                                    "retrieval": {
                                        "min_result_count": 1,
                                        "expected_entities": ["OpenAI"],
                                    }
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            dataset = load_evaluation_dataset(dataset_path)

            @contextmanager
            def fake_langsmith_context(*, config, run_name, metadata, inputs, tags=None):
                class FakeRun:
                    id = "dataset-run-123"

                yield FakeRun()

            feedback_calls: list[dict[str, object]] = []
            with patch(
                "graph_rag_app.agent_evaluation.langsmith_run_context",
                side_effect=fake_langsmith_context,
            ):
                with patch(
                    "graph_rag_app.agent_evaluation.write_langsmith_feedback"
                ) as write_feedback:
                    write_feedback.side_effect = lambda **kwargs: feedback_calls.append(kwargs)
                    report = run_evaluation_dataset(
                        dataset,
                        index_dir="agent",
                        agent_runner=lambda **_: AgentRunTrace(
                            answer="Local OpenAI answer.",
                            messages=[
                                {
                                    "role": "tool",
                                    "name": "local_rag_retrieve",
                                    "content": json.dumps(
                                        {
                                            "results": [
                                                {
                                                    "chunk_id": 1,
                                                    "score": 1.12,
                                                    "text": "OpenAI local notes.",
                                                    "document_id": "doc-1",
                                                    "source_path": "OpenAI/agents.md",
                                                    "section_title": "Overview",
                                                    "strategy": "hybrid",
                                                }
                                            ]
                                        }
                                    ),
                                }
                            ],
                        ),
                        app_config=build_app_config("."),
                    )

        self.assertEqual(report.langsmith_run_id, "dataset-run-123")
        self.assertTrue(any(call["key"] == "dataset_pass_rate" for call in feedback_calls))
        self.assertTrue(any(call["key"] == "dataset_layer_retrieval" for call in feedback_calls))
