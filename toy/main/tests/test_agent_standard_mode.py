from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.agent import build_agent
from graph_rag_app.config import (
    AppConfig,
    EmbeddingConfig,
    GenerationConfig,
    ModelConfig,
    RetrievalConfig,
    RuntimeConfig,
    WebConfig,
)


class AgentStandardModeTests(TestCase):
    def test_build_agent_loads_existing_index_without_ensuring_kb_index(self) -> None:
        app_config = AppConfig(
            kb_path=Path("unused-kb"),
            model=ModelConfig(api_key="test-key", api_base="https://example.invalid", model_name="model"),
            embedding=EmbeddingConfig(model="embed-model", max_batch_size=8),
            retrieval=RetrievalConfig(keyword_weight=0.25),
            web=WebConfig(enabled=False),
            generation=GenerationConfig(min_evidence_score=0.15, max_rounds=4),
            runtime=RuntimeConfig(),
        )

        with patch("graph_rag_app.agent.ChatOpenAI", return_value="model-instance"):
            with patch("graph_rag_app.agent.load_index", return_value="retriever") as load_index:
                with patch("graph_rag_app.agent.LocalRAGRetrieveTool", return_value="tool"):
                    with patch("graph_rag_app.agent.Agent", return_value="agent") as agent_cls:
                        result = build_agent(index_dir="existing-index", app_config=app_config)

        self.assertEqual(result, "agent")
        load_index.assert_called_once_with("existing-index", keyword_weight=0.25)
        agent_cls.assert_called_once()
