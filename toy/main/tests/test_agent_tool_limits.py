from __future__ import annotations

import unittest

from graph_rag_app.agent import Agent


class FakeMessage:
    def __init__(self, content: str = "", tool_calls: list[dict] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.type = "assistant"


class FakeModel:
    def __init__(self, message: FakeMessage):
        self._message = message

    def invoke(self, messages):  # noqa: ANN001
        return self._message


def _tool_round(tool_name: str, *, query: str, call_id: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "name": tool_name, "args": {"query": query}}],
        },
        {"role": "tool", "name": tool_name, "content": '{"result_count": 1, "results": []}'},
    ]


class AgentToolLimitTests(unittest.TestCase):
    def _build_agent(self, tool_calls: list[dict], *, max_rounds: int = 3) -> Agent:
        agent = Agent(max_rounds=max_rounds)
        agent.model = FakeModel(FakeMessage(content="need tool", tool_calls=tool_calls))
        return agent

    def test_new_question_ignores_prior_tool_history(self) -> None:
        state = {
            "messages": [
                {"role": "human", "content": "old question"},
                *_tool_round("local_rag_retrieve", query="q1", call_id="1"),
                *_tool_round("local_rag_retrieve", query="q2", call_id="2"),
                *_tool_round("local_rag_retrieve", query="q3", call_id="3"),
                {"role": "human", "content": "new question"},
            ]
        }
        agent = self._build_agent(
            [{"id": "4", "name": "local_rag_retrieve", "args": {"query": "fresh"}}]
        )

        result = agent.call_openai(state)

        self.assertEqual(result["messages"][0].tool_calls[0]["name"], "local_rag_retrieve")

    def test_other_tool_is_allowed_after_search_hits_limit(self) -> None:
        state = {
            "messages": [
                {"role": "human", "content": "question"},
                *_tool_round("local_rag_retrieve", query="q1", call_id="1"),
                *_tool_round("local_rag_retrieve", query="q2", call_id="2"),
                *_tool_round("local_rag_retrieve", query="q3", call_id="3"),
            ]
        }
        agent = self._build_agent([{"id": "4", "name": "web_search", "args": {"query": "next"}}])

        result = agent.call_openai(state)

        self.assertEqual(result["messages"][0].tool_calls[0]["name"], "web_search")

    def test_same_tool_is_blocked_after_consecutive_limit(self) -> None:
        state = {
            "messages": [
                {"role": "human", "content": "question"},
                *_tool_round("local_rag_retrieve", query="q1", call_id="1"),
                *_tool_round("local_rag_retrieve", query="q2", call_id="2"),
                *_tool_round("local_rag_retrieve", query="q3", call_id="3"),
            ]
        }
        agent = self._build_agent(
            [{"id": "4", "name": "local_rag_retrieve", "args": {"query": "repeat"}}]
        )

        result = agent.call_openai(state)
        message = result["messages"][0]

        self.assertEqual(message.tool_calls, [])
        self.assertIn("已达到最大检索轮次", message.content)

    def test_blocked_tool_is_filtered_when_other_tool_is_available(self) -> None:
        state = {
            "messages": [
                {"role": "human", "content": "question"},
                *_tool_round("local_rag_retrieve", query="q1", call_id="1"),
                *_tool_round("local_rag_retrieve", query="q2", call_id="2"),
                *_tool_round("local_rag_retrieve", query="q3", call_id="3"),
            ]
        }
        agent = self._build_agent(
            [
                {"id": "4", "name": "local_rag_retrieve", "args": {"query": "repeat"}},
                {"id": "5", "name": "web_search", "args": {"query": "fallback"}},
            ]
        )

        result = agent.call_openai(state)

        self.assertEqual(
            [tool_call["name"] for tool_call in result["messages"][0].tool_calls],
            ["web_search"],
        )


if __name__ == "__main__":
    unittest.main()
