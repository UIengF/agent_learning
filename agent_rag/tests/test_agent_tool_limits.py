from __future__ import annotations

import unittest

from graph_rag_app.agent import Agent, PROMPT
from graph_rag_app.evidence_cache import EvidenceCache
from graph_rag_app.session_summary import SessionSummary
from graph_rag_app.task_state import TaskState
from graph_rag_app.user_memory import UserMemory


class FakeMessage:
    def __init__(self, content: str = "", tool_calls: list[dict] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.type = "assistant"


class FakeModel:
    def __init__(self, message: FakeMessage):
        self._message = message
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001
        self.calls += 1
        return self._message


class SequenceModel:
    def __init__(self, messages: list[FakeMessage]):
        self._messages = list(messages)
        self.calls = 0

    def invoke(self, messages):  # noqa: ANN001
        self.calls += 1
        if not self._messages:
            raise AssertionError("No more fake messages configured")
        return self._messages.pop(0)


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

    def test_other_tool_is_allowed_before_total_limit_is_exhausted(self) -> None:
        state = {
            "messages": [
                {"role": "human", "content": "question"},
                *_tool_round("local_rag_retrieve", query="q1", call_id="1"),
                *_tool_round("local_rag_retrieve", query="q2", call_id="2"),
            ]
        }
        agent = self._build_agent([{"id": "4", "name": "web_search", "args": {"query": "next"}}])

        result = agent.call_openai(state)

        self.assertEqual(result["messages"][0].tool_calls[0]["name"], "web_search")

    def test_any_tool_is_blocked_after_total_limit_is_reached(self) -> None:
        state = {
            "messages": [
                {"role": "human", "content": "question"},
                *_tool_round("local_rag_retrieve", query="q1", call_id="1"),
                *_tool_round("web_search", query="q2", call_id="2"),
                *_tool_round("local_rag_retrieve", query="q3", call_id="3"),
            ]
        }
        agent = self._build_agent([{"id": "4", "name": "web_fetch", "args": {"url": "https://example.com"}}])
        agent.base_model = FakeModel(FakeMessage(content="final answer from available evidence"))

        result = agent.call_openai(state)
        message = result["messages"][0]

        self.assertEqual(message.tool_calls, [])
        self.assertEqual(message.content, "final answer from available evidence")
        self.assertEqual(agent.base_model.calls, 1)

    def test_total_limit_blocks_all_requested_tools_after_alternating_history(self) -> None:
        state = {
            "messages": [
                {"role": "human", "content": "question"},
                *_tool_round("local_rag_retrieve", query="q1", call_id="1"),
                *_tool_round("web_search", query="q2", call_id="2"),
                *_tool_round("web_fetch", query="https://example.com/3", call_id="3"),
            ]
        }
        agent = self._build_agent(
            [
                {"id": "4", "name": "local_rag_retrieve", "args": {"query": "repeat"}},
                {"id": "5", "name": "web_search", "args": {"query": "fallback"}},
            ]
        )
        agent.base_model = FakeModel(FakeMessage(content="final answer after alternating tools"))

        result = agent.call_openai(state)
        message = result["messages"][0]

        self.assertEqual(message.tool_calls, [])
        self.assertEqual(message.content, "final answer after alternating tools")

    def test_tool_limit_final_answer_strips_new_tool_calls(self) -> None:
        state = {
            "messages": [
                {"role": "human", "content": "question"},
                *_tool_round("local_rag_retrieve", query="q1", call_id="1"),
                *_tool_round("web_search", query="q2", call_id="2"),
                *_tool_round("local_rag_retrieve", query="q3", call_id="3"),
            ]
        }
        agent = self._build_agent([{"id": "4", "name": "web_fetch", "args": {"url": "https://example.com"}}])
        agent.base_model = FakeModel(
            FakeMessage(
                content="final answer even if the model tried another tool",
                tool_calls=[{"id": "5", "name": "web_search", "args": {"query": "more"}}],
            )
        )

        result = agent.call_openai(state)
        message = result["messages"][0]

        self.assertEqual(message.tool_calls, [])
        self.assertEqual(message.content, "final answer even if the model tried another tool")

    def test_build_llm_messages_compresses_older_live_turns_and_keeps_latest_turn_raw(self) -> None:
        agent = Agent(system="system")
        state = {
            "messages": [
                {"role": "human", "content": "question 1"},
                {"role": "assistant", "content": "answer 1"},
                {"role": "human", "content": "question 2"},
                {"role": "assistant", "content": "answer 2"},
                {"role": "human", "content": "question 3"},
                {"role": "assistant", "content": "answer 3"},
                {"role": "human", "content": "question 4"},
                {"role": "assistant", "content": "draft answer 4"},
                {"role": "tool", "name": "local_rag_retrieve", "content": '{"result_count": 1}'},
            ]
        }

        messages = agent.build_llm_messages(state)
        contents = [agent._message_content(message) for message in messages]

        self.assertEqual(contents[0], "system")
        self.assertIn("Session summary of earlier messages:", contents[1])
        self.assertIn("question 1", contents[1])
        self.assertIn("answer 1", contents[1])
        self.assertTrue(contents[2].startswith("Question frame:"))
        self.assertIn("question: question 4", contents[2])
        self.assertIn("Compressed live messages:", contents[3])
        self.assertIn("question 2", contents[3])
        self.assertIn("answer 2", contents[3])
        self.assertIn("question 3", contents[3])
        self.assertIn("answer 3", contents[3])
        self.assertIn("question 4", contents)
        self.assertIn("draft answer 4", contents)
        self.assertIn('{"result_count": 1}', contents)
        self.assertIn("You just received tool results.", contents[-1])

    def test_build_llm_messages_includes_session_summary_for_trimmed_messages(self) -> None:
        agent = Agent(system="system", recent_full_turns=1)
        state = {
            "messages": [
                {"role": "human", "content": "older question"},
                {"role": "assistant", "content": "older assistant reasoning"},
                {"role": "tool", "name": "local_rag_retrieve", "content": '{"result_count": 2}'},
                {"role": "human", "content": "current question"},
                {"role": "assistant", "content": "latest assistant reasoning"},
                {"role": "tool", "name": "web_search", "content": '{"result_count": 5}'},
            ]
        }

        messages = agent.build_llm_messages(state)
        contents = [agent._message_content(message) for message in messages]

        self.assertEqual(contents[0], "system")
        self.assertIn("Session summary of earlier messages:", contents[1])
        self.assertIn("older assistant reasoning", contents[1])
        self.assertIn('{"result_count": 2}', contents[1])
        self.assertTrue(contents[2].startswith("Question frame:"))
        self.assertIn("question: current question", contents[2])
        self.assertEqual(contents[3], "current question")
        self.assertIn("latest assistant reasoning", contents)
        self.assertIn('{"result_count": 5}', contents)

    def test_build_session_summary_returns_structured_object(self) -> None:
        agent = Agent(system="system", max_recent_messages=3)
        summary = agent._build_session_summary(
            [
                {"role": "human", "content": "compare openai and gemini agents"},
                {"role": "human", "content": "use local evidence only"},
                {"role": "assistant", "content": "local kb lacks gemini details"},
                {"role": "tool", "name": "local_rag_retrieve", "content": '{"result_count": 2}'},
                {"role": "human", "content": "what official docs should we fetch?"},
            ]
        )

        self.assertIsInstance(summary, SessionSummary)
        assert summary is not None
        self.assertEqual(summary.previous_topics, ("compare openai and gemini agents",))
        self.assertEqual(summary.persistent_user_constraints, ("use local evidence only",))
        self.assertEqual(summary.confirmed_facts, ("local kb lacks gemini details",))
        self.assertEqual(summary.open_questions, ("what official docs should we fetch?",))
        self.assertEqual(
            summary.tool_history,
            ('local_rag_retrieve: {"result_count": 2}',),
        )

    def test_build_task_state_returns_structured_object(self) -> None:
        agent = Agent(system="system")
        task_state = agent._build_task_state(
            [
                {
                    "role": "human",
                    "content": "What are the differences between OpenAI and Gemini agents?",
                },
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": '{"query":"OpenAI vs Gemini","result_count":0,"reason":"insufficient_evidence","results":[]}',
                },
            ]
        )

        self.assertIsInstance(task_state, TaskState)
        assert task_state is not None
        self.assertEqual(
            task_state.question,
            "What are the differences between OpenAI and Gemini agents?",
        )
        self.assertEqual(task_state.entities, ("OpenAI", "Gemini"))
        self.assertEqual(task_state.next_action, "web_search")
        self.assertEqual(task_state.evidence_sufficiency, "low")
        self.assertIn("insufficient_evidence", task_state.missing_information)

    def test_build_llm_messages_includes_task_state(self) -> None:
        agent = Agent(system="system", recent_full_turns=1)
        state = {
            "messages": [
                {
                    "role": "human",
                    "content": "What are the differences between OpenAI and Gemini agents?",
                },
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": '{"query":"OpenAI vs Gemini","result_count":0,"reason":"insufficient_evidence","results":[]}',
                },
            ]
        }

        messages = agent.build_llm_messages(state)
        contents = [agent._message_content(message) for message in messages]

        self.assertEqual(contents[0], "system")
        task_state_content = next(content for content in contents if content.startswith("Task state:"))
        self.assertIn("question: What are the differences between OpenAI and Gemini agents?", task_state_content)
        self.assertIn("next_action: web_search", task_state_content)

    def test_build_evidence_cache_returns_structured_object(self) -> None:
        agent = Agent(system="system")
        evidence_cache = agent._build_evidence_cache(
            [
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": '{"query":"OpenAI vs Gemini","result_count":1,"results":[]}',
                },
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": '{"query":"OpenAI Agents SDK vs Google ADK","result_count":5,"results":[]}',
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": '{"url":"https://example.com/page","title":"Example","text":"content"}',
                },
            ]
        )

        self.assertIsInstance(evidence_cache, EvidenceCache)
        assert evidence_cache is not None
        self.assertIn("OpenAI vs Gemini", evidence_cache.local_results_by_query)
        self.assertIn("OpenAI Agents SDK vs Google ADK", evidence_cache.web_results_by_query)
        self.assertIn("https://example.com/page", evidence_cache.fetched_pages_by_url)

    def test_take_action_reuses_cached_tool_result(self) -> None:
        class CountingTool:
            name = "web_fetch"

            def __init__(self) -> None:
                self.calls = 0

            def invoke(self, args):
                self.calls += 1
                return '{"url":"https://example.com/page","title":"Fresh","text":"fresh"}'

        tool = CountingTool()
        agent = Agent(tools=[tool])
        state = {
            "messages": [
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": '{"url":"https://example.com/page","title":"Cached","text":"cached"}',
                },
                type(
                    "Carrier",
                    (),
                    {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "name": "web_fetch",
                                "args": {"url": "https://example.com/page"},
                            }
                        ]
                    },
                )(),
            ]
        }

        result = agent.take_action(state)

        self.assertEqual(tool.calls, 0)
        self.assertEqual(len(result["messages"]), 1)
        self.assertIn('"title":"Cached"', result["messages"][0].content)

    def test_build_llm_messages_includes_user_memory(self) -> None:
        agent = Agent(
            system="system",
            user_memory=UserMemory(
                preferred_language="zh",
                answer_style="concise",
                stable_constraints=("Use local evidence first and do not invent facts.",),
            ),
        )
        state = {"messages": [{"role": "human", "content": "current question"}]}

        messages = agent.build_llm_messages(state)
        contents = [agent._message_content(message) for message in messages]

        self.assertEqual(contents[0], "system")
        user_memory_content = next(content for content in contents if content.startswith("User memory:"))
        self.assertIn("preferred_language: zh", user_memory_content)
        self.assertIn("answer_style: concise", user_memory_content)

    def test_build_context_result_tracks_context_layers(self) -> None:
        agent = Agent(
            system="system",
            recent_full_turns=1,
            user_memory=UserMemory(preferred_language="zh"),
        )
        state = {
            "messages": [
                {"role": "human", "content": "older question"},
                {"role": "assistant", "content": "older answer"},
                {
                    "role": "human",
                    "content": "What are the differences between OpenAI and Gemini agents?",
                },
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": '{"query":"OpenAI vs Gemini","result_count":0,"reason":"insufficient_evidence","results":[]}',
                },
            ]
        }

        result = agent.build_context_result(state)

        self.assertEqual(
            tuple(layer.name for layer in result.layers),
            (
                "system_prompt",
                "user_memory",
                "session_summary",
                "question_frame",
                "live_messages",
                "evidence_cache",
                "task_state",
                "reflection_prompt",
            ),
        )

    def test_call_openai_logs_question_frame_on_first_round(self) -> None:
        logged_entries: list[str] = []

        def append_log(_path, text: str) -> None:  # noqa: ANN001
            logged_entries.append(text)

        agent = Agent(
            system="system",
            append_log=append_log,
            ensure_log_file=lambda _path: None,
        )
        agent.model = FakeModel(FakeMessage(content="final answer", tool_calls=[]))
        state = {
            "messages": [
                {"role": "human", "content": "openai和gemini在agent实现中有哪些异同点"},
            ]
        }

        agent.call_openai(state)

        combined = "\n".join(logged_entries)
        self.assertIn("Question frame", combined)
        self.assertIn('"task_intent": "compare"', combined)
        self.assertIn('"target_entities": ["OpenAI", "Gemini"]', combined)

    def test_call_openai_rewrites_web_fetch_to_official_result_when_available(self) -> None:
        agent = self._build_agent(
            [{"id": "fetch-1", "name": "web_fetch", "args": {"url": "https://bbc.com/story"}}]
        )
        state = {
            "messages": [
                {"role": "human", "content": "What changed recently about OpenAI?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "search-1", "name": "web_search", "args": {"query": "recent OpenAI news"}}],
                },
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"recent OpenAI news","result_count":2,"results":['
                        '{"title":"OpenAI News","url":"https://openai.com/news","snippet":"Official updates",'
                        '"source":"duckduckgo_html","rank":1,"is_official":true},'
                        '{"title":"BBC story","url":"https://bbc.com/story","snippet":"Third-party coverage",'
                        '"source":"duckduckgo_html","rank":2,"is_official":false}]}'
                    ),
                },
            ]
        }

        result = agent.call_openai(state)

        self.assertEqual(
            result["messages"][0].tool_calls[0]["args"]["url"],
            "https://openai.com/news",
        )

    def test_call_openai_keeps_official_web_fetch_when_model_already_chose_it(self) -> None:
        agent = self._build_agent(
            [{"id": "fetch-1", "name": "web_fetch", "args": {"url": "https://openai.com/news"}}]
        )
        state = {
            "messages": [
                {"role": "human", "content": "What changed recently about OpenAI?"},
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"recent OpenAI news","result_count":1,"results":['
                        '{"title":"OpenAI News","url":"https://openai.com/news","snippet":"Official updates",'
                        '"source":"duckduckgo_html","rank":1,"is_official":true}]}'
                    ),
                },
            ]
        }

        result = agent.call_openai(state)

        self.assertEqual(
            result["messages"][0].tool_calls[0]["args"]["url"],
            "https://openai.com/news",
        )

    def test_call_openai_keeps_web_fetch_when_no_official_result_exists(self) -> None:
        agent = self._build_agent(
            [
                {
                    "id": "fetch-1",
                    "name": "web_fetch",
                    "args": {"url": "https://techcrunch.com/story"},
                }
            ]
        )
        state = {
            "messages": [
                {"role": "human", "content": "What changed recently about OpenAI?"},
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"recent OpenAI news","result_count":1,"results":['
                        '{"title":"TechCrunch story","url":"https://techcrunch.com/story",'
                        '"snippet":"Third-party coverage","source":"duckduckgo_html","rank":1,'
                        '"is_official":false}]}'
                    ),
                },
            ]
        }

        result = agent.call_openai(state)

        self.assertEqual(
            result["messages"][0].tool_calls[0]["args"]["url"],
            "https://techcrunch.com/story",
        )

    def test_call_openai_avoids_rewriting_to_same_official_url_after_fetch_failure(self) -> None:
        agent = self._build_agent(
            [{"id": "fetch-1", "name": "web_fetch", "args": {"url": "https://pcmag.com/openai"}}]
        )
        state = {
            "messages": [
                {"role": "human", "content": "What changed recently about OpenAI?"},
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"recent OpenAI news","result_count":2,"results":['
                        '{"title":"OpenAI News","url":"https://openai.com/news","snippet":"Official updates",'
                        '"source":"duckduckgo_html","rank":1,"is_official":true},'
                        '{"title":"Funding update","url":"https://openai.com/index/funding-update",'
                        '"snippet":"Official article","source":"duckduckgo_html","rank":2,"is_official":true}]}'
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": (
                        '{"error":"tool_execution_failed","tool_name":"web_fetch","tool_args":'
                        '{"url":"https://openai.com/news"},"error_type":"HTTPError","message":"HTTP Error 403: Forbidden"}'
                    ),
                },
            ]
        }

        result = agent.call_openai(state)

        self.assertEqual(
            result["messages"][0].tool_calls[0]["args"]["url"],
            "https://openai.com/index/funding-update",
        )

    def test_call_openai_keeps_non_official_fetch_when_only_official_url_already_failed(self) -> None:
        agent = self._build_agent(
            [{"id": "fetch-1", "name": "web_fetch", "args": {"url": "https://pcmag.com/openai"}}]
        )
        state = {
            "messages": [
                {"role": "human", "content": "What changed recently about OpenAI?"},
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"recent OpenAI news","result_count":1,"results":['
                        '{"title":"OpenAI News","url":"https://openai.com/news","snippet":"Official updates",'
                        '"source":"duckduckgo_html","rank":1,"is_official":true}]}'
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": (
                        '{"error":"tool_execution_failed","tool_name":"web_fetch","tool_args":'
                        '{"url":"https://openai.com/news"},"error_type":"HTTPError","message":"HTTP Error 403: Forbidden"}'
                    ),
                },
            ]
        }

        result = agent.call_openai(state)

        self.assertEqual(
            result["messages"][0].tool_calls[0]["args"]["url"],
            "https://pcmag.com/openai",
        )

    def test_call_openai_logs_context_metrics(self) -> None:
        logged_entries: list[str] = []

        def append_log(_path, text: str) -> None:  # noqa: ANN001
            logged_entries.append(text)

        agent = Agent(
            system="system",
            append_log=append_log,
            ensure_log_file=lambda _path: None,
        )
        agent.model = FakeModel(FakeMessage(content="final answer", tool_calls=[]))
        state = {"messages": [{"role": "human", "content": "current question"}]}

        agent.call_openai(state)

        combined = "\n".join(logged_entries)
        self.assertIn("Context metrics:", combined)
        self.assertIn("estimated_total_tokens:", combined)

    def test_call_openai_logs_structured_reflection_result_after_tool_output(self) -> None:
        logged_entries: list[str] = []

        def append_log(_path, text: str) -> None:  # noqa: ANN001
            logged_entries.append(text)

        agent = Agent(
            system="system",
            append_log=append_log,
            ensure_log_file=lambda _path: None,
        )
        agent.model = FakeModel(
            FakeMessage(
                content="need page body",
                tool_calls=[{"id": "fetch-1", "name": "web_fetch", "args": {"url": "https://openai.com/news"}}],
            )
        )
        state = {
            "messages": [
                {"role": "human", "content": "What changed recently about OpenAI agents?"},
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"recent OpenAI agents news","result_count":1,"results":['
                        '{"title":"OpenAI News","url":"https://openai.com/news","snippet":"Official updates",'
                        '"source":"duckduckgo_html","rank":1,"is_official":true}]}'
                    ),
                },
            ]
        }

        agent.call_openai(state)

        combined = "\n".join(logged_entries)
        self.assertIn("Reflection result", combined)
        self.assertIn('"llm_decision": "tool_use"', combined)
        self.assertIn('"recommended_next_action": "web_fetch"', combined)
        self.assertIn('"latest_tool_name": "web_search"', combined)
        self.assertIn('"question": "What changed recently about OpenAI agents?"', combined)

    def test_system_prompt_prefers_detailed_answers_by_default(self) -> None:
        self.assertIn("prefer a detailed, structured response", PROMPT)
        self.assertIn("multi-paragraph answer", PROMPT)


if __name__ == "__main__":
    unittest.main()

