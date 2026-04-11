from __future__ import annotations

from dataclasses import asdict
from unittest import TestCase
from unittest.mock import ANY
from unittest.mock import patch

from graph_rag_app.cli import main, parse_args
from graph_rag_app.scholar_search import ScholarSearchResponse
from graph_rag_app.web_types import FetchResult, SearchHit
from graph_rag_app.web_types import ScholarHit


class CliWebCommandTests(TestCase):
    def test_parse_args_supports_scholar_search_command(self) -> None:
        args = parse_args(["scholar", "search", "--topic", "graph rag", "--count", "4"])

        self.assertEqual(args.command, "scholar")
        self.assertEqual(args.scholar_command, "search")
        self.assertEqual(args.topic, "graph rag")
        self.assertEqual(args.count, 4)
        self.assertFalse(args.save_md)
        self.assertEqual(args.output_dir, "agent/scholar")

    def test_parse_args_supports_scholar_search_markdown_export_flags(self) -> None:
        args = parse_args(
            [
                "scholar",
                "search",
                "--topic",
                "graph rag",
                "--count",
                "4",
                "--save-md",
                "--output-dir",
                "custom-dir",
            ]
        )

        self.assertTrue(args.save_md)
        self.assertEqual(args.output_dir, "custom-dir")

    def test_parse_args_rejects_invalid_scholar_count(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["scholar", "search", "--topic", "graph rag", "--count", "0"])

    def test_parse_args_supports_web_search_command(self) -> None:
        args = parse_args(["web", "search", "--query", "agent updates", "--top-k", "4"])

        self.assertEqual(args.command, "web")
        self.assertEqual(args.web_command, "search")
        self.assertEqual(args.query, "agent updates")
        self.assertEqual(args.top_k, 4)

    def test_parse_args_supports_web_fetch_command(self) -> None:
        args = parse_args(["web", "fetch", "--url", "https://example.com/page"])

        self.assertEqual(args.command, "web")
        self.assertEqual(args.web_command, "fetch")
        self.assertEqual(args.url, "https://example.com/page")

    def test_main_dispatches_web_search_through_expected_backend_path(self) -> None:
        with patch("graph_rag_app.cli._print_json") as print_json:
            with patch("graph_rag_app.cli.build_app_config") as build_app_config:
                build_app_config.return_value.web.search_provider = "duckduckgo_html"
                build_app_config.return_value.web.search_top_k = 5
                build_app_config.return_value.web.fetch_timeout_seconds = 15
                build_app_config.return_value.web.user_agent = "graph-rag-agent/1.0"
                with patch("graph_rag_app.cli.DuckDuckGoHtmlSearchBackend") as backend_cls:
                    backend = backend_cls.return_value
                    backend.search.return_value = [
                        SearchHit(
                            title="result",
                            url="https://example.com/result",
                            snippet="result snippet",
                            source="duckduckgo_html",
                            rank=1,
                        )
                    ]

                    exit_code = main(["web", "search", "--query", "agent updates", "--top-k", "2"])

        self.assertEqual(exit_code, 0)
        print_json.assert_called_once()
        build_app_config.assert_called_once()
        backend_cls.assert_called_once_with(
            timeout_seconds=15,
            user_agent="graph-rag-agent/1.0",
        )
        backend.search.assert_called_once_with("agent updates", top_k=2)

    def test_main_dispatches_web_fetch_through_expected_fetch_path(self) -> None:
        with patch("graph_rag_app.cli._print_json") as print_json:
            with patch("graph_rag_app.cli.build_app_config") as build_app_config:
                build_app_config.return_value.web.search_provider = "duckduckgo_html"
                build_app_config.return_value.web.search_top_k = 5
                build_app_config.return_value.web.fetch_timeout_seconds = 12
                build_app_config.return_value.web.fetch_max_bytes = 1000
                build_app_config.return_value.web.fetch_max_chars = 500
                build_app_config.return_value.web.user_agent = "graph-rag-agent/1.0"
                with patch(
                    "graph_rag_app.cli.fetch_url",
                    return_value=FetchResult(
                        url="https://example.com/page",
                        final_url="https://example.com/page/final",
                        title="Fetched page",
                        text="Expanded page content.",
                        status_code=200,
                        content_type="text/html",
                        truncated=False,
                    ),
                ) as fetch_url:
                    exit_code = main(["web", "fetch", "--url", "https://example.com/page"])

        self.assertEqual(exit_code, 0)
        print_json.assert_called_once()
        build_app_config.assert_called_once()
        fetch_url.assert_called_once_with(
            "https://example.com/page",
            timeout_seconds=12,
            max_bytes=1000,
            max_chars=500,
            user_agent="graph-rag-agent/1.0",
        )

    def test_main_dispatches_scholar_search_through_expected_service_path(self) -> None:
        response = ScholarSearchResponse(
            topic="graph rag",
            planned_queries=["query one", "query two"],
            result_count=1,
            results=[
                ScholarHit(
                    title="Paper A",
                    url="https://example.com/a",
                    snippet="snippet",
                    publication_summary="2024",
                    year=2024,
                    cited_by_count=12,
                    resources=(),
                    source_query="query one",
                    rank=1,
                )
            ],
        )

        with patch("graph_rag_app.cli._print_json") as print_json:
            with patch("graph_rag_app.cli.build_app_config", return_value="app-config") as build_app_config:
                with patch(
                    "graph_rag_app.cli.run_scholar_search",
                    return_value=response,
                ) as run_scholar_search:
                    exit_code = main(["scholar", "search", "--topic", "graph rag", "--count", "2"])

        self.assertEqual(exit_code, 0)
        build_app_config.assert_called_once()
        run_scholar_search.assert_called_once_with(
            topic="graph rag",
            count=2,
            app_config="app-config",
        )
        print_json.assert_called_once_with(asdict(response))

    def test_main_skips_markdown_export_when_save_flag_not_set(self) -> None:
        response = ScholarSearchResponse(
            topic="graph rag",
            planned_queries=["query one"],
            result_count=0,
            results=[],
        )

        with patch("graph_rag_app.cli._print_json") as print_json:
            with patch("graph_rag_app.cli.build_app_config", return_value="app-config"):
                with patch("graph_rag_app.cli.run_scholar_search", return_value=response):
                    with patch("graph_rag_app.cli.save_scholar_search_markdown") as save_markdown:
                        exit_code = main(["scholar", "search", "--topic", "graph rag"])

        self.assertEqual(exit_code, 0)
        print_json.assert_called_once_with(asdict(response))
        save_markdown.assert_not_called()

    def test_main_saves_markdown_when_requested(self) -> None:
        response = ScholarSearchResponse(
            topic="graph rag",
            planned_queries=["query one", "query two"],
            result_count=1,
            results=[
                ScholarHit(
                    title="Paper A",
                    url="https://example.com/a",
                    snippet="snippet",
                    publication_summary="2024",
                    year=2024,
                    cited_by_count=12,
                    resources=(),
                    source_query="query one",
                    rank=1,
                )
            ],
        )

        with patch("graph_rag_app.cli._print_json") as print_json:
            with patch("graph_rag_app.cli.build_app_config", return_value="app-config"):
                with patch("graph_rag_app.cli.run_scholar_search", return_value=response):
                    with patch(
                        "graph_rag_app.cli.save_scholar_search_markdown",
                        return_value="agent/scholar/2026-04-12-graph-rag.md",
                    ) as save_markdown:
                        with patch("builtins.print") as print_output:
                            exit_code = main(
                                [
                                    "scholar",
                                    "search",
                                    "--topic",
                                    "graph rag",
                                    "--count",
                                    "2",
                                    "--save-md",
                                ]
                            )

        self.assertEqual(exit_code, 0)
        print_json.assert_called_once_with(asdict(response))
        save_markdown.assert_called_once_with(
            response,
            output_dir="agent/scholar",
            count_requested=2,
            now=ANY,
        )
        print_output.assert_called_once_with(
            "Saved Markdown to agent/scholar/2026-04-12-graph-rag.md"
        )

    def test_main_saves_markdown_to_custom_output_dir(self) -> None:
        response = ScholarSearchResponse(
            topic="graph rag",
            planned_queries=["query one"],
            result_count=0,
            results=[],
        )

        with patch("graph_rag_app.cli._print_json"):
            with patch("graph_rag_app.cli.build_app_config", return_value="app-config"):
                with patch("graph_rag_app.cli.run_scholar_search", return_value=response):
                    with patch(
                        "graph_rag_app.cli.save_scholar_search_markdown",
                        return_value="custom-dir/out.md",
                    ) as save_markdown:
                        with patch("builtins.print"):
                            exit_code = main(
                                [
                                    "scholar",
                                    "search",
                                    "--topic",
                                    "graph rag",
                                    "--save-md",
                                    "--output-dir",
                                    "custom-dir",
                                ]
                            )

        self.assertEqual(exit_code, 0)
        save_markdown.assert_called_once_with(
            response,
            output_dir="custom-dir",
            count_requested=5,
            now=ANY,
        )

    def test_main_rejects_unsupported_web_search_provider(self) -> None:
        with patch("graph_rag_app.cli.build_app_config") as build_app_config:
            build_app_config.return_value.web.search_provider = "serpapi"
            build_app_config.return_value.web.fetch_timeout_seconds = 15
            build_app_config.return_value.web.user_agent = "graph-rag-agent/1.0"

            with self.assertRaises(ValueError) as exc_info:
                main(["web", "search", "--query", "agent updates"])

        self.assertEqual(str(exc_info.exception), "Unsupported web search provider: serpapi")
        build_app_config.assert_called_once()

    def test_main_requires_web_subcommand(self) -> None:
        with self.assertRaises(SystemExit):
            main(["web"])
