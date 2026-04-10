# Agent Web Search And Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real internet search and webpage body fetching to the existing LangGraph agent without changing the core `llm -> action -> llm` loop.

**Architecture:** Keep the current agent loop and local RAG tool intact, then add a separate web service layer and two new tools: `web_search` and `web_fetch`. The search backend and fetch/extract logic live in focused modules so the initial provider can be replaced later without rewriting [agent.py](D:/Code/agent_learning/toy/main/graph_rag_app/agent.py).

**Tech Stack:** Python 3.12, `urllib.request`, `html.parser`, `json`, `dataclasses`, `langchain_core` `BaseTool`, `unittest`, `unittest.mock`.

---

### Task 1: Add web configuration and typed web models

**Files:**
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\config.py`
- Create: `D:\Code\agent_learning\toy\main\graph_rag_app\web_types.py`
- Create: `D:\Code\agent_learning\toy\main\tests\test_web_config.py`

- [ ] **Step 1: Write the failing config test**

```python
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from graph_rag_app.config import build_app_config


class WebConfigTests(unittest.TestCase):
    def test_build_app_config_includes_web_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            config = build_app_config("index-dir")

        self.assertTrue(config.web.enabled)
        self.assertEqual(config.web.search_provider, "duckduckgo_html")
        self.assertEqual(config.web.search_top_k, 5)
        self.assertEqual(config.web.fetch_max_chars, 6000)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_web_config -v`

Expected: FAIL with `AttributeError: 'AppConfig' object has no attribute 'web'`.

- [ ] **Step 3: Add the web config dataclasses and wire them into `AppConfig`**

```python
@dataclass(frozen=True)
class WebConfig:
    enabled: bool = True
    search_provider: str = "duckduckgo_html"
    search_top_k: int = 5
    fetch_timeout_seconds: int = 15
    fetch_max_bytes: int = 1_500_000
    fetch_max_chars: int = 6000
    user_agent: str = "graph-rag-agent/1.0"


@dataclass(frozen=True)
class AppConfig:
    kb_path: Path
    model: ModelConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    runtime: RuntimeConfig
    web: WebConfig
```

```python
return AppConfig(
    kb_path=Path(kb_path),
    model=...,
    embedding=...,
    retrieval=...,
    generation=...,
    runtime=...,
    web=WebConfig(
        enabled=parse_bool_env("RAG_WEB_ENABLED", True),
        search_provider=os.getenv("RAG_WEB_SEARCH_PROVIDER", "duckduckgo_html"),
        search_top_k=_parse_int(os.getenv("RAG_WEB_SEARCH_TOP_K"), 5),
        fetch_timeout_seconds=_parse_int(os.getenv("RAG_WEB_FETCH_TIMEOUT_SECONDS"), 15),
        fetch_max_bytes=_parse_int(os.getenv("RAG_WEB_FETCH_MAX_BYTES"), 1_500_000),
        fetch_max_chars=_parse_int(os.getenv("RAG_WEB_FETCH_MAX_CHARS"), 6000),
        user_agent=os.getenv("RAG_WEB_USER_AGENT", "graph-rag-agent/1.0"),
    ),
)
```

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    source: str
    rank: int


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    title: str
    text: str
    status_code: int
    content_type: str
    truncated: bool
```

- [ ] **Step 4: Re-run the config test**

Run: `python -m unittest tests.test_web_config -v`

Expected: PASS

### Task 2: Implement webpage fetching and正文提取

**Files:**
- Create: `D:\Code\agent_learning\toy\main\graph_rag_app\web_fetch.py`
- Create: `D:\Code\agent_learning\toy\main\tests\fixtures\web\article.html`
- Create: `D:\Code\agent_learning\toy\main\tests\test_web_fetch.py`

- [ ] **Step 1: Write the failing fetch tests**

```python
from __future__ import annotations

import unittest
from pathlib import Path

from graph_rag_app.web_fetch import extract_main_text, truncate_text


class WebFetchTests(unittest.TestCase):
    def test_extract_main_text_prefers_article_content(self) -> None:
        html = Path("tests/fixtures/web/article.html").read_text(encoding="utf-8")
        result = extract_main_text(html)
        self.assertIn("Agent systems work best when they can decide", result["text"])
        self.assertNotIn("Sign up for the newsletter", result["text"])
        self.assertEqual(result["title"], "How agents search the web")

    def test_truncate_text_preserves_word_boundaries(self) -> None:
        text = "alpha beta gamma delta epsilon"
        truncated, was_truncated = truncate_text(text, max_chars=18)
        self.assertEqual(truncated, "alpha beta gamma")
        self.assertTrue(was_truncated)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the fetch tests to verify they fail**

Run: `python -m unittest tests.test_web_fetch -v`

Expected: FAIL because `graph_rag_app.web_fetch` does not exist yet.

- [ ] **Step 3: Implement `web_fetch.py` with fetch, extraction, and truncation helpers**

```python
from __future__ import annotations

from html.parser import HTMLParser
from typing import Any
from urllib import request as urllib_request

from .web_types import FetchResult


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact, False
    candidate = compact[:max_chars].rsplit(" ", 1)[0].strip()
    return candidate or compact[:max_chars].strip(), True


def extract_main_text(html: str) -> dict[str, str]:
    parser = ArticleExtractor()
    parser.feed(html)
    title = parser.title.strip()
    body = parser.best_text().strip()
    return {"title": title, "text": body}


def fetch_url(
    url: str,
    *,
    timeout_seconds: int,
    max_bytes: int,
    max_chars: int,
    user_agent: str,
) -> FetchResult:
    request = urllib_request.Request(url, headers={"User-Agent": user_agent})
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            raise ValueError(f"Unsupported content type: {content_type}")
        raw = response.read(max_bytes + 1)
        html = raw[:max_bytes].decode("utf-8", errors="ignore")
        extracted = extract_main_text(html)
        text, truncated = truncate_text(extracted["text"], max_chars=max_chars)
        return FetchResult(
            url=url,
            final_url=response.geturl(),
            title=extracted["title"],
            text=text,
            status_code=getattr(response, "status", 200),
            content_type=content_type,
            truncated=truncated or len(raw) > max_bytes,
        )
```

- [ ] **Step 4: Add the HTML fixture**

```html
<html>
  <head><title>How agents search the web</title></head>
  <body>
    <header>Sign up for the newsletter</header>
    <main>
      <article>
        <p>Agent systems work best when they can decide which sources to inspect.</p>
        <p>Fetching the page body after search makes the final answer more grounded.</p>
      </article>
    </main>
    <footer>Share this article</footer>
  </body>
</html>
```

- [ ] **Step 5: Re-run the fetch tests**

Run: `python -m unittest tests.test_web_fetch -v`

Expected: PASS

### Task 3: Implement internet search backend with normalized search hits

**Files:**
- Create: `D:\Code\agent_learning\toy\main\graph_rag_app\web_search.py`
- Create: `D:\Code\agent_learning\toy\main\tests\fixtures\web\search_results.html`
- Create: `D:\Code\agent_learning\toy\main\tests\test_web_search.py`

- [ ] **Step 1: Write the failing search tests**

```python
from __future__ import annotations

import unittest
from pathlib import Path

from graph_rag_app.web_search import parse_duckduckgo_html


class WebSearchTests(unittest.TestCase):
    def test_parse_duckduckgo_html_returns_ranked_hits(self) -> None:
        html = Path("tests/fixtures/web/search_results.html").read_text(encoding="utf-8")
        hits = parse_duckduckgo_html(html, top_k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].title, "OpenAI agents overview")
        self.assertEqual(hits[0].url, "https://example.com/openai-agents")
        self.assertEqual(hits[0].rank, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the search tests to verify they fail**

Run: `python -m unittest tests.test_web_search -v`

Expected: FAIL because `graph_rag_app.web_search` does not exist yet.

- [ ] **Step 3: Implement the search backend and parser**

```python
from __future__ import annotations

from dataclasses import asdict
from html.parser import HTMLParser
from urllib.parse import quote_plus
from urllib import request as urllib_request

from .web_types import SearchHit


def parse_duckduckgo_html(html: str, top_k: int) -> list[SearchHit]:
    parser = DuckDuckGoResultParser()
    parser.feed(html)
    return parser.hits[:top_k]


class DuckDuckGoHtmlSearchBackend:
    source = "duckduckgo_html"

    def __init__(self, *, timeout_seconds: int, user_agent: str):
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def search(self, query: str, *, top_k: int) -> list[SearchHit]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        request = urllib_request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
            html = response.read().decode("utf-8", errors="ignore")
        return parse_duckduckgo_html(html, top_k=top_k)
```

- [ ] **Step 4: Add the search-result fixture**

```html
<html>
  <body>
    <div class="result">
      <a class="result__a" href="https://example.com/openai-agents">OpenAI agents overview</a>
      <a class="result__snippet">A summary of how OpenAI agents use tools and memory.</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://example.com/agent-search">Agent web search patterns</a>
      <a class="result__snippet">A practical guide to search plus fetch architectures.</a>
    </div>
  </body>
</html>
```

- [ ] **Step 5: Re-run the search tests**

Run: `python -m unittest tests.test_web_search -v`

Expected: PASS

### Task 4: Wrap search and fetch as agent tools

**Files:**
- Create: `D:\Code\agent_learning\toy\main\graph_rag_app\web_tools.py`
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\agent.py`
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\__init__.py`
- Create: `D:\Code\agent_learning\toy\main\tests\test_web_tools.py`
- Create: `D:\Code\agent_learning\toy\main\tests\test_agent_web_integration.py`

- [ ] **Step 1: Write the failing tool tests**

```python
from __future__ import annotations

import json
import unittest

from graph_rag_app.web_tools import WebFetchTool, WebSearchTool
from graph_rag_app.web_types import FetchResult, SearchHit


class WebToolTests(unittest.TestCase):
    def test_web_search_tool_returns_json_payload(self) -> None:
        class FakeBackend:
            def search(self, query: str, *, top_k: int):
                return [SearchHit(title="T1", url="https://example.com", snippet="S1", source="test", rank=1)]

        payload = json.loads(WebSearchTool(FakeBackend())._run("openai agent", top_k=3))
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["results"][0]["url"], "https://example.com")

    def test_web_fetch_tool_returns_json_payload(self) -> None:
        class FakeFetcher:
            def __call__(self, url: str):
                return FetchResult(
                    url=url,
                    final_url=url,
                    title="Doc",
                    text="Body text",
                    status_code=200,
                    content_type="text/html",
                    truncated=False,
                )

        payload = json.loads(WebFetchTool(FakeFetcher())._run("https://example.com"))
        self.assertEqual(payload["title"], "Doc")
        self.assertEqual(payload["text"], "Body text")
```

- [ ] **Step 2: Run the tool tests to verify they fail**

Run: `python -m unittest tests.test_web_tools -v`

Expected: FAIL because `graph_rag_app.web_tools` does not exist yet.

- [ ] **Step 3: Implement the tool wrappers**

```python
class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the public web and return ranked result metadata."
    args_schema: Type[BaseModel] = WebSearchInput

    _backend = PrivateAttr()

    def __init__(self, backend, **kwargs):
        super().__init__(**kwargs)
        self._backend = backend

    def _run(self, query: str, top_k: int = 5) -> str:
        hits = self._backend.search(query, top_k=top_k)
        return json.dumps(
            {
                "query": query,
                "result_count": len(hits),
                "results": [asdict(hit) for hit in hits],
            },
            ensure_ascii=False,
        )
```

```python
class WebFetchTool(BaseTool):
    name: str = "web_fetch"
    description: str = "Fetch a web page and return extracted main content."
    args_schema: Type[BaseModel] = WebFetchInput

    _fetcher = PrivateAttr()

    def __init__(self, fetcher, **kwargs):
        super().__init__(**kwargs)
        self._fetcher = fetcher

    def _run(self, url: str) -> str:
        result = self._fetcher(url)
        return json.dumps(asdict(result), ensure_ascii=False)
```

- [ ] **Step 4: Register the new tools in `build_agent()` and make logging generic**

```python
tools = [local_rag_tool]
if config.web.enabled:
    search_backend = DuckDuckGoHtmlSearchBackend(
        timeout_seconds=config.web.fetch_timeout_seconds,
        user_agent=config.web.user_agent,
    )

    def fetcher(url: str):
        return fetch_url(
            url,
            timeout_seconds=config.web.fetch_timeout_seconds,
            max_bytes=config.web.fetch_max_bytes,
            max_chars=config.web.fetch_max_chars,
            user_agent=config.web.user_agent,
        )

    tools.extend([WebSearchTool(search_backend), WebFetchTool(fetcher)])
```

```python
tool_args = tool_call.get("args", {})
self._append(
    "tool_invocation\n"
    f"name: {tool_call['name']}\n"
    f"args: {json.dumps(tool_args, ensure_ascii=False)}"
)
```

- [ ] **Step 5: Update the agent prompt so the model knows when to search and fetch**

```python
PROMPT = """\
You answer questions using the local knowledge base first.
If the local knowledge base lacks evidence or the user asks for recent public information, use web_search.
Do not answer from web_search snippets alone when the claim matters; fetch the most relevant page with web_fetch first.
When you use the web, cite the page title or URL in the answer.
"""
```

- [ ] **Step 6: Re-run the tool tests**

Run: `python -m unittest tests.test_web_tools -v`

Expected: PASS

### Task 5: Add debug CLI entrypoints and full verification

**Files:**
- Modify: `D:\Code\agent_learning\toy\main\graph_rag_app\cli.py`
- Create: `D:\Code\agent_learning\toy\main\tests\test_cli_web_commands.py`
- Test: `D:\Code\agent_learning\toy\main\tests\test_agent_web_integration.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
from __future__ import annotations

import unittest
from unittest.mock import patch

from graph_rag_app.cli import parse_args


class CliWebCommandTests(unittest.TestCase):
    def test_parse_args_supports_web_search(self) -> None:
        args = parse_args(["web", "search", "--query", "openai agent"])
        self.assertEqual(args.command, "web")
        self.assertEqual(args.web_command, "search")
        self.assertEqual(args.query, "openai agent")
```

- [ ] **Step 2: Run the CLI web tests to verify they fail**

Run: `python -m unittest tests.test_cli_web_commands -v`

Expected: FAIL because the `web` command tree does not exist yet.

- [ ] **Step 3: Add `web search` and `web fetch` subcommands for debugging**

```python
web_parser = subparsers.add_parser("web", help="Debug web search and fetch helpers.")
web_subparsers = web_parser.add_subparsers(dest="web_command")

web_search = web_subparsers.add_parser("search", help="Run a raw web search.")
web_search.add_argument("--query", required=True)
web_search.add_argument("--top-k", type=int, default=5)

web_fetch = web_subparsers.add_parser("fetch", help="Fetch and extract a web page.")
web_fetch.add_argument("--url", required=True)
```

- [ ] **Step 4: Run the full verification suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS for the old chunking and agent tests plus the new web config, fetch, search, tool, and CLI tests.

Run: `python graph_rag.py web search --query "openai agent"`

Expected: JSON array of normalized search hits.

Run: `python graph_rag.py ask --index-dir .\agent --question "What changed recently about OpenAI agents?"`

Expected: the answer uses the local index first and can fall back to `web_search` plus `web_fetch` when the model chooses them.

- [ ] **Step 5: Keep caching, reranking, and richer citation formatting out of this first implementation**

```text
Do not add HTTP cache, multi-provider ranking, or HTML readability libraries in this pass.
If extraction quality is insufficient after MVP verification, plan a second pass focused only on fetch quality.
```
