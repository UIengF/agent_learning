from __future__ import annotations

import json
import re
from urllib.parse import parse_qs, unquote, urlparse
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from bs4 import BeautifulSoup


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(slots=True)
class WebPage:
    title: str
    url: str
    content: str


class SearchTool(Protocol):
    async def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        ...


class WebReader(Protocol):
    async def read(self, url: str) -> WebPage:
        ...


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def normalize_result_url(url: str) -> str:
    normalized = (url or "").strip()
    if not normalized:
        return ""
    if normalized.startswith("//"):
        normalized = f"https:{normalized}"
    parsed = urlparse(normalized)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        normalized = unquote(target) if target else normalized
    elif "bing.com" in parsed.netloc and parsed.path.startswith("/ck/a"):
        target = parse_qs(parsed.query).get("u", [""])[0]
        normalized = unquote(target) if target else normalized
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return parsed.geturl()


def is_useful_result_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    lowered = f"{parsed.netloc}{parsed.path}".lower()
    blocked_terms = ("login", "signin", "signup", "/search", "javascript:", "accounts.google.com")
    return not any(term in lowered for term in blocked_terms)


def normalize_search_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    normalized: list[SearchResult] = []
    seen_urls: set[str] = set()
    seen_domains: set[str] = set()
    for result in results:
        clean_url = normalize_result_url(result.url)
        if not clean_url or not is_useful_result_url(clean_url):
            continue
        domain = urlparse(clean_url).netloc.lower()
        if clean_url in seen_urls:
            continue
        if domain in seen_domains and len(normalized) >= limit:
            continue
        seen_urls.add(clean_url)
        seen_domains.add(domain)
        normalized.append(
            SearchResult(
                title=result.title.strip(),
                url=clean_url,
                snippet=result.snippet.strip(),
            )
        )
        if len(normalized) >= limit:
            break
    return normalized


def build_query_variants(query: str) -> list[str]:
    base = " ".join(part for part in re.split(r"\s+", query.strip()) if part)
    if not base:
        return []
    variants = [base]
    tokens = [token for token in re.split(r"[^a-z0-9+-]+", base.lower()) if token]
    important = [
        token
        for token in tokens
        if token in {"python", "rust", "cli", "startup", "packaging", "distribution", "dependency", "error", "handling", "onboarding", "maintenance"}
    ]
    if important:
        variants.append(" ".join(dict.fromkeys(important)))
    compact = [token for token in tokens if token not in {"benchmark", "cross", "platform", "windows", "macos", "linux", "survey", "reproducibility", "single-binary", "file", "size"}]
    if compact:
        variants.append(" ".join(dict.fromkeys(compact[:6])))
    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        cleaned = " ".join(variant.split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


class DeterministicLLMBackend:
    async def plan(self, goal: str, round_index: int, artifacts: list[Any]) -> dict[str, Any]:
        del artifacts
        topics = self._extract_topics(goal) if round_index == 0 else [f"follow-up {round_index}"]
        tasks = [
            {
                "title": f"{topic.title()} evidence",
                "query": f"{topic} cli strengths",
                "description": f"Collect evidence about {topic}.",
            }
            for topic in topics
        ]
        return {
            "tasks": tasks,
            "notes": f"Generated {len(tasks)} bounded research tasks.",
            "token_usage": 12 + len(tasks),
            "cost_usd": 0.0,
        }

    async def summarize_research(self, goal: str, task: Any, pages: list[WebPage]) -> dict[str, Any]:
        del goal
        citations = [{"title": page.title, "url": page.url, "snippet": page.content[:140]} for page in pages]
        bullet_points = [f"- {page.title}: {page.content[:90]}" for page in pages]
        return {
            "summary": "\n".join(bullet_points) or f"No evidence collected for {task.title}.",
            "citations": citations,
            "token_usage": 20 + len(citations) * 3,
            "cost_usd": 0.0,
        }

    async def synthesize(self, goal: str, artifacts: list[Any]) -> dict[str, Any]:
        findings = []
        citation_lines = []
        seen_urls: set[str] = set()
        counter = 1
        for artifact in artifacts:
            label = artifact.citation_label or f"[{counter}]"
            finding = artifact.content.splitlines()[0] if artifact.content else artifact.title
            findings.append(f"- {artifact.title}: {finding[:120]}")
            for citation in artifact.citations:
                url = citation["url"]
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                citation_lines.append(f"{len(citation_lines) + 1}. {url}")
                if len(citation_lines) == 3:
                    break
            if len(citation_lines) == 3:
                break
            counter += 1
        markdown = (
            "# Research Report\n\n"
            f"## Goal\n{goal}\n\n"
            "## Findings\n"
            f"{chr(10).join(findings) if findings else '- No evidence gathered.'}\n\n"
            "## Sources\n"
            f"{chr(10).join(citation_lines) if citation_lines else '1. No sources recorded.'}\n\n"
            "## Conclusion\n"
            "Choose the faster-to-iterate stack for internal automation and the safer, faster runtime for distributed or security-sensitive CLI tools.\n"
        )
        return {"markdown": markdown, "token_usage": 18 + len(artifacts) * 2, "cost_usd": 0.0}

    async def single_agent_report(self, goal: str, pages: list[WebPage]) -> dict[str, Any]:
        findings = []
        sources = []
        for page in pages[:3]:
            findings.append(f"- {page.title}: {page.content[:90]}")
            sources.append(f"{len(sources) + 1}. {page.url}")
        markdown = (
            "# Research Report\n\n"
            f"## Goal\n{goal}\n\n"
            "## Findings\n"
            f"{chr(10).join(findings) if findings else '- No evidence gathered.'}\n\n"
            "## Sources\n"
            f"{chr(10).join(sources) if sources else '1. No sources recorded.'}\n\n"
            "## Conclusion\n"
            "Use the stack that best fits delivery speed, portability, and runtime safety for the target CLI workload.\n"
        )
        return {"markdown": markdown, "token_usage": 16 + len(pages) * 2, "cost_usd": 0.0}

    async def critique(self, goal: str, draft_markdown: str, artifacts: list[Any], round_index: int) -> dict[str, Any]:
        del goal
        urls = re.findall(r"https?://\S+", draft_markdown)
        approved = (
            "## Findings" in draft_markdown
            and "## Sources" in draft_markdown
            and "## Conclusion" in draft_markdown
            and len(urls) >= 3
            and len(artifacts) >= 2
        )
        feedback = (
            "Coverage is sufficient."
            if approved
            else "Minimum quality bar not met: include findings, at least three sources, and a concise conclusion."
        )
        follow_up_tasks = []
        if not approved:
            follow_up_tasks.append(
                {
                    "title": f"Gap analysis {round_index + 1}",
                    "query": "follow-up evidence and final recommendation",
                    "description": "Close missing evidence gaps and finalize the conclusion.",
                }
            )
        return {
            "approved": approved,
            "feedback": feedback,
            "follow_up_tasks": follow_up_tasks,
            "token_usage": 10,
            "cost_usd": 0.0,
        }

    def _extract_topics(self, goal: str) -> list[str]:
        normalized = goal.lower()
        compare_match = re.search(r"compare\s+(.+?)\s+and\s+(.+?)(?:\s+for|\s*$)", normalized)
        if compare_match:
            return [compare_match.group(1).strip(), compare_match.group(2).strip()]
        parts = [part.strip() for part in re.split(r"\band\b|,", normalized) if part.strip()]
        if len(parts) >= 2:
            return parts[:2]
        return [normalized[:40].strip() or "topic"]


class DeterministicSearchTool:
    async def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        topics = [part for part in query.split() if len(part) > 2][:limit]
        if not topics:
            topics = ["topic"]
        return [
            SearchResult(
                title=f"{topic.title()} source",
                url=f"https://example.com/{topic.lower()}",
                snippet=f"Synthetic search hit for {topic}.",
            )
            for topic in topics[:limit]
        ]


class DeterministicWebReader:
    async def read(self, url: str) -> WebPage:
        slug = url.rsplit("/", maxsplit=1)[-1] or "source"
        content = f"{slug.title()} evidence about local automation, ergonomics, and deployment."
        return WebPage(title=f"{slug.title()} page", url=url, content=content)


class DuckDuckGoSearchTool:
    def __init__(self, *, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.user_agent = user_agent

    async def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        ) as client:
            response = await client.get("https://html.duckduckgo.com/html/", params={"q": query})
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SearchResult] = []
        for result in soup.select(".result")[:limit]:
            link = result.select_one(".result__a")
            snippet = result.select_one(".result__snippet")
            if not link or not link.get("href"):
                continue
            results.append(
                SearchResult(
                    title=link.get_text(" ", strip=True),
                    url=link["href"],
                    snippet=snippet.get_text(" ", strip=True) if snippet else "",
                )
            )
        return normalize_search_results(results, limit)


class BingSearchTool:
    def __init__(self, *, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.user_agent = user_agent

    async def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        async with httpx.AsyncClient(
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        ) as client:
            response = await client.get("https://www.bing.com/search", params={"q": query})
            response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SearchResult] = []
        for result in soup.select("li.b_algo")[: limit * 2]:
            link = result.select_one("h2 a")
            snippet = result.select_one(".b_caption p")
            if not link or not link.get("href"):
                continue
            results.append(
                SearchResult(
                    title=link.get_text(" ", strip=True),
                    url=link["href"],
                    snippet=snippet.get_text(" ", strip=True) if snippet else "",
                )
            )
        return normalize_search_results(results, limit)


class FallbackSearchTool:
    def __init__(self, *tools: SearchTool) -> None:
        self.tools = list(tools)

    async def search(self, query: str, limit: int = 3) -> list[SearchResult]:
        merged: list[SearchResult] = []
        for variant in build_query_variants(query):
            for tool in self.tools:
                try:
                    results = await tool.search(variant, limit=limit)
                except Exception:
                    continue
                merged.extend(results)
                normalized = normalize_search_results(merged, limit)
                if len(normalized) >= limit:
                    return normalized
        return normalize_search_results(merged, limit)


class HttpWebReader:
    def __init__(
        self,
        *,
        retries: int = 2,
        timeout: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
        min_content_length: int = 160,
    ) -> None:
        self.retries = retries
        self.timeout = timeout
        self.user_agent = user_agent
        self.min_content_length = min_content_length

    async def read(self, url: str) -> WebPage:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout,
                    follow_redirects=True,
                    headers={"User-Agent": self.user_agent},
                ) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.get_text(" ", strip=True) if soup.title else url
                text = soup.get_text("\n", strip=True)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                content = "\n".join(lines)
                content = re.sub(r"\n{3,}", "\n\n", content)[:4000]
                if len(content) < self.min_content_length:
                    raise ValueError(f"Thin content extracted from {url}")
                return WebPage(title=title, url=url, content=content)
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
        raise last_error if last_error else RuntimeError(f"Failed to read {url}")


class OpenAICompatibleLLMBackend:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        plan_max_tokens: int = 500,
        research_max_tokens: int = 700,
        synthesize_max_tokens: int = 1200,
        critique_max_tokens: int = 700,
        single_agent_max_tokens: int = 1200,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.plan_max_tokens = plan_max_tokens
        self.research_max_tokens = research_max_tokens
        self.synthesize_max_tokens = synthesize_max_tokens
        self.critique_max_tokens = critique_max_tokens
        self.single_agent_max_tokens = single_agent_max_tokens

    async def plan(self, goal: str, round_index: int, artifacts: list[Any]) -> dict[str, Any]:
        prompt = (
            "Return JSON with keys tasks, notes. "
            "Create at most 3 targeted research tasks. Each task must include title, query, description. "
            "Prefer short, evidence-seeking tasks and avoid duplicate coverage. "
            f"Goal: {goal}\nRound: {round_index}\nExisting artifact count: {len(artifacts)}"
        )
        return await self._chat_json(prompt, max_tokens=self.plan_max_tokens)

    async def summarize_research(self, goal: str, task: Any, pages: list[WebPage]) -> dict[str, Any]:
        compact_pages = [{"title": page.title, "url": page.url, "content": page.content[:800]} for page in pages]
        prompt = (
            "Return JSON with keys summary, citations. citations must be a list of {title,url,snippet}. "
            "Summary must be 3 short bullet lines max and grounded only in the provided pages. "
            f"Goal: {goal}\nTask: {task.title}\nPages: {json.dumps(compact_pages, ensure_ascii=False)}"
        )
        return await self._chat_json(prompt, max_tokens=self.research_max_tokens)

    async def synthesize(self, goal: str, artifacts: list[Any]) -> dict[str, Any]:
        payload = [
            {"title": artifact.title, "content": artifact.content[:400], "citations": artifact.citations[:3]}
            for artifact in artifacts
        ]
        prompt = (
            "Write a concise markdown research report. "
            "Output exactly these sections in order: Goal, Findings, Sources, Conclusion. "
            "Use plain markdown, not tables. "
            "Findings must be 3 bullet points max, each under 28 words. "
            "Sources must be exactly 3 numbered markdown links, one per line. "
            "Conclusion must be exactly 2 sentences and directly answer the user's decision question. "
            "Do not repeat evidence verbatim. Do not leave any section empty. "
            f"Goal: {goal}\nEvidence: {json.dumps(payload, ensure_ascii=False)}"
        )
        markdown, token_usage, cost_usd = await self._chat_text(prompt, max_tokens=self.synthesize_max_tokens)
        return {"markdown": markdown, "token_usage": token_usage, "cost_usd": cost_usd}

    async def critique(self, goal: str, draft_markdown: str, artifacts: list[Any], round_index: int) -> dict[str, Any]:
        payload = {"goal": goal, "round_index": round_index, "artifact_count": len(artifacts), "draft": draft_markdown[:1800]}
        prompt = (
            "Return JSON with approved:boolean, feedback:string, follow_up_tasks:list. "
            "Approve unless a mandatory quality bar fails. Mandatory bar: draft includes Findings, Sources, Conclusion, at least 3 verifiable source URLs, and is not obviously truncated. "
            "Optional improvement suggestions belong in feedback but must still set approved=true. "
            "Only set approved=false when the mandatory bar fails, and in that case create at most 2 follow_up_tasks. "
            "Each follow_up_task must include title, query, description. "
            f"Input: {json.dumps(payload, ensure_ascii=False)}"
        )
        return await self._chat_json(prompt, max_tokens=self.critique_max_tokens)

    async def single_agent_report(self, goal: str, pages: list[WebPage]) -> dict[str, Any]:
        payload = [{"title": page.title, "url": page.url, "content": page.content[:800]} for page in pages[:6]]
        prompt = (
            "Write a concise markdown research report from the provided web pages. "
            "Output exactly these sections in order: Goal, Findings, Sources, Conclusion. "
            "Use plain markdown. Findings must be 3 bullet points max. "
            "Sources must be exactly 3 numbered URLs, one per line. "
            "Conclusion must be exactly 2 sentences. "
            "Use only the provided pages and do not invent sources. "
            f"Goal: {goal}\nPages: {json.dumps(payload, ensure_ascii=False)}"
        )
        markdown, token_usage, cost_usd = await self._chat_text(prompt, max_tokens=self.single_agent_max_tokens)
        return {"markdown": markdown, "token_usage": token_usage, "cost_usd": cost_usd}

    async def _chat_json(self, prompt: str, max_tokens: int = 800) -> dict[str, Any]:
        text, token_usage, cost_usd = await self._chat_text(prompt, json_mode=True, max_tokens=max_tokens)
        payload = json.loads(text)
        payload.setdefault("token_usage", token_usage)
        payload.setdefault("cost_usd", cost_usd)
        return payload

    async def _chat_text(self, prompt: str, json_mode: bool = False, max_tokens: int = 800) -> tuple[str, int, float]:
        body = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": "You are a precise research workflow assistant. Be concise, finish every requested section, and prefer short complete outputs over long incomplete ones."},
                {"role": "user", "content": prompt},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        token_usage = int(usage.get("total_tokens", max(1, len(prompt) // 4)))
        return message, token_usage, 0.0
