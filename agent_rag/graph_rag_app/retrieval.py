from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from .config import DEFAULT_KEYWORD_WEIGHT, DEFAULT_TOP_K
from .corpus import chunk_text, load_corpus_text, tokenize


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    score: float
    text: str
    document_id: str = ""
    source_path: str = ""
    section_title: str = ""
    strategy: str = "hybrid"


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
        strategy: str = "hybrid",
    ) -> Sequence[SearchResult]:
        ...

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> Sequence[SearchResult | Mapping[str, Any]]:
        ...


_METADATA_STOPWORDS = {
    "a",
    "about",
    "agent",
    "agents",
    "ai",
    "an",
    "and",
    "announcement",
    "announcements",
    "change",
    "changes",
    "compare",
    "comparison",
    "difference",
    "differences",
    "for",
    "in",
    "latest",
    "news",
    "of",
    "on",
    "recent",
    "the",
    "to",
    "update",
    "updates",
    "vs",
    "what",
}
_HEADER_VALUE_PATTERN = re.compile(r"(?im)^\[(SOURCE|TITLE|SECTION):\s*(.*?)\]\s*$")


def _query_focus_tokens(query: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in tokenize(query):
        normalized = token.strip().lower()
        if not normalized or normalized.isdigit():
            continue
        if len(normalized) < 3:
            continue
        if normalized in _METADATA_STOPWORDS:
            continue
        tokens.append(normalized)
    return tuple(dict.fromkeys(tokens))


def _extract_header_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in _HEADER_VALUE_PATTERN.findall(text):
        metadata[key.lower()] = value.strip()
    return metadata


def _normalized_metadata_fields(result: SearchResult) -> tuple[str, str, str]:
    header_metadata = _extract_header_metadata(result.text)
    source_path = (result.source_path or header_metadata.get("source") or "").lower()
    title = header_metadata.get("title", "").lower()
    section_title = (result.section_title or header_metadata.get("section") or "").lower()
    return source_path, title, section_title


def _metadata_bonus(result: SearchResult, query_tokens: tuple[str, ...]) -> float:
    if not query_tokens:
        return 0.0

    source_path, title, section_title = _normalized_metadata_fields(result)
    bonus = 0.0
    for token in query_tokens:
        if source_path and token in source_path:
            bonus += 0.40
        if title and token in title:
            bonus += 0.30
        if section_title and token in section_title:
            bonus += 0.20
    return min(bonus, 2.40)


def rerank_results_by_metadata(
    query: str,
    results: Sequence[SearchResult],
    *,
    top_k: int,
) -> list[SearchResult]:
    query_tokens = _query_focus_tokens(query)
    if not results or not query_tokens:
        return list(results)[:top_k]

    reranked: list[SearchResult] = []
    for result in results:
        bonus = _metadata_bonus(result, query_tokens)
        reranked.append(
            SearchResult(
                chunk_id=result.chunk_id,
                score=result.score + bonus,
                text=result.text,
                document_id=result.document_id,
                source_path=result.source_path,
                section_title=result.section_title,
                strategy=result.strategy if bonus <= 0 else f"{result.strategy}+metadata",
            )
        )

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked[:top_k]


def normalize_search_result(item: SearchResult | Mapping[str, Any]) -> SearchResult:
    if isinstance(item, SearchResult):
        return item
    return SearchResult(
        chunk_id=int(item["chunk_id"]),
        score=float(item["score"]),
        text=str(item["text"]),
        document_id=str(item.get("document_id", "")),
        source_path=str(item.get("source_path", "")),
        section_title=str(item.get("section_title", "")),
        strategy=str(item.get("strategy", "hybrid")),
    )


class DashScopeEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "text-embedding-v3",
        timeout_seconds: int = 30,
        max_batch_size: int = 10,
    ):
        if not api_key:
            raise ValueError("api_key is required for embedding client.")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_batch_size = max(1, max_batch_size)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib_request.Request(
            url=f"{self.api_base}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:  # pragma: no cover - network call
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Embedding request failed: {exc.code} {detail}") from exc
        except urllib_error.URLError as exc:  # pragma: no cover - network call
            raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc

        body = json.loads(raw)
        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected embedding response body: {raw}")
        data.sort(key=lambda item: item.get("index", 0))

        vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(f"Unexpected embedding item: {item}")
            vectors.append([float(value) for value in embedding])

        if len(vectors) != len(texts):
            raise RuntimeError("Embedding response length does not match input length.")
        return vectors

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch_size):
            batch = texts[start:start + self.max_batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text])
        return vectors[0] if vectors else []


class LocalRAGStore:
    def __init__(
        self,
        kb_path: str | Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        embedding_client: EmbeddingBackend | None = None,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    ):
        self.kb_path = Path(kb_path)
        self.embedding_client = embedding_client
        self.keyword_weight = max(0.0, min(1.0, keyword_weight))
        self.text = load_corpus_text(self.kb_path)
        self.chunks = chunk_text(self.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not self.chunks:
            raise ValueError("鏂囨。涓湭鎻愬彇鍒板彲妫€绱㈠唴瀹广€?")

        self.chunk_token_counts = [Counter(tokenize(chunk)) for chunk in self.chunks]
        self.chunk_lengths = [sum(counter.values()) for counter in self.chunk_token_counts]
        self.avg_chunk_length = (
            sum(self.chunk_lengths) / len(self.chunk_lengths) if self.chunk_lengths else 1.0
        )
        self.bm25_idf = self._build_bm25_idf(self.chunk_token_counts)
        self.idf = self._build_idf(self.chunk_token_counts)
        self.chunk_keyword_vectors = [self._vectorize(counter) for counter in self.chunk_token_counts]
        self.chunk_vectors = self.chunk_keyword_vectors
        self.chunk_dense_vectors = self._build_dense_vectors()

    @staticmethod
    def _build_idf(counters: list[Counter[str]]) -> dict[str, float]:
        doc_count = len(counters)
        document_frequency: Counter[str] = Counter()
        for counter in counters:
            document_frequency.update(counter.keys())

        return {
            token: math.log((1 + doc_count) / (1 + freq)) + 1.0
            for token, freq in document_frequency.items()
        }

    @staticmethod
    def _build_bm25_idf(counters: list[Counter[str]]) -> dict[str, float]:
        doc_count = len(counters)
        document_frequency: Counter[str] = Counter()
        for counter in counters:
            document_frequency.update(counter.keys())

        return {
            token: math.log(1 + ((doc_count - freq + 0.5) / (freq + 0.5)))
            for token, freq in document_frequency.items()
        }

    def _vectorize(self, token_counts: Counter[str]) -> dict[str, float]:
        total = sum(token_counts.values())
        if total == 0:
            return {}

        vector = {
            token: (count / total) * self.idf.get(token, 0.0)
            for token, count in token_counts.items()
            if token in self.idf
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {token: value / norm for token, value in vector.items()}

    @staticmethod
    def _normalize_dense_vector(vector: list[float]) -> list[float]:
        if not vector:
            return []
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return []
        return [value / norm for value in vector]

    def _bm25_score(
        self,
        doc_token_counts: Counter[str],
        query_token_counts: Counter[str],
        doc_length: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        if not doc_token_counts or not query_token_counts:
            return 0.0

        norm = k1 * (1.0 - b + b * (doc_length / max(self.avg_chunk_length, 1e-9)))
        score = 0.0
        for token in query_token_counts:
            term_freq = doc_token_counts.get(token, 0)
            if term_freq <= 0:
                continue
            idf = self.bm25_idf.get(token, 0.0)
            score += idf * ((term_freq * (k1 + 1.0)) / (term_freq + norm))
        return score

    @staticmethod
    def _dense_cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(l_value * r_value for l_value, r_value in zip(left, right))

    @staticmethod
    def _stable_hash(token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big")
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return bucket, sign

    def _hashed_dense_vector(self, token_counts: Counter[str], dim: int = 256) -> list[float]:
        vector = [0.0] * dim
        for token, count in token_counts.items():
            bucket, sign = self._stable_hash(token)
            vector[bucket % dim] += sign * float(count)
        return self._normalize_dense_vector(vector)

    def _build_dense_vectors(self) -> list[list[float]]:
        if self.embedding_client is not None:
            dense_vectors = self.embedding_client.embed_documents(self.chunks)
            return [self._normalize_dense_vector(vector) for vector in dense_vectors]
        return [self._hashed_dense_vector(token_counts) for token_counts in self.chunk_token_counts]

    @staticmethod
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}

        values = list(scores.values())
        high = max(values)
        low = min(values)
        if math.isclose(high, low):
            if high <= 0:
                return {chunk_id: 0.0 for chunk_id in scores}
            return {chunk_id: 1.0 for chunk_id in scores}

        denominator = high - low
        return {chunk_id: (score - low) / denominator for chunk_id, score in scores.items()}

    def _keyword_scores(self, query: str) -> dict[int, float]:
        query_token_counts = Counter(tokenize(query))
        scores: dict[int, float] = {}
        for index, chunk_token_counts in enumerate(self.chunk_token_counts):
            score = self._bm25_score(
                doc_token_counts=chunk_token_counts,
                query_token_counts=query_token_counts,
                doc_length=self.chunk_lengths[index],
            )
            if score > 0:
                scores[index] = score
        return scores

    def _dense_scores(self, query: str) -> dict[int, float]:
        if self.embedding_client is not None:
            query_vector = self._normalize_dense_vector(self.embedding_client.embed_query(query))
        else:
            query_vector = self._hashed_dense_vector(Counter(tokenize(query)))

        scores: dict[int, float] = {}
        for index, chunk_vector in enumerate(self.chunk_dense_vectors):
            score = self._dense_cosine_similarity(query_vector, chunk_vector)
            if score > 0:
                scores[index] = score
        return scores

    def search_results(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        if not query.strip():
            return []

        keyword_scores = self._keyword_scores(query)
        dense_scores = self._dense_scores(query)
        if not keyword_scores and not dense_scores:
            return []

        keyword_scores = self._normalize_scores(keyword_scores)
        dense_scores = self._normalize_scores(dense_scores)

        dense_weight = 1.0 - self.keyword_weight
        candidates = set(keyword_scores) | set(dense_scores)
        fused_scores: list[SearchResult] = []
        for chunk_id in candidates:
            score = (
                self.keyword_weight * keyword_scores.get(chunk_id, 0.0)
                + dense_weight * dense_scores.get(chunk_id, 0.0)
            )
            if score <= 0:
                continue
            fused_scores.append(
                SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=self.chunks[chunk_id],
                    strategy="hybrid",
                )
            )

        fused_scores.sort(key=lambda item: item.score, reverse=True)
        return rerank_results_by_metadata(query, fused_scores, top_k=top_k)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
        strategy: str = "hybrid",
    ) -> list[SearchResult]:
        del filters
        if strategy == "hybrid":
            return self.search_results(query, top_k=top_k)

        if not query.strip():
            return []

        if strategy == "sparse":
            sparse_scores = self._normalize_scores(self._keyword_scores(query))
            ranked = sorted(sparse_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
            results = [
                SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=self.chunks[chunk_id],
                    strategy="sparse",
                )
                for chunk_id, score in ranked
            ]
            return rerank_results_by_metadata(query, results, top_k=top_k)

        if strategy == "dense":
            dense_scores = self._normalize_scores(self._dense_scores(query))
            ranked = sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
            results = [
                SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=self.chunks[chunk_id],
                    strategy="dense",
                )
                for chunk_id, score in ranked
            ]
            return rerank_results_by_metadata(query, results, top_k=top_k)

        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.retrieve(query, top_k=top_k, strategy="hybrid")]
