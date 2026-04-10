from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DEFAULT_INDEX_DB_FILENAME,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_TOP_K,
    IndexBuildConfig,
)
from .corpus import chunk_text, load_docx_text, load_markdown_text, tokenize
from .retrieval import (
    DashScopeEmbeddingClient,
    EmbeddingBackend,
    LocalRAGStore,
    SearchResult,
)


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    source_path: str
    text: str
    updated_at: float
    section_title: str = ""


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    document_id: str
    source_path: str
    section_title: str
    chunk_index: int
    updated_at: float
    text: str
    token_counts: dict[str, int]
    dense_vector: list[float]


@dataclass(frozen=True)
class BuildReport:
    index_dir: str
    chunk_count: int
    document_count: int
    built_at: str
    index_db_path: str
    manifest_path: str


class IndexedRetriever:
    def __init__(self, index_dir: str | Path, keyword_weight: float | None = None):
        self.index_dir = Path(index_dir)
        self.manifest_path = self.index_dir / DEFAULT_MANIFEST_FILENAME
        self.db_path = self.index_dir / DEFAULT_INDEX_DB_FILENAME
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.keyword_weight = float(
            keyword_weight if keyword_weight is not None else manifest["config"]["keyword_weight"]
        )
        self.chunk_records = self._load_chunk_records(self.db_path)
        self.chunk_token_counts = [Counter(record["token_counts"]) for record in self.chunk_records]
        self.chunk_lengths = [int(record["doc_length"]) for record in self.chunk_records]
        self.avg_chunk_length = (
            sum(self.chunk_lengths) / len(self.chunk_lengths) if self.chunk_lengths else 1.0
        )
        self.bm25_idf = self._load_idf_table(self.db_path, "bm25_idf")
        self.idf = self._load_idf_table(self.db_path, "idf")
        self.chunk_dense_vectors = [record["dense_vector"] for record in self.chunk_records]

    @staticmethod
    def _connect(db_path: Path) -> sqlite3.Connection:
        return sqlite3.connect(db_path)

    @classmethod
    def _load_chunk_records(cls, db_path: Path) -> list[dict[str, Any]]:
        conn = cls._connect(db_path)
        try:
            rows = conn.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    source_path,
                    section_title,
                    chunk_index,
                    updated_at,
                    text,
                    doc_length,
                    token_counts_json,
                    dense_vector_json
                FROM chunks
                ORDER BY chunk_id
                """
            ).fetchall()
        finally:
            conn.close()

        records = []
        for row in rows:
            records.append(
                {
                    "chunk_id": int(row[0]),
                    "document_id": str(row[1]),
                    "source_path": str(row[2]),
                    "section_title": str(row[3] or ""),
                    "chunk_index": int(row[4]),
                    "updated_at": float(row[5]),
                    "text": str(row[6]),
                    "doc_length": int(row[7]),
                    "token_counts": json.loads(row[8]),
                    "dense_vector": json.loads(row[9]),
                }
            )
        return records

    @classmethod
    def _load_idf_table(cls, db_path: Path, table_name: str) -> dict[str, float]:
        conn = cls._connect(db_path)
        try:
            rows = conn.execute(f"SELECT token, value FROM {table_name}").fetchall()
        finally:
            conn.close()
        return {str(token): float(value) for token, value in rows}

    @staticmethod
    def _normalize_dense_vector(vector: list[float]) -> list[float]:
        return LocalRAGStore._normalize_dense_vector(vector)

    @staticmethod
    def _dense_cosine_similarity(left: list[float], right: list[float]) -> float:
        return LocalRAGStore._dense_cosine_similarity(left, right)

    def _hashed_dense_vector(self, token_counts: Counter[str], dim: int) -> list[float]:
        vector = [0.0] * dim
        for token, count in token_counts.items():
            bucket, sign = LocalRAGStore._stable_hash(token)
            vector[bucket % dim] += sign * float(count)
        return self._normalize_dense_vector(vector)

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
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        return LocalRAGStore._normalize_scores(scores)

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.strip().lower().split())

    def _keyword_scores(self, query: str) -> dict[int, float]:
        query_token_counts = Counter(tokenize(self._normalize_query(query)))
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
        query_tokens = Counter(tokenize(self._normalize_query(query)))
        dim = len(self.chunk_dense_vectors[0]) if self.chunk_dense_vectors else 256
        query_vector = self._hashed_dense_vector(query_tokens, dim=dim)
        scores: dict[int, float] = {}
        for index, chunk_vector in enumerate(self.chunk_dense_vectors):
            score = self._dense_cosine_similarity(query_vector, chunk_vector)
            if score > 0:
                scores[index] = score
        return scores

    def _build_result(self, chunk_id: int, score: float, strategy: str) -> SearchResult:
        record = self.chunk_records[chunk_id]
        return SearchResult(
            chunk_id=chunk_id,
            score=score,
            text=record["text"],
            document_id=record["document_id"],
            source_path=record["source_path"],
            section_title=record["section_title"],
            strategy=strategy,
        )

    @staticmethod
    def _matches_filters(record: dict[str, Any], filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if record.get(key) != expected:
                return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, Any] | None = None,
        strategy: str = "hybrid",
    ) -> list[SearchResult]:
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        if strategy == "sparse":
            sparse_scores = self._normalize_scores(self._keyword_scores(normalized_query))
            ranked = sorted(sparse_scores.items(), key=lambda item: item[1], reverse=True)
            results = [
                self._build_result(chunk_id, score, "sparse")
                for chunk_id, score in ranked
                if self._matches_filters(self.chunk_records[chunk_id], filters)
            ]
            return results[:top_k]

        if strategy == "dense":
            dense_scores = self._normalize_scores(self._dense_scores(normalized_query))
            ranked = sorted(dense_scores.items(), key=lambda item: item[1], reverse=True)
            results = [
                self._build_result(chunk_id, score, "dense")
                for chunk_id, score in ranked
                if self._matches_filters(self.chunk_records[chunk_id], filters)
            ]
            return results[:top_k]

        keyword_scores = self._normalize_scores(self._keyword_scores(normalized_query))
        dense_scores = self._normalize_scores(self._dense_scores(normalized_query))
        dense_weight = 1.0 - self.keyword_weight
        candidates = set(keyword_scores) | set(dense_scores)
        fused_scores = []
        for chunk_id in candidates:
            if not self._matches_filters(self.chunk_records[chunk_id], filters):
                continue
            score = (
                self.keyword_weight * keyword_scores.get(chunk_id, 0.0)
                + dense_weight * dense_scores.get(chunk_id, 0.0)
            )
            if score <= 0:
                continue
            fused_scores.append((chunk_id, score))

        fused_scores.sort(key=lambda item: item[1], reverse=True)
        return [self._build_result(chunk_id, score, "hybrid") for chunk_id, score in fused_scores[:top_k]]

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.retrieve(query=query, top_k=top_k, strategy="hybrid")]


def _document_id_for(path: Path) -> str:
    return path.resolve().as_posix()


def is_index_directory(path: str | Path) -> bool:
    candidate = Path(path)
    return (
        candidate.is_dir()
        and (candidate / DEFAULT_MANIFEST_FILENAME).exists()
        and (candidate / DEFAULT_INDEX_DB_FILENAME).exists()
    )


def default_index_dir_for_kb(kb_path: str | Path) -> Path:
    source = Path(kb_path).resolve()
    digest = hashlib.sha1(source.as_posix().encode("utf-8")).hexdigest()[:12]
    return source.parent / ".graph_rag_index" / digest


def _load_source_documents(kb_path: str | Path) -> list[SourceDocument]:
    root = Path(kb_path)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge base path not found: {root}")

    documents: list[SourceDocument] = []
    if root.is_dir():
        for path in sorted(candidate for candidate in root.rglob("*.md") if candidate.is_file()):
            documents.append(
                SourceDocument(
                    document_id=_document_id_for(path),
                    source_path=path.relative_to(root).as_posix(),
                    text=load_markdown_text(path),
                    updated_at=path.stat().st_mtime,
                )
            )
        if not documents:
            raise ValueError(f"No markdown files found in directory: {root}")
        return documents

    if root.suffix.lower() == ".md":
        return [
            SourceDocument(
                document_id=_document_id_for(root),
                source_path=root.name,
                text=load_markdown_text(root),
                updated_at=root.stat().st_mtime,
            )
        ]

    if root.suffix.lower() == ".docx":
        return [
            SourceDocument(
                document_id=_document_id_for(root),
                source_path=root.name,
                text=load_docx_text(root),
                updated_at=root.stat().st_mtime,
            )
        ]

    raise ValueError(f"Unsupported knowledge base path: {root}")


def _latest_source_mtime(kb_path: str | Path) -> float:
    root = Path(kb_path)
    if root.is_dir():
        markdown_files = [path for path in root.rglob("*.md") if path.is_file()]
        if not markdown_files:
            raise ValueError(f"No markdown files found in directory: {root}")
        return max(path.stat().st_mtime for path in markdown_files)
    return root.stat().st_mtime


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = stripped.lstrip("#").strip()
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [section for section in sections if section[1]]


def _fallback_dense_vector(token_counts: Counter[str], dim: int) -> list[float]:
    vector = [0.0] * dim
    for token, count in token_counts.items():
        bucket, sign = LocalRAGStore._stable_hash(token)
        vector[bucket % dim] += sign * float(count)
    return LocalRAGStore._normalize_dense_vector(vector)


def _document_title_for(document: SourceDocument) -> str:
    frontmatter_match = re.search(r"(?im)^title:\s*[\"']?(.*?)[\"']?\s*$", document.text)
    if frontmatter_match:
        title = frontmatter_match.group(1).strip()
        if title:
            return title

    stem = Path(document.source_path).stem
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}\s+", "", stem)
    if " - " in stem:
        stem = stem.split(" - ", 1)[1].strip()
    return stem.strip()


def _build_chunk_header(source_path: str, document_title: str, section_title: str) -> str:
    parts = [f"[SOURCE: {source_path}]"]
    if document_title:
        parts.append(f"[TITLE: {document_title}]")
    if section_title:
        parts.append(f"[SECTION: {section_title}]")
    return "\n".join(parts)


def _take_prefix_for_completion(text: str, limit: int) -> str:
    if limit <= 0:
        return ""

    paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    selected: list[str] = []
    total = 0
    for paragraph in paragraphs:
        addition = len(paragraph) + (2 if selected else 0)
        if selected and total + addition > limit:
            break
        if len(paragraph) <= limit and total + addition <= limit:
            selected.append(paragraph)
            total += addition
            continue
        break
    if selected:
        return "\n\n".join(selected)

    return text[:limit].strip()


def _expand_short_lead_chunk(
    chunks: list[str],
    *,
    target_size: int,
    min_body_chars: int = 120,
) -> list[str]:
    if len(chunks) < 2:
        return chunks
    first_chunk = chunks[0].strip()
    if len(first_chunk) >= min_body_chars:
        return chunks

    remaining = target_size - len(first_chunk) - 2
    if remaining <= 0:
        return chunks

    next_chunk = chunks[1].strip()
    if next_chunk.startswith(first_chunk):
        next_chunk = next_chunk[len(first_chunk):].strip()

    supplement = _take_prefix_for_completion(next_chunk, remaining)
    if not supplement:
        return chunks

    expanded = list(chunks)
    expanded[0] = f"{first_chunk}\n\n{supplement}".strip()
    return expanded


def _enhance_chunk_text(header: str, body: str) -> str:
    if not header:
        return body.strip()
    if not body.strip():
        return header
    return f"{header}\n\n{body.strip()}"


def _build_chunk_records(
    documents: Iterable[SourceDocument],
    config: IndexBuildConfig,
    embedding_client: EmbeddingBackend | None,
) -> list[ChunkRecord]:
    interim_chunks: list[dict[str, Any]] = []
    for document in documents:
        document_title = _document_title_for(document)
        if document.source_path.lower().endswith(".md"):
            sections = _split_markdown_sections(document.text) or [("", document.text)]
        else:
            sections = [("", document.text)]

        chunk_index = 0
        for section_title, section_text in sections:
            header = _build_chunk_header(document.source_path, document_title, section_title)
            body_chunk_size = max(120, config.chunk_size - len(header) - 2)
            body_chunk_overlap = min(config.chunk_overlap, max(0, body_chunk_size - 1))
            chunks = chunk_text(
                section_text,
                chunk_size=body_chunk_size,
                chunk_overlap=body_chunk_overlap,
            )
            if section_title:
                chunks = _expand_short_lead_chunk(chunks, target_size=body_chunk_size)
            for chunk in chunks:
                enhanced_text = _enhance_chunk_text(header, chunk)
                interim_chunks.append(
                    {
                        "document_id": document.document_id,
                        "source_path": document.source_path,
                        "section_title": section_title,
                        "chunk_index": chunk_index,
                        "updated_at": document.updated_at,
                        "text": enhanced_text,
                        "token_counts": dict(Counter(tokenize(enhanced_text))),
                    }
                )
                chunk_index += 1

    texts = [item["text"] for item in interim_chunks]
    if embedding_client is not None:
        dense_vectors = embedding_client.embed_documents(texts)
        dense_vectors = [LocalRAGStore._normalize_dense_vector(vector) for vector in dense_vectors]
    else:
        dense_vectors = [
            _fallback_dense_vector(Counter(item["token_counts"]), dim=config.dense_dim)
            for item in interim_chunks
        ]

    records: list[ChunkRecord] = []
    for chunk_id, (item, dense_vector) in enumerate(zip(interim_chunks, dense_vectors)):
        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=item["document_id"],
                source_path=item["source_path"],
                section_title=item["section_title"],
                chunk_index=item["chunk_index"],
                updated_at=item["updated_at"],
                text=item["text"],
                token_counts=item["token_counts"],
                dense_vector=dense_vector,
            )
        )
    return records


def _build_idf_tables(records: list[ChunkRecord]) -> tuple[dict[str, float], dict[str, float]]:
    counters = [Counter(record.token_counts) for record in records]
    idf = LocalRAGStore._build_idf(counters)
    bm25_idf = LocalRAGStore._build_bm25_idf(counters)
    return idf, bm25_idf


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY,
            document_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            section_title TEXT,
            chunk_index INTEGER NOT NULL,
            updated_at REAL NOT NULL,
            text TEXT NOT NULL,
            doc_length INTEGER NOT NULL,
            token_counts_json TEXT NOT NULL,
            dense_vector_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS idf (token TEXT PRIMARY KEY, value REAL NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS bm25_idf (token TEXT PRIMARY KEY, value REAL NOT NULL)")
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM idf")
    conn.execute("DELETE FROM bm25_idf")
    return conn


def _default_embedding_client() -> EmbeddingBackend | None:
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return None
    return DashScopeEmbeddingClient(
        api_key=api_key,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"),
        max_batch_size=int(os.getenv("DASHSCOPE_EMBEDDING_BATCH_SIZE", "10")),
    )


def build_index(
    kb_path: str | Path,
    output_dir: str | Path,
    config: IndexBuildConfig | None = None,
    embedding_client: EmbeddingBackend | None = None,
) -> BuildReport:
    index_config = config or IndexBuildConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    documents = _load_source_documents(kb_path)
    records = _build_chunk_records(documents, index_config, embedding_client or _default_embedding_client())
    idf, bm25_idf = _build_idf_tables(records)

    db_path = output / DEFAULT_INDEX_DB_FILENAME
    manifest_path = output / DEFAULT_MANIFEST_FILENAME
    conn = _init_db(db_path)
    try:
        conn.executemany(
            """
            INSERT INTO chunks (
                chunk_id, document_id, source_path, section_title, chunk_index, updated_at,
                text, doc_length, token_counts_json, dense_vector_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    record.chunk_id,
                    record.document_id,
                    record.source_path,
                    record.section_title,
                    record.chunk_index,
                    record.updated_at,
                    record.text,
                    sum(record.token_counts.values()),
                    json.dumps(record.token_counts, ensure_ascii=False, sort_keys=True),
                    json.dumps(record.dense_vector),
                )
                for record in records
            ],
        )
        conn.executemany(
            "INSERT INTO idf(token, value) VALUES(?, ?)",
            sorted(idf.items()),
        )
        conn.executemany(
            "INSERT INTO bm25_idf(token, value) VALUES(?, ?)",
            sorted(bm25_idf.items()),
        )
        conn.commit()
    finally:
        conn.close()

    built_at = datetime.now(UTC).isoformat()
    manifest = {
        "version": index_config.index_version,
        "kb_path": str(Path(kb_path).resolve()),
        "index_db": db_path.name,
        "chunk_count": len(records),
        "document_count": len(documents),
        "built_at": built_at,
        "config": asdict(index_config),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return BuildReport(
        index_dir=str(output),
        chunk_count=len(records),
        document_count=len(documents),
        built_at=built_at,
        index_db_path=str(db_path),
        manifest_path=str(manifest_path),
    )


def load_index(index_dir: str | Path, *, keyword_weight: float | None = None) -> IndexedRetriever:
    return IndexedRetriever(index_dir=index_dir, keyword_weight=keyword_weight)


def inspect_index(index_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(index_dir) / DEFAULT_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_dir"] = str(Path(index_dir).resolve())
    return manifest


def resolve_existing_index_for_kb(kb_path: str | Path) -> Path:
    if is_index_directory(kb_path):
        return Path(kb_path)

    index_dir = default_index_dir_for_kb(kb_path)
    if not is_index_directory(index_dir):
        raise FileNotFoundError(
            f"No built index found for '{kb_path}'. Run `index build` before asking questions."
        )
    return index_dir


def ensure_index_for_kb(
    kb_path: str | Path,
    *,
    config: IndexBuildConfig | None = None,
    embedding_client: EmbeddingBackend | None = None,
) -> Path:
    if is_index_directory(kb_path):
        return Path(kb_path)

    index_dir = default_index_dir_for_kb(kb_path)
    manifest_path = index_dir / DEFAULT_MANIFEST_FILENAME
    needs_rebuild = True
    if manifest_path.exists() and (index_dir / DEFAULT_INDEX_DB_FILENAME).exists():
        try:
            source_mtime = _latest_source_mtime(kb_path)
            needs_rebuild = manifest_path.stat().st_mtime < source_mtime
        except Exception:
            needs_rebuild = True

    if needs_rebuild:
        build_index(
            kb_path=kb_path,
            output_dir=index_dir,
            config=config,
            embedding_client=embedding_client,
        )
    return index_dir
