from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_TOP_K = 3
DEFAULT_KEYWORD_WEIGHT = 0.5
DEFAULT_SESSION_ID = "graph_rag_default"
DEFAULT_CHECKPOINT_DB = "checkpoints.db"
DEFAULT_MODEL_NAME = "qwen3.6-plus"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_INDEX_VERSION = 1
DEFAULT_INDEX_DB_FILENAME = "retrieval.sqlite3"
DEFAULT_MANIFEST_FILENAME = "manifest.json"


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _parse_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    api_base: str = DEFAULT_API_BASE
    model_name: str = DEFAULT_MODEL_NAME


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    max_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    top_k: int = DEFAULT_TOP_K


@dataclass(frozen=True)
class IndexBuildConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    dense_dim: int = 256
    index_version: int = DEFAULT_INDEX_VERSION


@dataclass(frozen=True)
class RetrievalRuntimeConfig:
    top_k: int = DEFAULT_TOP_K
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    strategy: str = "hybrid"


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
class GenerationConfig:
    max_rounds: int = 3
    min_evidence_score: float = 0.0
    allow_query_decomposition: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    session_id: str = DEFAULT_SESSION_ID
    checkpoint_db: str = DEFAULT_CHECKPOINT_DB
    resume: bool = False
    interrupt_after: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AppConfig:
    kb_path: Path
    model: ModelConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    web: WebConfig
    generation: GenerationConfig
    runtime: RuntimeConfig


def build_app_config(
    kb_path: str | Path,
    *,
    session_id: str = DEFAULT_SESSION_ID,
    checkpoint_db: str = DEFAULT_CHECKPOINT_DB,
    resume: bool = False,
    interrupt_after: list[str] | tuple[str, ...] | None = None,
) -> AppConfig:
    web_defaults = WebConfig()
    return AppConfig(
        kb_path=Path(kb_path),
        model=ModelConfig(api_key=os.getenv("DASHSCOPE_API_KEY", "")),
        embedding=EmbeddingConfig(
            model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            max_batch_size=_parse_int(
                os.getenv("DASHSCOPE_EMBEDDING_BATCH_SIZE"),
                DEFAULT_EMBEDDING_BATCH_SIZE,
            ),
        ),
        retrieval=RetrievalConfig(
            keyword_weight=_parse_float(
                os.getenv("RAG_KEYWORD_WEIGHT"),
                DEFAULT_KEYWORD_WEIGHT,
            ),
        ),
        web=WebConfig(
            enabled=parse_bool_env("RAG_WEB_ENABLED", web_defaults.enabled),
            search_provider=os.getenv("RAG_WEB_SEARCH_PROVIDER", web_defaults.search_provider),
            search_top_k=_parse_int(
                os.getenv("RAG_WEB_SEARCH_TOP_K"),
                web_defaults.search_top_k,
            ),
            fetch_timeout_seconds=_parse_int(
                os.getenv("RAG_WEB_FETCH_TIMEOUT_SECONDS"),
                web_defaults.fetch_timeout_seconds,
            ),
            fetch_max_bytes=_parse_int(
                os.getenv("RAG_WEB_FETCH_MAX_BYTES"),
                web_defaults.fetch_max_bytes,
            ),
            fetch_max_chars=_parse_int(
                os.getenv("RAG_WEB_FETCH_MAX_CHARS"),
                web_defaults.fetch_max_chars,
            ),
            user_agent=os.getenv("RAG_WEB_USER_AGENT", web_defaults.user_agent),
        ),
        generation=GenerationConfig(
            max_rounds=_parse_int(os.getenv("RAG_MAX_ROUNDS"), GenerationConfig().max_rounds),
            min_evidence_score=_parse_float(
                os.getenv("RAG_MIN_EVIDENCE_SCORE"),
                GenerationConfig().min_evidence_score,
            ),
            allow_query_decomposition=parse_bool_env(
                "RAG_ALLOW_QUERY_DECOMPOSITION",
                GenerationConfig().allow_query_decomposition,
            ),
        ),
        runtime=RuntimeConfig(
            session_id=session_id,
            checkpoint_db=checkpoint_db,
            resume=resume,
            interrupt_after=tuple(interrupt_after or ()),
        ),
    )
