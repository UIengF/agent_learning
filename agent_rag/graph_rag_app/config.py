from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_TOP_K = 3
DEFAULT_KEYWORD_WEIGHT = 0.5
DEFAULT_SESSION_ID = "graph_rag_default"
DEFAULT_CHECKPOINT_DB = "runtime/checkpoints.db"
DEFAULT_USER_MEMORY_PATH = "runtime/user_memory.json"
DEFAULT_USER_ID = "default_user"
DEFAULT_MODEL_NAME = "qwen3.6-plus"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_INDEX_VERSION = 1
DEFAULT_INDEX_DB_FILENAME = "retrieval.sqlite3"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_MAX_RECENT_MESSAGES = 8
DEFAULT_RECENT_FULL_TURNS = 3
DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_CONTEXT_TOKENS = 100000
DEFAULT_LIVE_MESSAGES_KEEP_TURNS = 1
DEFAULT_LIVE_MESSAGES_MAX_FETCH_CHARS = 180
DEFAULT_LIVE_MESSAGES_MAX_SEARCH_RESULTS = 3
DEFAULT_WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36 graph-rag-agent/1.0"
)


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
    user_agent: str = DEFAULT_WEB_USER_AGENT


@dataclass(frozen=True)
class ScholarConfig:
    enabled: bool = True
    api_key: str = ""
    default_count: int = 5
    max_count: int = 20
    engine: str = "google_scholar"


@dataclass(frozen=True)
class ContextConfig:
    max_recent_messages: int = DEFAULT_MAX_RECENT_MESSAGES
    recent_full_turns: int = DEFAULT_RECENT_FULL_TURNS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    live_messages_compression_enabled: bool = True
    live_messages_keep_turns: int = DEFAULT_LIVE_MESSAGES_KEEP_TURNS
    live_messages_max_fetch_chars: int = DEFAULT_LIVE_MESSAGES_MAX_FETCH_CHARS
    live_messages_max_search_results: int = DEFAULT_LIVE_MESSAGES_MAX_SEARCH_RESULTS


@dataclass(frozen=True)
class GenerationConfig:
    max_rounds: int = 8
    min_evidence_score: float = 0.0
    allow_query_decomposition: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    session_id: str = DEFAULT_SESSION_ID
    checkpoint_db: str = DEFAULT_CHECKPOINT_DB
    user_memory_path: str = DEFAULT_USER_MEMORY_PATH
    user_id: str = DEFAULT_USER_ID
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
    scholar: ScholarConfig = field(default_factory=ScholarConfig)
    context: ContextConfig = field(default_factory=ContextConfig)


def build_app_config(
    kb_path: str | Path,
    *,
    session_id: str = DEFAULT_SESSION_ID,
    checkpoint_db: str = DEFAULT_CHECKPOINT_DB,
    resume: bool = False,
    interrupt_after: list[str] | tuple[str, ...] | None = None,
) -> AppConfig:
    web_defaults = WebConfig()
    scholar_defaults = ScholarConfig()
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
        scholar=ScholarConfig(
            enabled=parse_bool_env("RAG_SCHOLAR_ENABLED", scholar_defaults.enabled),
            api_key=os.getenv("SERPAPI_API_KEY", scholar_defaults.api_key),
            default_count=max(
                1,
                min(
                    20,
                    _parse_int(
                        os.getenv("RAG_SCHOLAR_DEFAULT_COUNT"),
                        scholar_defaults.default_count,
                    ),
                ),
            ),
            max_count=max(
                1,
                min(
                    20,
                    _parse_int(
                        os.getenv("RAG_SCHOLAR_MAX_COUNT"),
                        scholar_defaults.max_count,
                    ),
                ),
            ),
            engine=os.getenv("RAG_SCHOLAR_ENGINE", scholar_defaults.engine),
        ),
        context=ContextConfig(
            max_recent_messages=_parse_int(
                os.getenv("RAG_MAX_RECENT_MESSAGES"),
                DEFAULT_MAX_RECENT_MESSAGES,
            ),
            recent_full_turns=_parse_int(
                os.getenv("RAG_RECENT_FULL_TURNS"),
                DEFAULT_RECENT_FULL_TURNS,
            ),
            max_context_chars=_parse_int(
                os.getenv("RAG_MAX_CONTEXT_CHARS"),
                DEFAULT_MAX_CONTEXT_CHARS,
            ),
            max_context_tokens=_parse_int(
                os.getenv("RAG_MAX_CONTEXT_TOKENS"),
                DEFAULT_MAX_CONTEXT_TOKENS,
            ),
            live_messages_compression_enabled=parse_bool_env(
                "RAG_LIVE_MESSAGES_COMPRESSION_ENABLED",
                True,
            ),
            live_messages_keep_turns=_parse_int(
                os.getenv("RAG_LIVE_MESSAGES_KEEP_TURNS"),
                DEFAULT_LIVE_MESSAGES_KEEP_TURNS,
            ),
            live_messages_max_fetch_chars=_parse_int(
                os.getenv("RAG_LIVE_MESSAGES_MAX_FETCH_CHARS"),
                DEFAULT_LIVE_MESSAGES_MAX_FETCH_CHARS,
            ),
            live_messages_max_search_results=_parse_int(
                os.getenv("RAG_LIVE_MESSAGES_MAX_SEARCH_RESULTS"),
                DEFAULT_LIVE_MESSAGES_MAX_SEARCH_RESULTS,
            ),
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
            user_memory_path=os.getenv("RAG_USER_MEMORY_PATH", DEFAULT_USER_MEMORY_PATH),
            user_id=os.getenv("RAG_USER_ID", DEFAULT_USER_ID),
            resume=resume,
            interrupt_after=tuple(interrupt_after or ()),
        ),
    )
