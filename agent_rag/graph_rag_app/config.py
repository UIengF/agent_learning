from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"
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
DEFAULT_MAX_CONTEXT_CHARS = 20000
DEFAULT_MAX_CONTEXT_TOKENS = 100000
DEFAULT_LIVE_MESSAGES_KEEP_TURNS = 1
DEFAULT_LIVE_MESSAGES_MAX_FETCH_CHARS = 180
DEFAULT_LIVE_MESSAGES_MAX_SEARCH_RESULTS = 3
DEFAULT_WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36 graph-rag-agent/1.0"
)
DEFAULT_SEARXNG_URL = "http://127.0.0.1:8080"


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_env_file(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else PROJECT_ENV_PATH
    if not env_path.is_file():
        import sys as _sys
        print(
            f"[graph_rag] WARNING: No .env file found at {env_path}. "
            f"Using defaults — model={DEFAULT_MODEL_NAME}, "
            f"api_base={DEFAULT_API_BASE}. "
            f"Copy .env.example to .env to customize.",
            file=_sys.stderr,
        )
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        key = name.strip()
        if not key or key in os.environ:
            continue
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1]
        os.environ[key] = cleaned


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
class MetadataRerankConfig:
    source_path_token_bonus: float = 0.40
    title_token_bonus: float = 0.30
    section_title_token_bonus: float = 0.20
    max_bonus: float = 2.40
    authority_boost: float = 0.50


@dataclass(frozen=True)
class IndexBuildConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT
    dense_dim: int = 256
    index_version: int = DEFAULT_INDEX_VERSION
    force_rebuild: bool = False


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
    searxng_url: str = DEFAULT_SEARXNG_URL
    searxng_engines: str = ""
    searxng_categories: str = "general"
    searxng_language: str = "zh-CN"


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
class EvalJudgeConfig:
    enabled: bool = False
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model_name: str = DEFAULT_MODEL_NAME


@dataclass(frozen=True)
class LangSmithConfig:
    enabled: bool = False
    tracing_v2: bool = False
    api_key: str = ""
    project_name: str = "agent-rag"
    endpoint: str = ""


@dataclass(frozen=True)
class HarnessConfig:
    skills_dir: str = "skills"
    structured_trace_enabled: bool = True
    structured_trace_dir: str = "runtime/traces"


@dataclass(frozen=True)
class PermissionConfig:
    allowed_index_roots: str = "."
    allow_local_web_fetch: bool = False
    allow_private_web_fetch: bool = False


@dataclass(frozen=True)
class JobConfig:
    runtime_dir: str = "runtime/jobs"
    max_log_chars: int = 12000


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
    eval_judge: EvalJudgeConfig = field(default_factory=EvalJudgeConfig)
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    permissions: PermissionConfig = field(default_factory=PermissionConfig)
    jobs: JobConfig = field(default_factory=JobConfig)
    metadata_rerank: MetadataRerankConfig = field(default_factory=MetadataRerankConfig)


def _build_model_config() -> ModelConfig:
    return ModelConfig(
        api_key=os.getenv("RAG_MODEL_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")),
        api_base=os.getenv("RAG_MODEL_API_BASE", DEFAULT_API_BASE),
        model_name=os.getenv("RAG_MODEL_NAME", DEFAULT_MODEL_NAME),
    )


def _build_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        max_batch_size=_parse_int(
            os.getenv("DASHSCOPE_EMBEDDING_BATCH_SIZE"),
            DEFAULT_EMBEDDING_BATCH_SIZE,
        ),
    )


def _build_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        keyword_weight=_parse_float(
            os.getenv("RAG_KEYWORD_WEIGHT"),
            DEFAULT_KEYWORD_WEIGHT,
        ),
    )


def _build_metadata_rerank_config() -> MetadataRerankConfig:
    defaults = MetadataRerankConfig()
    return MetadataRerankConfig(
        source_path_token_bonus=_parse_float(
            os.getenv("RAG_METADATA_SOURCE_PATH_TOKEN_BONUS"),
            defaults.source_path_token_bonus,
        ),
        title_token_bonus=_parse_float(
            os.getenv("RAG_METADATA_TITLE_TOKEN_BONUS"),
            defaults.title_token_bonus,
        ),
        section_title_token_bonus=_parse_float(
            os.getenv("RAG_METADATA_SECTION_TITLE_TOKEN_BONUS"),
            defaults.section_title_token_bonus,
        ),
        max_bonus=_parse_float(os.getenv("RAG_METADATA_MAX_BONUS"), defaults.max_bonus),
        authority_boost=_parse_float(
            os.getenv("RAG_METADATA_AUTHORITY_BOOST"),
            defaults.authority_boost,
        ),
    )


def _build_web_config() -> WebConfig:
    defaults = WebConfig()
    return WebConfig(
        enabled=parse_bool_env("RAG_WEB_ENABLED", defaults.enabled),
        search_provider=os.getenv("RAG_WEB_SEARCH_PROVIDER", defaults.search_provider),
        search_top_k=_parse_int(
            os.getenv("RAG_WEB_SEARCH_TOP_K"),
            defaults.search_top_k,
        ),
        fetch_timeout_seconds=_parse_int(
            os.getenv("RAG_WEB_FETCH_TIMEOUT_SECONDS"),
            defaults.fetch_timeout_seconds,
        ),
        fetch_max_bytes=_parse_int(
            os.getenv("RAG_WEB_FETCH_MAX_BYTES"),
            defaults.fetch_max_bytes,
        ),
        fetch_max_chars=_parse_int(
            os.getenv("RAG_WEB_FETCH_MAX_CHARS"),
            defaults.fetch_max_chars,
        ),
        user_agent=os.getenv("RAG_WEB_USER_AGENT", defaults.user_agent),
        searxng_url=os.getenv("RAG_SEARXNG_URL", defaults.searxng_url),
        searxng_engines=os.getenv("RAG_SEARXNG_ENGINES", defaults.searxng_engines),
        searxng_categories=os.getenv(
            "RAG_SEARXNG_CATEGORIES",
            defaults.searxng_categories,
        ),
        searxng_language=os.getenv("RAG_SEARXNG_LANGUAGE", defaults.searxng_language),
    )


def _build_scholar_config() -> ScholarConfig:
    defaults = ScholarConfig()
    return ScholarConfig(
        enabled=parse_bool_env("RAG_SCHOLAR_ENABLED", defaults.enabled),
        api_key=os.getenv("SERPAPI_API_KEY", defaults.api_key),
        default_count=max(
            1,
            min(
                20,
                _parse_int(
                    os.getenv("RAG_SCHOLAR_DEFAULT_COUNT"),
                    defaults.default_count,
                ),
            ),
        ),
        max_count=max(
            1,
            min(
                20,
                _parse_int(
                    os.getenv("RAG_SCHOLAR_MAX_COUNT"),
                    defaults.max_count,
                ),
            ),
        ),
        engine=os.getenv("RAG_SCHOLAR_ENGINE", defaults.engine),
    )


def _build_context_config() -> ContextConfig:
    return ContextConfig(
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
    )


def _build_generation_config() -> GenerationConfig:
    defaults = GenerationConfig()
    return GenerationConfig(
        max_rounds=_parse_int(os.getenv("RAG_MAX_ROUNDS"), defaults.max_rounds),
        min_evidence_score=_parse_float(
            os.getenv("RAG_MIN_EVIDENCE_SCORE"),
            defaults.min_evidence_score,
        ),
        allow_query_decomposition=parse_bool_env(
            "RAG_ALLOW_QUERY_DECOMPOSITION",
            defaults.allow_query_decomposition,
        ),
    )


def _build_eval_judge_config() -> EvalJudgeConfig:
    defaults = EvalJudgeConfig()
    return EvalJudgeConfig(
        enabled=parse_bool_env("RAG_EVAL_JUDGE_ENABLED", defaults.enabled),
        api_key=os.getenv(
            "RAG_EVAL_JUDGE_API_KEY",
            os.getenv("RAG_MODEL_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")),
        ),
        api_base=os.getenv(
            "RAG_EVAL_JUDGE_API_BASE",
            os.getenv("RAG_MODEL_API_BASE", DEFAULT_API_BASE),
        ),
        model_name=os.getenv(
            "RAG_EVAL_JUDGE_MODEL",
            os.getenv("RAG_MODEL_NAME", DEFAULT_MODEL_NAME),
        ),
    )


def _build_langsmith_config() -> LangSmithConfig:
    defaults = LangSmithConfig()
    return LangSmithConfig(
        enabled=parse_bool_env("RAG_LANGSMITH_ENABLED", defaults.enabled),
        tracing_v2=parse_bool_env("LANGCHAIN_TRACING_V2", defaults.tracing_v2),
        api_key=os.getenv("LANGCHAIN_API_KEY", defaults.api_key),
        project_name=os.getenv("LANGCHAIN_PROJECT", defaults.project_name),
        endpoint=os.getenv("LANGCHAIN_ENDPOINT", defaults.endpoint),
    )


def _build_harness_config() -> HarnessConfig:
    defaults = HarnessConfig()
    return HarnessConfig(
        skills_dir=os.getenv("RAG_SKILLS_DIR", defaults.skills_dir),
        structured_trace_enabled=parse_bool_env(
            "RAG_STRUCTURED_TRACE_ENABLED",
            defaults.structured_trace_enabled,
        ),
        structured_trace_dir=os.getenv(
            "RAG_STRUCTURED_TRACE_DIR",
            defaults.structured_trace_dir,
        ),
    )


def _build_permission_config() -> PermissionConfig:
    defaults = PermissionConfig()
    return PermissionConfig(
        allowed_index_roots=os.getenv(
            "RAG_ALLOWED_INDEX_ROOTS",
            defaults.allowed_index_roots,
        ),
        allow_local_web_fetch=parse_bool_env(
            "RAG_ALLOW_LOCAL_WEB_FETCH",
            defaults.allow_local_web_fetch,
        ),
        allow_private_web_fetch=parse_bool_env(
            "RAG_ALLOW_PRIVATE_WEB_FETCH",
            defaults.allow_private_web_fetch,
        ),
    )


def _build_job_config() -> JobConfig:
    defaults = JobConfig()
    return JobConfig(
        runtime_dir=os.getenv("RAG_JOB_RUNTIME_DIR", defaults.runtime_dir),
        max_log_chars=_parse_int(os.getenv("RAG_JOB_MAX_LOG_CHARS"), defaults.max_log_chars),
    )


def _build_runtime_config(
    session_id: str,
    checkpoint_db: str,
    resume: bool,
    interrupt_after: list[str] | tuple[str, ...] | None,
) -> RuntimeConfig:
    return RuntimeConfig(
        session_id=session_id,
        checkpoint_db=checkpoint_db,
        user_memory_path=os.getenv("RAG_USER_MEMORY_PATH", DEFAULT_USER_MEMORY_PATH),
        user_id=os.getenv("RAG_USER_ID", DEFAULT_USER_ID),
        resume=resume,
        interrupt_after=tuple(interrupt_after or ()),
    )


def build_app_config(
    kb_path: str | Path,
    *,
    session_id: str = DEFAULT_SESSION_ID,
    checkpoint_db: str = DEFAULT_CHECKPOINT_DB,
    resume: bool = False,
    interrupt_after: list[str] | tuple[str, ...] | None = None,
) -> AppConfig:
    load_env_file()
    return AppConfig(
        kb_path=Path(kb_path),
        model=_build_model_config(),
        embedding=_build_embedding_config(),
        retrieval=_build_retrieval_config(),
        metadata_rerank=_build_metadata_rerank_config(),
        web=_build_web_config(),
        scholar=_build_scholar_config(),
        context=_build_context_config(),
        generation=_build_generation_config(),
        eval_judge=_build_eval_judge_config(),
        langsmith=_build_langsmith_config(),
        harness=_build_harness_config(),
        permissions=_build_permission_config(),
        jobs=_build_job_config(),
        runtime=_build_runtime_config(
            session_id=session_id,
            checkpoint_db=checkpoint_db,
            resume=resume,
            interrupt_after=interrupt_after,
        ),
    )
