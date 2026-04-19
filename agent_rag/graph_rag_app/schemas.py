from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .config import DEFAULT_SESSION_ID, DEFAULT_TOP_K


class ConversationTurn(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    index_dir: str = Field(..., min_length=1)
    session_id: str = DEFAULT_SESSION_ID
    resume: bool = False
    history: list[ConversationTurn] = Field(default_factory=list)

    @field_validator("question", "index_dir", "session_id")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be blank.")
        return stripped


class AskResponse(BaseModel):
    answer: str
    question: str
    index_dir: str
    session_id: str
    resume: bool = False
    context_included: bool = False
    elapsed_seconds: float
    sources: list[dict[str, object]] = Field(default_factory=list)
    retrieval_debug: dict[str, object] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    index_dir: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    strategy: str = Field("hybrid", pattern="^(sparse|dense|hybrid)$")

    @field_validator("query", "index_dir")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be blank.")
        return stripped


class RetrievalResultItem(BaseModel):
    chunk_id: int
    score: float
    text: str
    document_id: str
    source_path: str
    section_title: str = ""
    strategy: str


class RetrieveResponse(BaseModel):
    query: str
    index_dir: str
    top_k: int
    strategy: str
    result_count: int
    results: list[RetrievalResultItem]


class ConfigResponse(BaseModel):
    index_dir: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
    service: str
    index_dir: str


class StatusResponse(BaseModel):
    service: str
    index_dir: str
    python_executable: str
    langgraph_available: bool
    api_key_configured: bool
    model_name: str
    web_enabled: bool
    web_search_provider: str
    scholar_enabled: bool


class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be blank.")
        return stripped


class WebSearchHit(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[WebSearchHit]
    debug: dict[str, object] = Field(default_factory=dict)


class WebFetchRequest(BaseModel):
    url: str = Field(..., min_length=1)

    @field_validator("url")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be blank.")
        return stripped


class ScholarSearchRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    count: int = Field(5, ge=1, le=20)
    index_dir: str = "agent"

    @field_validator("topic", "index_dir")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Value must not be blank.")
        return stripped


class ScholarPaper(BaseModel):
    title: str
    link: str = ""
    snippet: str = ""
    publication_info: str = ""
    cited_by: int | None = None


class ScholarSearchResponse(BaseModel):
    topic: str
    count_requested: int
    query_count: int
    paper_count: int
    papers: list[ScholarPaper]
