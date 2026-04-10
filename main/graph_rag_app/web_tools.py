from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Callable, Type

from pydantic import BaseModel, Field, PrivateAttr

from .config import DEFAULT_TOP_K
from .web_types import FetchResult

try:
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover - keep imports usable without runtime deps
    class BaseTool:  # type: ignore[override]
        args_schema: Type[BaseModel] = BaseModel

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def invoke(self, input: dict[str, Any], **kwargs: Any) -> str:
            schema = self.args_schema
            if hasattr(schema, "model_validate"):
                validated = schema.model_validate(input)
                data = validated.model_dump()
            else:  # pragma: no cover - Pydantic v1 compatibility
                validated = schema.parse_obj(input)
                data = validated.dict()
            return self._run(**data)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query for public web results.")
    top_k: int | None = Field(
        None,
        ge=1,
        le=10,
        description="Maximum number of search results to return.",
    )


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the public web for recent or missing information after checking the local knowledge base."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    _backend: Any = PrivateAttr()
    _default_top_k: int = PrivateAttr(default=DEFAULT_TOP_K)

    def __init__(self, backend: Any, **kwargs: Any) -> None:
        default_top_k = int(kwargs.pop("default_top_k", DEFAULT_TOP_K))
        super().__init__(**kwargs)
        self._backend = backend
        self._default_top_k = max(1, default_top_k)

    def _run(self, query: str, top_k: int | None = None) -> str:
        resolved_top_k = self._default_top_k if top_k is None else top_k
        results = self._backend.search(query, top_k=resolved_top_k)
        payload = {
            "query": query,
            "result_count": len(results),
            "results": [asdict(item) for item in results],
        }
        debug_info = getattr(self._backend, "last_debug", None)
        if isinstance(debug_info, dict) and debug_info:
            payload["debug"] = debug_info
        return json.dumps(payload, ensure_ascii=False)


class WebFetchInput(BaseModel):
    url: str = Field(..., description="URL to fetch.")


class WebFetchTool(BaseTool):
    name: str = "web_fetch"
    description: str = "Fetch a web page and return extracted page content for grounded answers."
    args_schema: Type[BaseModel] = WebFetchInput

    _fetcher: Callable[[str], FetchResult] = PrivateAttr()

    def __init__(self, fetcher: Callable[[str], FetchResult], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fetcher = fetcher

    def _run(self, url: str) -> str:
        result = self._fetcher(url)
        return json.dumps(asdict(result), ensure_ascii=False)
