from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr

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


class ScholarSearchInput(BaseModel):
    topic: str = Field(..., description="Research topic to expand into Google Scholar queries.")
    count: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Maximum number of papers to return.",
    )


class ScholarSearchTool(BaseTool):
    name: str = "scholar_search"
    description: str = (
        "Search Google Scholar papers from a topic, expanding the topic into academic keywords first."
    )
    args_schema: Type[BaseModel] = ScholarSearchInput

    _searcher: Any = PrivateAttr()
    _default_count: int = PrivateAttr(default=5)

    def __init__(self, searcher: Any, **kwargs: Any) -> None:
        default_count = int(kwargs.pop("default_count", 5))
        super().__init__(**kwargs)
        self._searcher = searcher
        self._default_count = max(1, min(20, default_count))

    def _run(self, topic: str, count: int | None = None) -> str:
        resolved_count = self._default_count if count is None else count
        response = self._searcher.search(topic, resolved_count)
        return json.dumps(asdict(response), ensure_ascii=False)
