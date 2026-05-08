from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from .config import LangSmithConfig

try:
    from langsmith import Client
    from langsmith.run_helpers import trace, tracing_context
except ImportError:  # pragma: no cover - optional runtime integration
    Client = None
    trace = None
    tracing_context = None


def langsmith_tracing_enabled(config: LangSmithConfig) -> bool:
    return bool(
        config.enabled
        and config.tracing_v2
        and config.api_key
        and tracing_context is not None
        and trace is not None
    )


def build_langsmith_client(config: LangSmithConfig) -> Any | None:
    if not langsmith_tracing_enabled(config) or Client is None:
        return None
    return Client(api_key=config.api_key, api_url=config.endpoint or None)


def build_langsmith_run_name(default_run_name: str, metadata: dict[str, Any] | None = None) -> str:
    if not metadata:
        return default_run_name

    mode = str(metadata.get("mode", "") or "").strip().lower()
    dataset_name = str(metadata.get("dataset_name", "") or "").strip()
    case_id = str(metadata.get("case_id", "") or "").strip()
    if mode == "eval" and case_id:
        return f"graph_rag.eval.case:{case_id}"
    if mode == "eval_dataset" and dataset_name:
        return f"graph_rag.eval.dataset:{dataset_name}"
    return default_run_name


def build_langsmith_tags(
    metadata: dict[str, Any] | None = None,
    explicit_tags: list[str] | None = None,
) -> list[str] | None:
    raw_tags: list[str] = []
    if explicit_tags:
        raw_tags.extend(str(tag).strip() for tag in explicit_tags if str(tag).strip())
    if metadata:
        mode = str(metadata.get("mode", "") or "").strip()
        dataset_name = str(metadata.get("dataset_name", "") or "").strip()
        group = str(metadata.get("group", "") or "").strip()
        case_id = str(metadata.get("case_id", "") or "").strip()
        if mode:
            raw_tags.append(f"mode:{mode}")
        if dataset_name:
            raw_tags.append(f"dataset:{dataset_name}")
        if group:
            raw_tags.append(f"group:{group}")
        if case_id:
            raw_tags.append(f"case:{case_id}")

    tags: list[str] = []
    seen: set[str] = set()
    for tag in raw_tags:
        if tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags or None


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Any:
    original = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value:
                os.environ[key] = value
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def langsmith_run_context(
    *,
    config: LangSmithConfig,
    run_name: str,
    metadata: dict[str, Any],
    inputs: dict[str, Any],
    tags: list[str] | None = None,
) -> Any:
    if not langsmith_tracing_enabled(config):
        yield None
        return

    env_updates = {
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY": config.api_key,
        "LANGCHAIN_PROJECT": config.project_name,
    }
    if config.endpoint:
        env_updates["LANGCHAIN_ENDPOINT"] = config.endpoint

    with _temporary_env(env_updates):
        with tracing_context(
            enabled=True,
            project_name=config.project_name,
            metadata=metadata,
            tags=tags,
        ):
            with trace(
                run_name,
                run_type="chain",
                project_name=config.project_name,
                inputs=inputs,
                metadata=metadata,
                tags=tags,
            ) as run_tree:
                yield run_tree


def write_langsmith_feedback(
    *,
    config: LangSmithConfig,
    run_id: str | None,
    key: str,
    score: float | bool | None,
    value: str | dict[str, Any] | None,
    comment: str | None = None,
    source_info: dict[str, Any] | None = None,
) -> None:
    if not run_id:
        return
    client = build_langsmith_client(config)
    if client is None:
        return
    client.create_feedback(
        run_id=run_id,
        key=key,
        score=score,
        value=value,
        comment=comment,
        source_info=source_info or {},
    )
