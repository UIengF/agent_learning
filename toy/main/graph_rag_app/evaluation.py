from __future__ import annotations

import math
from typing import Any, Iterable

from .config import DEFAULT_TOP_K
from .retrieval import Retriever


def hit_rate_at_k(relevances: list[int]) -> float:
    return 1.0 if any(relevances) else 0.0


def reciprocal_rank(relevances: list[int]) -> float:
    for index, value in enumerate(relevances, start=1):
        if value:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevances: list[int]) -> float:
    dcg = 0.0
    for index, value in enumerate(relevances, start=1):
        if value:
            dcg += value / math.log2(index + 1)
    ideal = sorted(relevances, reverse=True)
    idcg = 0.0
    for index, value in enumerate(ideal, start=1):
        if value:
            idcg += value / math.log2(index + 1)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval_cases(
    retriever: Retriever,
    cases: Iterable[dict[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
    strategy: str = "hybrid",
) -> dict[str, Any]:
    case_list = list(cases)
    reports = []
    for case in case_list:
        results = retriever.retrieve(case["query"], top_k=top_k, strategy=strategy)
        relevant_paths = set(case.get("relevant_source_paths", []))
        relevances = [1 if item.source_path in relevant_paths else 0 for item in results]
        reports.append(
            {
                "query": case["query"],
                "relevances": relevances,
                "hit_rate_at_k": hit_rate_at_k(relevances),
                "reciprocal_rank": reciprocal_rank(relevances),
                "ndcg_at_k": ndcg_at_k(relevances),
            }
        )

    if not reports:
        return {
            "case_count": 0,
            "hit_rate_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "cases": [],
        }

    return {
        "case_count": len(reports),
        "hit_rate_at_k": sum(item["hit_rate_at_k"] for item in reports) / len(reports),
        "mrr": sum(item["reciprocal_rank"] for item in reports) / len(reports),
        "ndcg_at_k": sum(item["ndcg_at_k"] for item in reports) / len(reports),
        "cases": reports,
    }
