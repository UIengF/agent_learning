from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_rag_app.eval_datasets import EvaluationCase, load_evaluation_dataset  # noqa: E402
from graph_rag_app.indexing import load_index  # noqa: E402


@dataclass(frozen=True)
class RetrievalCaseMetrics:
    case_id: str
    question: str
    expected_source_paths: list[str]
    expected_entities: list[str]
    first_relevant_rank: int | None
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    reciprocal_rank_at_10: float
    ndcg_at_10: float
    entity_coverage_at_5: float
    top_sources: list[str]


def _expected_sources(case: EvaluationCase) -> list[str]:
    retrieval = case.assertions.retrieval
    if retrieval is None:
        return []
    return list(retrieval.expected_source_paths)


def _expected_entities(case: EvaluationCase) -> list[str]:
    retrieval = case.assertions.retrieval
    if retrieval is not None and retrieval.expected_entities:
        return list(retrieval.expected_entities)
    return list(case.expected_entities)


def _source_key(value: str) -> str:
    return "".join(char.lower() for char in value if char.isascii() and char.isalnum())


def _source_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    current: list[str] = []
    for char in value.lower():
        if char.isascii() and char.isalnum():
            current.append(char)
            continue
        if current:
            token = "".join(current)
            if len(token) >= 3:
                tokens.add(token)
            current = []
    if current:
        token = "".join(current)
        if len(token) >= 3:
            tokens.add(token)
    return tokens


def _source_matches(actual: str, expected_sources: set[str]) -> bool:
    if actual in expected_sources:
        return True
    actual_key = _source_key(actual)
    if any(actual_key == _source_key(expected) for expected in expected_sources):
        return True
    actual_tokens = _source_tokens(actual)
    for expected in expected_sources:
        expected_tokens = _source_tokens(expected)
        if not expected_tokens:
            continue
        overlap = len(actual_tokens & expected_tokens) / len(expected_tokens)
        if overlap >= 0.8:
            return True
    return False


def _entity_coverage(results: list[Any], entities: list[str], *, top_k: int) -> float:
    if not entities:
        return 1.0
    haystack_parts: list[str] = []
    for item in results[:top_k]:
        haystack_parts.extend(
            [
                str(getattr(item, "source_path", "")),
                str(getattr(item, "section_title", "")),
                str(getattr(item, "text", "")),
            ]
        )
    haystack = "\n".join(haystack_parts).lower()
    covered = sum(1 for entity in entities if entity.lower() in haystack)
    return covered / len(entities)


def _evaluate_case(
    case: EvaluationCase,
    *,
    retriever: Any,
    strategy: str,
    max_k: int,
) -> RetrievalCaseMetrics:
    results = list(retriever.retrieve(case.question, top_k=max_k, strategy=strategy))
    expected_sources = set(_expected_sources(case))
    expected_entities = _expected_entities(case)
    ranks = [
        index
        for index, item in enumerate(results, start=1)
        if _source_matches(str(getattr(item, "source_path", "")), expected_sources)
    ]
    first_rank = min(ranks) if ranks else None
    reciprocal_rank = 1.0 / first_rank if first_rank is not None and first_rank <= 10 else 0.0
    ndcg = 1.0 / math.log2(first_rank + 1) if first_rank is not None and first_rank <= 10 else 0.0
    top_sources = [str(getattr(item, "source_path", "")) for item in results[:5]]
    return RetrievalCaseMetrics(
        case_id=case.id,
        question=case.question,
        expected_source_paths=sorted(expected_sources),
        expected_entities=expected_entities,
        first_relevant_rank=first_rank,
        hit_at_1=first_rank == 1,
        hit_at_3=first_rank is not None and first_rank <= 3,
        hit_at_5=first_rank is not None and first_rank <= 5,
        hit_at_10=first_rank is not None and first_rank <= 10,
        reciprocal_rank_at_10=reciprocal_rank,
        ndcg_at_10=ndcg,
        entity_coverage_at_5=_entity_coverage(results, expected_entities, top_k=5),
        top_sources=top_sources,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(case_metrics: list[RetrievalCaseMetrics]) -> dict[str, Any]:
    count = len(case_metrics)
    first_ranks = [
        float(item.first_relevant_rank)
        for item in case_metrics
        if item.first_relevant_rank is not None
    ]
    return {
        "case_count": count,
        "hit_at_1": _mean([1.0 if item.hit_at_1 else 0.0 for item in case_metrics]),
        "recall_at_3": _mean([1.0 if item.hit_at_3 else 0.0 for item in case_metrics]),
        "recall_at_5": _mean([1.0 if item.hit_at_5 else 0.0 for item in case_metrics]),
        "recall_at_10": _mean([1.0 if item.hit_at_10 else 0.0 for item in case_metrics]),
        "mrr_at_10": _mean([item.reciprocal_rank_at_10 for item in case_metrics]),
        "ndcg_at_10": _mean([item.ndcg_at_10 for item in case_metrics]),
        "entity_coverage_at_5": _mean([item.entity_coverage_at_5 for item in case_metrics]),
        "average_first_relevant_rank": _mean(first_ranks),
        "missed_case_ids": [
            item.case_id for item in case_metrics if item.first_relevant_rank is None
        ],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['dataset']['name']} Retrieval Evaluation",
        "",
        f"- Dataset version: `{report['dataset']['version']}`",
        f"- Index dir: `{report['index_dir']}`",
        f"- Evaluated at: `{report['evaluated_at']}`",
        f"- Max K: `{report['max_k']}`",
        "",
        "## Metrics",
        "",
        "| Strategy | Hit@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | EntityCov@5 | Avg First Rank | Misses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, payload in report["strategies"].items():
        summary = payload["summary"]
        lines.append(
            "| {strategy} | {hit1:.3f} | {r3:.3f} | {r5:.3f} | {r10:.3f} | {mrr:.3f} | {ndcg:.3f} | {ec:.3f} | {rank:.2f} | {misses} |".format(
                strategy=strategy,
                hit1=summary["hit_at_1"],
                r3=summary["recall_at_3"],
                r5=summary["recall_at_5"],
                r10=summary["recall_at_10"],
                mrr=summary["mrr_at_10"],
                ndcg=summary["ndcg_at_10"],
                ec=summary["entity_coverage_at_5"],
                rank=summary["average_first_relevant_rank"],
                misses=len(summary["missed_case_ids"]),
            )
        )
    lines.extend(
        [
            "",
            "## Hybrid Case Details",
            "",
            "| Case | Rank | Expected Source | Top Source | EntityCov@5 |",
            "| --- | ---: | --- | --- | ---: |",
        ]
    )
    for item in report["strategies"]["hybrid"]["cases"]:
        expected = "<br>".join(item["expected_source_paths"])
        top_source = item["top_sources"][0] if item["top_sources"] else ""
        rank = item["first_relevant_rank"] if item["first_relevant_rank"] is not None else "miss"
        lines.append(
            f"| `{item['case_id']}` | {rank} | {expected} | {top_source} | {item['entity_coverage_at_5']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_evaluation(
    *,
    dataset_path: str | Path,
    index_dir: str | Path,
    output_dir: str | Path,
    strategies: list[str],
    max_k: int,
) -> dict[str, Any]:
    dataset = load_evaluation_dataset(dataset_path)
    retriever = load_index(index_dir)
    strategy_results: dict[str, Any] = {}
    for strategy in strategies:
        cases = [
            _evaluate_case(case, retriever=retriever, strategy=strategy, max_k=max_k)
            for case in dataset.cases
        ]
        strategy_results[strategy] = {
            "summary": _summarize(cases),
            "cases": [asdict(item) for item in cases],
        }

    report = {
        "dataset": dataset.metadata.model_dump(),
        "dataset_path": str(dataset_path),
        "index_dir": str(index_dir),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "max_k": max_k,
        "strategies": strategy_results,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_stem = f"{dataset.metadata.name}-report"
    report_path = output / f"{report_stem}.json"
    markdown_path = output / f"{report_stem}.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    report["report_path"] = str(report_path)
    report["markdown_path"] = str(markdown_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate offline academic retrieval quality.")
    parser.add_argument("--dataset", default="evals/datasets/academic-retrieval-30.jsonl")
    parser.add_argument("--index-dir", default="agent")
    parser.add_argument("--output-dir", default="runtime/evals/academic-retrieval-30")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--strategy",
        action="append",
        choices=["sparse", "dense", "hybrid"],
        default=None,
        help="Retrieval strategy to evaluate. Repeatable. Defaults to all strategies.",
    )
    args = parser.parse_args()
    report = run_evaluation(
        dataset_path=args.dataset,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        strategies=args.strategy or ["sparse", "dense", "hybrid"],
        max_k=args.max_k,
    )
    print(json.dumps({k: v["summary"] for k, v in report["strategies"].items()}, indent=2))
    print(f"Saved JSON report to {report['report_path']}")
    print(f"Saved Markdown report to {report['markdown_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
