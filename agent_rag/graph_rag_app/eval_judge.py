from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnswerJudgeResult:
    passed: bool
    score: float
    reasons: list[str]
    grounded: bool
    complete: bool


ANSWER_JUDGE_SCHEMA = {
    "passed": "boolean",
    "score": "number in [0,1]",
    "reasons": "array of strings",
    "grounded": "boolean",
    "complete": "boolean",
}


def build_answer_judge_prompt(
    *,
    question: str,
    expected_entities: list[str],
    must_cover_points: list[str],
    reference_answer: str,
    sources: list[dict[str, Any]],
    answer: str,
) -> str:
    return (
        "You are grading an evaluation case for a RAG agent answer.\n"
        "Return JSON only.\n"
        "Required schema keys:\n"
        f"{json.dumps(ANSWER_JUDGE_SCHEMA, ensure_ascii=False, indent=2)}\n"
        "Scoring rules:\n"
        "- grounded=true only when the answer is supported by the provided sources.\n"
        "- complete=true only when the answer covers the core comparison or requested answer points.\n"
        "- passed=true only when grounded and complete are both true.\n"
        "Question:\n"
        f"{question}\n"
        "Expected entities:\n"
        f"{json.dumps(expected_entities, ensure_ascii=False)}\n"
        "Required answer points:\n"
        f"{json.dumps(must_cover_points, ensure_ascii=False)}\n"
        "Reference answer:\n"
        f"{reference_answer}\n"
        "Available sources:\n"
        f"{json.dumps(sources, ensure_ascii=False)}\n"
        "Candidate answer:\n"
        f"{answer}\n"
    )


def parse_answer_judge_payload(raw_text: str) -> AnswerJudgeResult:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Judge payload must be a JSON object.")

    required_keys = {"passed", "score", "reasons", "grounded", "complete"}
    if not required_keys.issubset(payload):
        raise ValueError("Judge payload is missing required keys.")
    if not isinstance(payload["reasons"], list):
        raise ValueError("Judge payload field 'reasons' must be a list.")

    return AnswerJudgeResult(
        passed=bool(payload["passed"]),
        score=float(payload["score"]),
        reasons=[str(item) for item in payload["reasons"]],
        grounded=bool(payload["grounded"]),
        complete=bool(payload["complete"]),
    )
