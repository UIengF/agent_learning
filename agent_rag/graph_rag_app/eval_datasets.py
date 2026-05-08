from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class EvaluationDatasetValidationError(ValueError):
    pass


def _strip_nonblank(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("Value must not be blank.")
    return stripped


class DatasetMetadata(BaseModel):
    name: str
    version: str
    tags: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)

    @field_validator("name", "version")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        return _strip_nonblank(value)


class TrajectoryAssertions(BaseModel):
    must_use_tools: list[str] = Field(default_factory=list)
    must_not_use_tools: list[str] = Field(default_factory=list)


class SourceAssertions(BaseModel):
    min_source_count: int | None = Field(default=None, ge=0)
    required_source_types: list[str] = Field(default_factory=list)
    require_entity_coverage: bool = False


class WebSearchAssertions(BaseModel):
    required_query_terms: list[str] = Field(default_factory=list)
    min_result_count: int | None = Field(default=None, ge=0)


class RetrievalAssertions(BaseModel):
    min_result_count: int | None = Field(default=None, ge=0)
    expected_source_paths: list[str] = Field(default_factory=list)
    expected_entities: list[str] = Field(default_factory=list)


class AnswerAssertions(BaseModel):
    must_cover_points: list[str] = Field(default_factory=list)
    reference_answer: str | None = None

    @field_validator("reference_answer")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_nonblank(value)


class QuestionAssertions(BaseModel):
    expected_intent: str | None = None
    expected_focus_dimensions: list[str] = Field(default_factory=list)

    @field_validator("expected_intent")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_nonblank(value)


class EvaluationAssertions(BaseModel):
    question: QuestionAssertions | None = None
    retrieval: RetrievalAssertions | None = None
    trajectory: TrajectoryAssertions | None = None
    answer: AnswerAssertions | None = None
    sources: SourceAssertions | None = None
    source_quality: SourceAssertions | None = None
    web_search: WebSearchAssertions | None = None


class EvaluationCase(BaseModel):
    id: str
    question: str
    tags: list[str] = Field(default_factory=list)
    group: str | None = None
    expected_entities: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    assertions: EvaluationAssertions = Field(default_factory=EvaluationAssertions)

    @field_validator("id", "question")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        return _strip_nonblank(value)

    @field_validator("group", "reference_answer")
    @classmethod
    def _validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_nonblank(value)


class EvaluationDataset(BaseModel):
    metadata: DatasetMetadata
    cases: list[EvaluationCase]
    source_path: str
    applied_tags: list[str] = Field(default_factory=list)


def _parse_dataset_record(record: dict[str, Any], *, line_number: int) -> DatasetMetadata:
    try:
        return DatasetMetadata.model_validate(record["dataset"])
    except ValidationError as exc:
        raise EvaluationDatasetValidationError(
            f"Invalid dataset metadata at line {line_number}: {exc}"
        ) from exc


def _parse_case_record(record: dict[str, Any], *, line_number: int) -> EvaluationCase:
    case_id = str(record.get("id", "<unknown-case>"))
    try:
        return EvaluationCase.model_validate(record)
    except ValidationError as exc:
        raise EvaluationDatasetValidationError(
            f"Invalid evaluation case '{case_id}' at line {line_number}: {exc}"
        ) from exc


def load_evaluation_dataset(
    path: str | Path,
    *,
    include_tags: set[str] | None = None,
) -> EvaluationDataset:
    dataset_path = Path(path)
    metadata: DatasetMetadata | None = None
    cases: list[EvaluationCase] = []
    normalized_tags = sorted({tag.strip() for tag in (include_tags or set()) if tag.strip()})

    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDatasetValidationError(
                f"Invalid JSON at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise EvaluationDatasetValidationError(
                f"Invalid record at line {line_number}: expected an object."
            )
        if "dataset" in record:
            if metadata is not None:
                raise EvaluationDatasetValidationError(
                    f"Duplicate dataset metadata at line {line_number}."
                )
            metadata = _parse_dataset_record(record, line_number=line_number)
            continue
        case = _parse_case_record(record, line_number=line_number)
        if normalized_tags and not (set(case.tags) & set(normalized_tags)):
            continue
        cases.append(case)

    if metadata is None:
        raise EvaluationDatasetValidationError("Dataset metadata is required.")

    return EvaluationDataset(
        metadata=metadata,
        cases=cases,
        source_path=str(dataset_path),
        applied_tags=normalized_tags,
    )
