from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class EvalSampleSpec(BaseModel):
    sample_id: str = Field(min_length=1)
    document_path: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    chunking_strategy: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("sample_id", "document_path", "question", "reference")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Field must not be empty.")
        return normalized


def load_eval_dataset(dataset_path: Path) -> list[EvalSampleSpec]:
    suffix = dataset_path.suffix.lower()
    raw_text = dataset_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Dataset file '{dataset_path}' is empty.")

    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(raw_text)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and "samples" in payload:
            rows = payload["samples"]
        else:
            raise ValueError(
                "JSON dataset files must contain either a top-level list or a {'samples': [...]} object."
            )
    else:
        raise ValueError("Unsupported dataset format. Use .jsonl or .json.")

    samples = [EvalSampleSpec.model_validate(row) for row in rows]

    seen_sample_ids: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id found in dataset: {sample.sample_id}")
        seen_sample_ids.add(sample.sample_id)

    return samples


def resolve_document_path(dataset_path: Path, document_path: str) -> Path:
    candidate = Path(document_path)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = (dataset_path.parent / candidate).resolve()

    if not resolved.exists():
        raise FileNotFoundError(
            f"Document path '{document_path}' does not exist relative to dataset '{dataset_path}'."
        )

    return resolved
