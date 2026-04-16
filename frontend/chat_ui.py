from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_chat_payload(
    *,
    question: str,
    active_document_id: str | None,
    retrieval_mode: str,
) -> dict[str, str]:
    payload = {
        "question": question,
        "retrieval_mode": retrieval_mode,
    }
    if active_document_id:
        payload["document_id"] = active_document_id
    return payload


def build_assistant_message(
    response_payload: Mapping[str, Any] | None,
    *,
    fallback_answer: str,
) -> dict[str, Any]:
    payload = dict(response_payload or {})
    answer = str(payload.get("answer") or fallback_answer)
    return {
        "role": "assistant",
        "content": answer,
        "sources": list(payload.get("sources") or []),
        "citation_verification": payload.get("citation_verification"),
    }


def format_source_label(source: Mapping[str, Any]) -> str:
    parts = [str(source.get("source_id", "S?")), str(source.get("source", "document"))]
    page = source.get("page")
    chunk_index = source.get("chunk_index")

    if page is not None:
        parts.append(f"page {page}")
    if chunk_index is not None:
        parts.append(f"chunk {chunk_index}")

    return " | ".join(parts)


def summarize_citation_verification(
    citation_verification: Mapping[str, Any] | None,
) -> str | None:
    if not citation_verification:
        return None

    if (
        citation_verification.get("grounded")
        and citation_verification.get("all_citations_valid")
    ):
        return "Citation check: grounded in the cited sources."

    summary_parts: list[str] = []
    missing_source_ids = citation_verification.get("missing_source_ids") or []
    unsupported_claims = citation_verification.get("unsupported_claims") or []

    if missing_source_ids:
        summary_parts.append(f"unknown source IDs: {', '.join(missing_source_ids)}")
    if unsupported_claims:
        summary_parts.append(
            f"{len(unsupported_claims)} unsupported or weakly supported claim(s)"
        )

    if not summary_parts:
        summary_parts.append("review recommended")

    return "Citation check: " + "; ".join(summary_parts) + "."
