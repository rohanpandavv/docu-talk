from __future__ import annotations

"""Lightweight in-process observability for RAG chat requests.

Cost prefers provider-reported USD when available. Most LangChain responses only
expose token usage, so we estimate cost with hardcoded model pricing constants.
That keeps the implementation simple and explicit, but it should be treated as
an estimate rather than billing-grade accounting.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import math
from threading import Lock
from typing import Any

from schemas import (
    ObservabilityRecord,
    ObservabilityResponse,
    ObservabilitySummary,
)

RECENT_REQUESTS_LIMIT = 50
COST_ESTIMATION_STRATEGY = (
    "Prefers provider-reported cost when available; otherwise estimates USD from token "
    "usage and hardcoded per-model pricing constants."
)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_million_tokens: float
    output_per_million_tokens: float
    cache_write_5m_per_million_tokens: float
    cache_write_1h_per_million_tokens: float
    cache_read_per_million_tokens: float


@dataclass(slots=True)
class UsageEstimate:
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_creation_5m_input_tokens: int = 0
    cache_creation_1h_input_tokens: int = 0
    direct_cost_usd: float | None = None


MODEL_PRICING_BY_PREFIX: dict[str, ModelPricing] = {
    "claude-haiku-4-5": ModelPricing(
        input_per_million_tokens=1.0,
        output_per_million_tokens=5.0,
        cache_write_5m_per_million_tokens=1.25,
        cache_write_1h_per_million_tokens=2.0,
        cache_read_per_million_tokens=0.10,
    ),
    "claude-sonnet-4-5": ModelPricing(
        input_per_million_tokens=3.0,
        output_per_million_tokens=15.0,
        cache_write_5m_per_million_tokens=3.75,
        cache_write_1h_per_million_tokens=6.0,
        cache_read_per_million_tokens=0.30,
    ),
    "claude-3-5-haiku": ModelPricing(
        input_per_million_tokens=0.80,
        output_per_million_tokens=4.0,
        cache_write_5m_per_million_tokens=1.0,
        cache_write_1h_per_million_tokens=1.6,
        cache_read_per_million_tokens=0.08,
    ),
    "claude-sonnet-4": ModelPricing(
        input_per_million_tokens=3.0,
        output_per_million_tokens=15.0,
        cache_write_5m_per_million_tokens=3.75,
        cache_write_1h_per_million_tokens=6.0,
        cache_read_per_million_tokens=0.30,
    ),
    "claude-sonnet-3-7": ModelPricing(
        input_per_million_tokens=3.0,
        output_per_million_tokens=15.0,
        cache_write_5m_per_million_tokens=3.75,
        cache_write_1h_per_million_tokens=6.0,
        cache_read_per_million_tokens=0.30,
    ),
    "claude-3-5-sonnet": ModelPricing(
        input_per_million_tokens=3.0,
        output_per_million_tokens=15.0,
        cache_write_5m_per_million_tokens=3.75,
        cache_write_1h_per_million_tokens=6.0,
        cache_read_per_million_tokens=0.30,
    ),
    "claude-opus-4-1": ModelPricing(
        input_per_million_tokens=15.0,
        output_per_million_tokens=75.0,
        cache_write_5m_per_million_tokens=18.75,
        cache_write_1h_per_million_tokens=30.0,
        cache_read_per_million_tokens=1.50,
    ),
    "claude-opus-4-5": ModelPricing(
        input_per_million_tokens=5.0,
        output_per_million_tokens=25.0,
        cache_write_5m_per_million_tokens=6.25,
        cache_write_1h_per_million_tokens=10.0,
        cache_read_per_million_tokens=0.50,
    ),
}


def _normalize_model_name(model_name: str | None) -> str:
    return (model_name or "").strip().lower().replace(" ", "-")


def _lookup_model_pricing(model_name: str | None) -> ModelPricing | None:
    normalized_model_name = _normalize_model_name(model_name)
    for model_prefix, pricing in MODEL_PRICING_BY_PREFIX.items():
        if normalized_model_name == model_prefix or normalized_model_name.startswith(
            f"{model_prefix}-"
        ):
            return pricing
    return None


def _coerce_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_direct_cost_usd(response_metadata: dict[str, Any]) -> float | None:
    candidates: list[object] = [
        response_metadata.get("cost_usd"),
        response_metadata.get("estimated_cost_usd"),
    ]

    usage = response_metadata.get("usage")
    if isinstance(usage, dict):
        candidates.extend(
            [
                usage.get("cost_usd"),
                usage.get("estimated_cost_usd"),
            ]
        )
        cost_payload = usage.get("cost")
        if isinstance(cost_payload, dict):
            candidates.extend([cost_payload.get("usd"), cost_payload.get("amount_usd")])

    billing = response_metadata.get("billing")
    if isinstance(billing, dict):
        candidates.extend([billing.get("cost_usd"), billing.get("estimated_cost_usd")])

    for candidate in candidates:
        direct_cost_usd = _coerce_float(candidate)
        if direct_cost_usd is not None:
            return max(direct_cost_usd, 0.0)

    return None


def extract_usage_estimate(
    response: object,
    *,
    fallback_model_name: str | None = None,
) -> UsageEstimate:
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    if not isinstance(response_metadata, dict):
        response_metadata = {}

    model_name = (
        response_metadata.get("model_name")
        or response_metadata.get("model")
        or fallback_model_name
        or ""
    )
    direct_cost_usd = _extract_direct_cost_usd(response_metadata)

    if isinstance(usage_metadata, dict) and usage_metadata:
        input_token_details = usage_metadata.get("input_token_details", {}) or {}
        if not isinstance(input_token_details, dict):
            input_token_details = {}

        return UsageEstimate(
            model_name=str(model_name),
            input_tokens=_coerce_int(usage_metadata.get("input_tokens")),
            output_tokens=_coerce_int(usage_metadata.get("output_tokens")),
            cache_read_input_tokens=_coerce_int(input_token_details.get("cache_read")),
            cache_creation_input_tokens=_coerce_int(input_token_details.get("cache_creation")),
            cache_creation_5m_input_tokens=_coerce_int(
                input_token_details.get("ephemeral_5m_input_tokens")
            ),
            cache_creation_1h_input_tokens=_coerce_int(
                input_token_details.get("ephemeral_1h_input_tokens")
            ),
            direct_cost_usd=direct_cost_usd,
        )

    raw_usage = response_metadata.get("usage") or response_metadata.get("token_usage") or {}
    if not isinstance(raw_usage, dict):
        raw_usage = {}

    return UsageEstimate(
        model_name=str(model_name),
        input_tokens=_coerce_int(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens"))),
        output_tokens=_coerce_int(
            raw_usage.get("output_tokens", raw_usage.get("completion_tokens"))
        ),
        cache_read_input_tokens=_coerce_int(
            raw_usage.get("cache_read_input_tokens", raw_usage.get("cache_read_tokens"))
        ),
        cache_creation_input_tokens=_coerce_int(
            raw_usage.get("cache_creation_input_tokens", raw_usage.get("cache_creation_tokens"))
        ),
        cache_creation_5m_input_tokens=_coerce_int(raw_usage.get("ephemeral_5m_input_tokens")),
        cache_creation_1h_input_tokens=_coerce_int(raw_usage.get("ephemeral_1h_input_tokens")),
        direct_cost_usd=direct_cost_usd,
    )


def estimate_cost_usd(
    usage: UsageEstimate,
    *,
    default_prompt_cache_ttl: str = "5m",
) -> float:
    if usage.direct_cost_usd is not None:
        return round(usage.direct_cost_usd, 6)

    pricing = _lookup_model_pricing(usage.model_name)
    if pricing is None:
        return 0.0

    cache_creation_5m_tokens = usage.cache_creation_5m_input_tokens
    cache_creation_1h_tokens = usage.cache_creation_1h_input_tokens
    unspecified_cache_creation_tokens = max(
        usage.cache_creation_input_tokens
        - cache_creation_5m_tokens
        - cache_creation_1h_tokens,
        0,
    )
    if unspecified_cache_creation_tokens:
        if default_prompt_cache_ttl == "1h":
            cache_creation_1h_tokens += unspecified_cache_creation_tokens
        else:
            cache_creation_5m_tokens += unspecified_cache_creation_tokens

    base_input_tokens = max(
        usage.input_tokens - usage.cache_read_input_tokens - usage.cache_creation_input_tokens,
        0,
    )
    estimated_cost_usd = (
        base_input_tokens * pricing.input_per_million_tokens
        + usage.output_tokens * pricing.output_per_million_tokens
        + usage.cache_read_input_tokens * pricing.cache_read_per_million_tokens
        + cache_creation_5m_tokens * pricing.cache_write_5m_per_million_tokens
        + cache_creation_1h_tokens * pricing.cache_write_1h_per_million_tokens
    ) / 1_000_000

    return round(estimated_cost_usd, 6)


class RequestCostTracker:
    def __init__(self, *, default_model_name: str, prompt_cache_ttl: str):
        self.default_model_name = default_model_name
        self.prompt_cache_ttl = prompt_cache_ttl
        self.estimated_cost_usd = 0.0

    def capture(self, response: object) -> None:
        usage = extract_usage_estimate(
            response,
            fallback_model_name=self.default_model_name,
        )
        self.estimated_cost_usd += estimate_cost_usd(
            usage,
            default_prompt_cache_ttl=self.prompt_cache_ttl,
        )
        self.estimated_cost_usd = round(self.estimated_cost_usd, 6)


class ObservabilityService:
    def __init__(self, *, recent_requests_limit: int = RECENT_REQUESTS_LIMIT):
        self.recent_requests_limit = recent_requests_limit
        self._records: list[ObservabilityRecord] = []
        self._lock = Lock()

    def record_request(
        self,
        *,
        request_id: str,
        retrieval_mode: str,
        document_id: str | None,
        success: bool,
        error_type: str | None,
        total_latency_ms: float,
        retrieval_latency_ms: float | None,
        generation_latency_ms: float | None,
        estimated_cost_usd: float,
        timestamp: datetime | None = None,
    ) -> ObservabilityRecord:
        record = ObservabilityRecord(
            request_id=request_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            retrieval_mode=retrieval_mode,
            document_id=document_id,
            success=success,
            error_type=error_type,
            total_latency_ms=round(total_latency_ms, 3),
            retrieval_latency_ms=(
                round(retrieval_latency_ms, 3)
                if retrieval_latency_ms is not None
                else None
            ),
            generation_latency_ms=(
                round(generation_latency_ms, 3)
                if generation_latency_ms is not None
                else None
            ),
            estimated_cost_usd=round(max(estimated_cost_usd, 0.0), 6),
        )
        with self._lock:
            self._records.append(record)
        return record

    def snapshot(self) -> ObservabilityResponse:
        with self._lock:
            records = list(self._records)

        successful_requests = sum(1 for record in records if record.success)
        failed_requests = len(records) - successful_requests
        total_cost_usd = round(sum(record.estimated_cost_usd for record in records), 6)
        total_requests = len(records)
        latencies = [record.total_latency_ms for record in records]

        summary = ObservabilitySummary(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            failure_rate=(
                round(failed_requests / total_requests, 6) if total_requests else 0.0
            ),
            latency_p50_ms=self._percentile(latencies, 50),
            latency_p95_ms=self._percentile(latencies, 95),
            average_cost_usd=(
                round(total_cost_usd / total_requests, 6) if total_requests else 0.0
            ),
            total_cost_usd=total_cost_usd,
        )
        recent_requests = list(reversed(records[-self.recent_requests_limit :]))

        return ObservabilityResponse(
            summary=summary,
            recent_requests=recent_requests,
            cost_estimation_strategy=COST_ESTIMATION_STRATEGY,
        )

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def _percentile(self, values: list[float], percentile: int) -> float | None:
        if not values:
            return None

        sorted_values = sorted(values)
        rank = max(1, math.ceil((percentile / 100) * len(sorted_values)))
        index = min(rank - 1, len(sorted_values) - 1)
        return round(sorted_values[index], 3)


@lru_cache
def get_observability_service() -> ObservabilityService:
    return ObservabilityService()
