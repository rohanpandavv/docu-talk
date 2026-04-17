from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from services.observability import ObservabilityService, RequestCostTracker


class ObservabilityServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = ObservabilityService(recent_requests_limit=3)
        self.timestamp = datetime(2026, 4, 18, tzinfo=timezone.utc)

    def test_metrics_recording_stores_recent_request_details(self):
        self.service.record_request(
            request_id="req-1",
            timestamp=self.timestamp,
            retrieval_mode="chunk",
            document_id="doc-1",
            success=True,
            error_type=None,
            total_latency_ms=120.0,
            retrieval_latency_ms=35.0,
            generation_latency_ms=85.0,
            estimated_cost_usd=0.0012,
        )

        snapshot = self.service.snapshot()
        self.assertEqual(snapshot.summary.total_requests, 1)
        self.assertEqual(snapshot.summary.successful_requests, 1)
        self.assertEqual(len(snapshot.recent_requests), 1)
        self.assertEqual(snapshot.recent_requests[0].request_id, "req-1")
        self.assertEqual(snapshot.recent_requests[0].document_id, "doc-1")
        self.assertEqual(snapshot.recent_requests[0].estimated_cost_usd, 0.0012)

    def test_percentiles_use_nearest_rank(self):
        for index in range(1, 21):
            self.service.record_request(
                request_id=f"req-{index}",
                timestamp=self.timestamp,
                retrieval_mode="chunk",
                document_id="doc-1",
                success=True,
                error_type=None,
                total_latency_ms=float(index * 10),
                retrieval_latency_ms=10.0,
                generation_latency_ms=float(index * 10 - 10),
                estimated_cost_usd=0.0,
            )

        summary = self.service.snapshot().summary
        self.assertEqual(summary.latency_p50_ms, 100.0)
        self.assertEqual(summary.latency_p95_ms, 190.0)

    def test_failure_rate_uses_failed_requests_over_total_requests(self):
        self.service.record_request(
            request_id="req-1",
            timestamp=self.timestamp,
            retrieval_mode="chunk",
            document_id="doc-1",
            success=True,
            error_type=None,
            total_latency_ms=100.0,
            retrieval_latency_ms=30.0,
            generation_latency_ms=70.0,
            estimated_cost_usd=0.001,
        )
        for request_id in ("req-2", "req-3"):
            self.service.record_request(
                request_id=request_id,
                timestamp=self.timestamp,
                retrieval_mode="chunk",
                document_id="doc-1",
                success=False,
                error_type="UpstreamServiceError",
                total_latency_ms=200.0,
                retrieval_latency_ms=50.0,
                generation_latency_ms=150.0,
                estimated_cost_usd=0.0,
            )

        summary = self.service.snapshot().summary
        self.assertEqual(summary.total_requests, 3)
        self.assertEqual(summary.successful_requests, 1)
        self.assertEqual(summary.failed_requests, 2)
        self.assertAlmostEqual(summary.failure_rate, 2 / 3, places=6)

    def test_cost_estimation_uses_token_pricing_when_direct_cost_is_missing(self):
        tracker = RequestCostTracker(
            default_model_name="claude-haiku-4-5-20251001",
            prompt_cache_ttl="5m",
        )
        response = SimpleNamespace(
            usage_metadata={
                "input_tokens": 1200,
                "output_tokens": 300,
                "total_tokens": 1500,
                "input_token_details": {
                    "cache_read": 200,
                    "cache_creation": 100,
                    "ephemeral_5m_input_tokens": 100,
                },
            },
            response_metadata={"model_name": "claude-haiku-4-5-20251001"},
        )

        tracker.capture(response)

        self.assertAlmostEqual(tracker.estimated_cost_usd, 0.002545, places=6)
