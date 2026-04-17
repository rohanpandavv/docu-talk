import unittest

from fastapi.testclient import TestClient

from main import app
from services.observability import get_observability_service
from services.rag import get_rag_service


class FakeRagService:
    def __init__(self):
        self.upload_calls = []
        self.chat_calls = []

    def ingest_document(self, filename, content_type, content, chunking_strategy=None):
        self.upload_calls.append((filename, content_type, content, chunking_strategy))
        return {
            "message": "Document indexed successfully!",
            "document_id": "doc-123",
            "filename": filename,
            "page_count": 1,
            "chunk_count": 1,
            "chunking_strategy": chunking_strategy or "research_paper",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        }

    def chat(self, request):
        self.chat_calls.append(request)
        return {
            "answer": f"Answer for {request.question} [S1]",
            "document_id": request.document_id or "doc-123",
            "sources": [
                {
                    "source_id": "S1",
                    "source": "demo.txt",
                    "page": 1,
                    "chunk_index": 0,
                    "retrieval_unit": request.retrieval_mode,
                    "excerpt": "demo excerpt",
                }
            ],
            "citation_verification": {
                "grounded": True,
                "all_citations_valid": True,
                "cited_source_ids": ["S1"],
                "missing_source_ids": [],
                "unsupported_claims": [],
            },
        }

    def list_documents(self):
        return {
            "active_document_id": "doc-123",
            "documents": [
                {
                    "document_id": "doc-123",
                    "filename": "demo.txt",
                    "content_type": "text/plain",
                    "page_count": 1,
                    "chunk_count": 1,
                    "chunking_strategy": "research_paper",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "created_at": "2026-04-08T00:00:00+00:00",
                    "is_active": True,
                }
            ],
        }

    def activate_document(self, document_id):
        return {
            "document_id": document_id,
            "filename": "demo.txt",
            "content_type": "text/plain",
            "page_count": 1,
            "chunk_count": 1,
            "chunking_strategy": "research_paper",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "created_at": "2026-04-08T00:00:00+00:00",
            "is_active": True,
        }

    def delete_document(self, document_id):
        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
        }

    def list_chunking_strategies(self):
        return {
            "default_strategy": "research_paper",
            "strategies": [
                {
                    "key": "research_paper",
                    "label": "Research Paper",
                    "description": "Balanced chunks for sectioned academic writing.",
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                },
                {
                    "key": "notes_transcript",
                    "label": "Notes / Transcript",
                    "description": "Smaller chunks for fast topic shifts.",
                    "chunk_size": 650,
                    "chunk_overlap": 120,
                },
            ],
        }


class FakeObservabilityService:
    def snapshot(self):
        return {
            "summary": {
                "total_requests": 4,
                "successful_requests": 3,
                "failed_requests": 1,
                "failure_rate": 0.25,
                "latency_p50_ms": 120.0,
                "latency_p95_ms": 210.0,
                "average_cost_usd": 0.00125,
                "total_cost_usd": 0.005,
            },
            "recent_requests": [
                {
                    "request_id": "req-123",
                    "timestamp": "2026-04-18T00:00:00+00:00",
                    "retrieval_mode": "chunk",
                    "document_id": "doc-123",
                    "success": True,
                    "error_type": None,
                    "total_latency_ms": 110.0,
                    "retrieval_latency_ms": 25.0,
                    "generation_latency_ms": 85.0,
                    "estimated_cost_usd": 0.001,
                }
            ],
            "cost_estimation_strategy": (
                "Prefers provider-reported cost when available; otherwise estimates USD "
                "from token usage and hardcoded per-model pricing constants."
            ),
        }


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.fake_service = FakeRagService()
        self.fake_observability = FakeObservabilityService()
        app.dependency_overrides[get_rag_service] = lambda: self.fake_service
        app.dependency_overrides[get_observability_service] = lambda: self.fake_observability
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_upload_returns_document_metadata(self):
        response = self.client.post(
            "/upload",
            data={"chunking_strategy": "notes_transcript"},
            files={"file": ("demo.txt", b"hello world", "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_id"], "doc-123")
        self.assertEqual(self.fake_service.upload_calls[0][0], "demo.txt")
        self.assertEqual(self.fake_service.upload_calls[0][3], "notes_transcript")
        self.assertEqual(response.json()["page_count"], 1)
        self.assertEqual(response.json()["chunking_strategy"], "notes_transcript")

    def test_chat_rejects_blank_question(self):
        response = self.client.post("/chat", json={"question": "   "})

        self.assertEqual(response.status_code, 422)

    def test_chat_accepts_non_default_retrieval_modes(self):
        for retrieval_mode in ("page", "cag", "hybrid"):
            response = self.client.post(
                "/chat",
                json={"question": "What is this about?", "retrieval_mode": retrieval_mode},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.fake_service.chat_calls[-1].retrieval_mode, retrieval_mode)
            self.assertEqual(response.json()["sources"][0]["retrieval_unit"], retrieval_mode)

    def test_chat_returns_source_ids_and_citation_verification(self):
        response = self.client.post(
            "/chat",
            json={"question": "What is this about?", "retrieval_mode": "chunk"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["sources"][0]["source_id"], "S1")
        self.assertEqual(payload["citation_verification"]["cited_source_ids"], ["S1"])
        self.assertTrue(payload["citation_verification"]["grounded"])

    def test_list_documents_returns_active_document(self):
        response = self.client.get("/documents")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["active_document_id"], "doc-123")
        self.assertEqual(payload["documents"][0]["filename"], "demo.txt")

    def test_activate_document_returns_document_summary(self):
        response = self.client.post("/documents/doc-456/activate")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_id"], "doc-456")

    def test_list_chunking_strategies_returns_available_options(self):
        response = self.client.get("/chunking-strategies")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["default_strategy"], "research_paper")
        self.assertEqual(payload["strategies"][0]["key"], "research_paper")

    def test_observability_returns_aggregate_metrics(self):
        response = self.client.get("/observability")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"]["total_requests"], 4)
        self.assertEqual(payload["summary"]["failure_rate"], 0.25)
        self.assertEqual(payload["summary"]["latency_p95_ms"], 210.0)
        self.assertEqual(payload["recent_requests"][0]["request_id"], "req-123")
        self.assertEqual(payload["recent_requests"][0]["retrieval_mode"], "chunk")
