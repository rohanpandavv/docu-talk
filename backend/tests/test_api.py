import unittest

from fastapi.testclient import TestClient

from main import app
from services.rag import get_rag_service


class FakeRagService:
    def __init__(self):
        self.upload_calls = []

    def ingest_document(self, filename, content_type, content):
        self.upload_calls.append((filename, content_type, content))
        return {
            "message": "Document indexed successfully!",
            "document_id": "doc-123",
            "filename": filename,
            "chunk_count": 1,
        }

    def chat(self, request):
        return {
            "answer": f"Answer for {request.question}",
            "document_id": request.document_id or "doc-123",
            "sources": [
                {
                    "source": "demo.txt",
                    "page": 1,
                    "chunk_index": 0,
                    "excerpt": "demo excerpt",
                }
            ],
        }

    def list_documents(self):
        return {
            "active_document_id": "doc-123",
            "documents": [
                {
                    "document_id": "doc-123",
                    "filename": "demo.txt",
                    "content_type": "text/plain",
                    "chunk_count": 1,
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
            "chunk_count": 1,
            "created_at": "2026-04-08T00:00:00+00:00",
            "is_active": True,
        }

    def delete_document(self, document_id):
        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
        }


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.fake_service = FakeRagService()
        app.dependency_overrides[get_rag_service] = lambda: self.fake_service
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_upload_returns_document_metadata(self):
        response = self.client.post(
            "/upload",
            files={"file": ("demo.txt", b"hello world", "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_id"], "doc-123")
        self.assertEqual(self.fake_service.upload_calls[0][0], "demo.txt")

    def test_chat_rejects_blank_question(self):
        response = self.client.post("/chat", json={"question": "   "})

        self.assertEqual(response.status_code, 422)

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
