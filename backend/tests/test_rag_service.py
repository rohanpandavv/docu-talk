import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from config import Settings
from schemas import ChatRequest
from services.errors import DocumentProcessingError
from services.rag import RagService


class FakeVectorStore:
    def __init__(self):
        self.add_calls = []
        self.search_calls = []
        self.search_results = {}
        self.get_calls = []
        self.get_results = {}

    def add_texts(self, texts, metadatas, ids):
        self.add_calls.append(
            {
                "texts": texts,
                "metadatas": metadatas,
                "ids": ids,
            }
        )

    def similarity_search(self, query, k, filter):
        self.search_calls.append({"query": query, "k": k, "filter": filter})
        return self.search_results.get(self._normalize_filter(filter), [])

    def get(self, where, include):
        self.get_calls.append({"where": where, "include": include})
        return self.get_results.get(
            self._normalize_filter(where),
            {"ids": [], "documents": [], "metadatas": []},
        )

    def delete(self, ids):
        return None

    def _normalize_filter(self, filter_value):
        if not isinstance(filter_value, dict):
            return repr(filter_value)

        if "$and" in filter_value:
            return (
                "$and",
                tuple(
                    tuple(sorted(item.items()))
                    for item in filter_value["$and"]
                ),
            )

        return tuple(sorted(filter_value.items()))


class TestableRagService(RagService):
    def __init__(self, settings, vectorstore):
        super().__init__(settings)
        self._test_vectorstore = vectorstore
        self.generated_answers = []
        self.cag_messages = []

    def _get_vectorstore(self):
        return self._test_vectorstore

    def _generate_answer(self, question, document_id, retrieved_documents):
        self.generated_answers.append(
            {
                "question": question,
                "document_id": document_id,
                "retrieved_documents": retrieved_documents,
            }
        )
        return f"Answer using {len(retrieved_documents)} retrieved unit(s)"

    def _invoke_cag_model(self, messages):
        self.cag_messages.append(messages)
        return SimpleNamespace(content="Answer using full-document context")


class RagServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.settings = Settings(
            app_name="DocuTalk Test",
            log_level="INFO",
            openai_api_key="test-openai-key",
            anthropic_api_key="test-anthropic-key",
            embedding_model="text-embedding-3-small",
            chat_model="claude-haiku-4-5-20251001",
            chroma_directory=root / "chroma",
            documents_registry_path=root / "data" / "documents.json",
            chunk_size=1000,
            chunk_overlap=200,
            retrieve_k=3,
            max_upload_size_bytes=1024 * 1024,
            provider_max_retries=0,
            openai_timeout_seconds=30,
            anthropic_timeout_seconds=30,
            chroma_anonymized_telemetry=False,
            cag_max_pages=12,
            cag_max_characters=50000,
            anthropic_prompt_cache_ttl="5m",
        )
        self.vectorstore = FakeVectorStore()
        self.service = TestableRagService(self.settings, self.vectorstore)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ingest_document_indexes_page_and_chunk_records(self):
        response = self.service.ingest_document(
            filename="demo.txt",
            content_type="text/plain",
            content=b"One short page of text.",
        )

        self.assertEqual(response.page_count, 1)
        self.assertEqual(response.chunk_count, 1)
        self.assertEqual(len(self.vectorstore.add_calls), 1)

        add_call = self.vectorstore.add_calls[0]
        retrieval_units = [metadata["retrieval_unit"] for metadata in add_call["metadatas"]]
        self.assertEqual(retrieval_units, ["page", "chunk"])
        self.assertEqual(add_call["ids"][0], f"{response.document_id}:page:1")
        self.assertEqual(add_call["ids"][1], f"{response.document_id}:chunk:0")

    def test_chat_uses_page_retrieval_mode_when_requested(self):
        upload = self.service.ingest_document(
            filename="demo.txt",
            content_type="text/plain",
            content=b"One short page of text.",
        )
        page_filter = (
            "$and",
            (
                (("document_id", upload.document_id),),
                (("retrieval_unit", "page"),),
            ),
        )
        self.vectorstore.search_results[page_filter] = [
            Document(
                page_content="Full page answer context",
                metadata={
                    "source": "demo.txt",
                    "page": 1,
                    "retrieval_unit": "page",
                },
            )
        ]

        response = self.service.chat(
            ChatRequest(
                question="Summarize the page",
                document_id=upload.document_id,
                retrieval_mode="page",
            )
        )

        self.assertEqual(
            self.vectorstore.search_calls[0]["filter"],
            {
                "$and": [
                    {"document_id": upload.document_id},
                    {"retrieval_unit": "page"},
                ]
            },
        )
        self.assertEqual(response.sources[0].retrieval_unit, "page")
        self.assertIsNone(response.sources[0].chunk_index)

    def test_chunk_retrieval_falls_back_to_legacy_documents(self):
        self.service.registry.add_document(
            document_id="doc-legacy",
            filename="legacy.txt",
            content_type="text/plain",
            page_count=1,
            chunk_count=1,
            chunking_strategy="research_paper",
            chunk_size=1000,
            chunk_overlap=200,
        )
        legacy_result = [
            Document(
                page_content="Legacy chunk context",
                metadata={
                    "source": "legacy.txt",
                    "page": 1,
                    "chunk_index": 0,
                },
            )
        ]
        self.vectorstore.search_results[(("document_id", "doc-legacy"),)] = legacy_result

        result = self.service.answer_with_context(
            ChatRequest(question="What does the legacy doc say?", document_id="doc-legacy")
        )

        self.assertEqual(len(self.vectorstore.search_calls), 2)
        self.assertEqual(
            self.vectorstore.search_calls[1]["filter"],
            {"document_id": "doc-legacy"},
        )
        self.assertEqual(result.retrieved_documents, legacy_result)

    def test_page_retrieval_requires_page_indexes(self):
        self.service.registry.add_document(
            document_id="doc-legacy",
            filename="legacy.txt",
            content_type="text/plain",
            page_count=1,
            chunk_count=1,
            chunking_strategy="research_paper",
            chunk_size=1000,
            chunk_overlap=200,
        )

        with self.assertRaises(DocumentProcessingError) as context:
            self.service.answer_with_context(
                ChatRequest(
                    question="What does the legacy doc say?",
                    document_id="doc-legacy",
                    retrieval_mode="page",
                )
            )

        self.assertIn("Re-upload it to build page indexes", str(context.exception))

    def test_cag_mode_loads_page_context_in_order(self):
        self.service.registry.add_document(
            document_id="doc-cag",
            filename="demo.txt",
            content_type="text/plain",
            page_count=2,
            chunk_count=2,
            chunking_strategy="research_paper",
            chunk_size=1000,
            chunk_overlap=200,
        )
        page_filter = (
            "$and",
            (
                (("document_id", "doc-cag"),),
                (("retrieval_unit", "page"),),
            ),
        )
        self.vectorstore.get_results[page_filter] = {
            "ids": ["doc-cag:page:2", "doc-cag:page:1"],
            "documents": ["Second page context", "First page context"],
            "metadatas": [
                {"source": "demo.txt", "page": 2, "retrieval_unit": "page"},
                {"source": "demo.txt", "page": 1, "retrieval_unit": "page"},
            ],
        }

        response = self.service.chat(
            ChatRequest(
                question="What is this document about?",
                document_id="doc-cag",
                retrieval_mode="cag",
            )
        )

        self.assertEqual(
            self.vectorstore.get_calls[0]["where"],
            {
                "$and": [
                    {"document_id": "doc-cag"},
                    {"retrieval_unit": "page"},
                ]
            },
        )
        self.assertEqual(response.answer, "Answer using full-document context")
        self.assertEqual([source.page for source in response.sources], [1, 2])
        self.assertEqual([source.retrieval_unit for source in response.sources], ["cag", "cag"])
        self.assertEqual(
            self.cag_document_text(),
            "Document context:\n[Page 1]\nFirst page context\n\n[Page 2]\nSecond page context",
        )
        self.assertEqual(
            self.service.cag_messages[0][1].content[0]["cache_control"],
            {"type": "ephemeral", "ttl": "5m"},
        )

    def test_cag_requires_page_indexes(self):
        self.service.registry.add_document(
            document_id="doc-legacy",
            filename="legacy.txt",
            content_type="text/plain",
            page_count=1,
            chunk_count=1,
            chunking_strategy="research_paper",
            chunk_size=1000,
            chunk_overlap=200,
        )

        with self.assertRaises(DocumentProcessingError) as context:
            self.service.answer_with_context(
                ChatRequest(
                    question="What does the legacy doc say?",
                    document_id="doc-legacy",
                    retrieval_mode="cag",
                )
            )

        self.assertIn("Re-upload it to build page indexes", str(context.exception))

    def test_cag_rejects_large_documents(self):
        oversized_settings = Settings(
            app_name="DocuTalk Test",
            log_level="INFO",
            openai_api_key="test-openai-key",
            anthropic_api_key="test-anthropic-key",
            embedding_model="text-embedding-3-small",
            chat_model="claude-haiku-4-5-20251001",
            chroma_directory=Path(self.temp_dir.name) / "oversized-chroma",
            documents_registry_path=Path(self.temp_dir.name) / "oversized-data" / "documents.json",
            chunk_size=1000,
            chunk_overlap=200,
            retrieve_k=3,
            max_upload_size_bytes=1024 * 1024,
            provider_max_retries=0,
            openai_timeout_seconds=30,
            anthropic_timeout_seconds=30,
            chroma_anonymized_telemetry=False,
            cag_max_pages=1,
            cag_max_characters=20,
            anthropic_prompt_cache_ttl="5m",
        )
        oversized_service = TestableRagService(oversized_settings, FakeVectorStore())
        oversized_service.registry.add_document(
            document_id="doc-big",
            filename="big.txt",
            content_type="text/plain",
            page_count=2,
            chunk_count=2,
            chunking_strategy="research_paper",
            chunk_size=1000,
            chunk_overlap=200,
        )

        with self.assertRaises(DocumentProcessingError) as context:
            oversized_service.answer_with_context(
                ChatRequest(
                    question="Summarize this document",
                    document_id="doc-big",
                    retrieval_mode="cag",
                )
            )

        self.assertIn("too large for CAG mode", str(context.exception))

    def cag_document_text(self):
        return self.service.cag_messages[0][1].content[0]["text"]
