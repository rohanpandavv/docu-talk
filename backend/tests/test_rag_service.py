import json
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
        self.search_with_score_calls = []
        self.search_with_score_results = {}
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

    def similarity_search_with_score(self, query, k, filter):
        self.search_with_score_calls.append({"query": query, "k": k, "filter": filter})
        return self.search_with_score_results.get(self._normalize_filter(filter), [])

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
        self.answer_prompt_inputs = []
        self.cag_messages = []
        self.evaluator_prompt_inputs = []
        self.answer_content = "Answer using retrieved context [S1]"
        self.cag_answer_content = "Answer using full-document context [S1]"
        self.evaluator_content = json.dumps(
            {
                "grounded": True,
                "unsupported_claims": [],
            }
        )

    def _get_vectorstore(self):
        return self._test_vectorstore

    def _invoke_answer_prompt(self, prompt_input):
        self.answer_prompt_inputs.append(prompt_input)
        return SimpleNamespace(content=self.answer_content)

    def _invoke_cag_model(self, messages):
        self.cag_messages.append(messages)
        return SimpleNamespace(content=self.cag_answer_content)

    def _invoke_grounding_evaluator(self, prompt_input):
        self.evaluator_prompt_inputs.append(prompt_input)
        return SimpleNamespace(content=self.evaluator_content)


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
        self.assertEqual(response.answer, "Answer using retrieved context [S1]")
        self.assertEqual(response.sources[0].source_id, "S1")
        self.assertEqual(response.sources[0].retrieval_unit, "page")
        self.assertIsNone(response.sources[0].chunk_index)
        self.assertTrue(response.citation_verification.grounded)
        self.assertEqual(
            self.service.answer_prompt_inputs[0]["source_ids"],
            "S1",
        )
        self.assertIn("[S1] | demo.txt | page 1 | unit page", self.service.answer_prompt_inputs[0]["context"])

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
        self.assertEqual(response.answer, "Answer using full-document context [S1]")
        self.assertEqual([source.source_id for source in response.sources], ["S1", "S2"])
        self.assertEqual([source.page for source in response.sources], [1, 2])
        self.assertEqual([source.retrieval_unit for source in response.sources], ["cag", "cag"])
        self.assertEqual(
            self.cag_document_text(),
            (
                "Valid source IDs:\nS1, S2\n\nDocument context:\n"
                "[S1] | demo.txt | page 1 | unit cag\nFirst page context\n\n"
                "[S2] | demo.txt | page 2 | unit cag\nSecond page context"
            ),
        )
        self.assertEqual(
            self.service.cag_messages[0][1].content[0]["cache_control"],
            {"type": "ephemeral", "ttl": "5m"},
        )

    def test_hybrid_mode_combines_vector_and_lexical_signals(self):
        self.service.registry.add_document(
            document_id="doc-hybrid",
            filename="demo.txt",
            content_type="text/plain",
            page_count=1,
            chunk_count=3,
            chunking_strategy="research_paper",
            chunk_size=1000,
            chunk_overlap=200,
        )
        hybrid_filter = (
            "$and",
            (
                (("document_id", "doc-hybrid"),),
                (("retrieval_unit", "chunk"),),
            ),
        )
        self.vectorstore.get_results[hybrid_filter] = {
            "ids": ["chunk-0", "chunk-1", "chunk-2"],
            "documents": [
                "Transformer overview and architecture notes",
                "BM25 keyword matching with sparse retrieval",
                "Hybrid retrieval combines BM25 with dense vector search",
            ],
            "metadatas": [
                {
                    "document_id": "doc-hybrid",
                    "source": "demo.txt",
                    "page": 1,
                    "chunk_index": 0,
                    "retrieval_unit": "chunk",
                },
                {
                    "document_id": "doc-hybrid",
                    "source": "demo.txt",
                    "page": 1,
                    "chunk_index": 1,
                    "retrieval_unit": "chunk",
                },
                {
                    "document_id": "doc-hybrid",
                    "source": "demo.txt",
                    "page": 1,
                    "chunk_index": 2,
                    "retrieval_unit": "chunk",
                },
            ],
        }
        self.vectorstore.search_with_score_results[hybrid_filter] = [
            (
                Document(
                    page_content="Transformer overview and architecture notes",
                    metadata={
                        "document_id": "doc-hybrid",
                        "source": "demo.txt",
                        "page": 1,
                        "chunk_index": 0,
                        "retrieval_unit": "chunk",
                    },
                ),
                0.10,
            ),
            (
                Document(
                    page_content="Hybrid retrieval combines BM25 with dense vector search",
                    metadata={
                        "document_id": "doc-hybrid",
                        "source": "demo.txt",
                        "page": 1,
                        "chunk_index": 2,
                        "retrieval_unit": "chunk",
                    },
                ),
                0.20,
            ),
            (
                Document(
                    page_content="BM25 keyword matching with sparse retrieval",
                    metadata={
                        "document_id": "doc-hybrid",
                        "source": "demo.txt",
                        "page": 1,
                        "chunk_index": 1,
                        "retrieval_unit": "chunk",
                    },
                ),
                0.30,
            ),
        ]

        response = self.service.chat(
            ChatRequest(
                question="bm25 hybrid retrieval",
                document_id="doc-hybrid",
                retrieval_mode="hybrid",
            )
        )

        self.assertEqual(response.sources[0].source_id, "S1")
        self.assertEqual(response.sources[0].chunk_index, 2)
        self.assertEqual(
            self.vectorstore.search_with_score_calls[0]["filter"],
            {
                "$and": [
                    {"document_id": "doc-hybrid"},
                    {"retrieval_unit": "chunk"},
                ]
            },
        )

    def test_hybrid_mode_falls_back_to_legacy_chunk_records(self):
        self.service.registry.add_document(
            document_id="doc-legacy",
            filename="legacy.txt",
            content_type="text/plain",
            page_count=1,
            chunk_count=2,
            chunking_strategy="research_paper",
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.vectorstore.get_results[(("document_id", "doc-legacy"),)] = {
            "ids": ["legacy-0", "legacy-1"],
            "documents": [
                "Legacy dense retrieval chunk",
                "Legacy BM25 chunk with keyword hits",
            ],
            "metadatas": [
                {
                    "document_id": "doc-legacy",
                    "source": "legacy.txt",
                    "page": 1,
                    "chunk_index": 0,
                },
                {
                    "document_id": "doc-legacy",
                    "source": "legacy.txt",
                    "page": 1,
                    "chunk_index": 1,
                },
            ],
        }
        self.vectorstore.search_with_score_results[(("document_id", "doc-legacy"),)] = [
            (
                Document(
                    page_content="Legacy BM25 chunk with keyword hits",
                    metadata={
                        "document_id": "doc-legacy",
                        "source": "legacy.txt",
                        "page": 1,
                        "chunk_index": 1,
                    },
                ),
                0.2,
            )
        ]

        response = self.service.chat(
            ChatRequest(
                question="keyword hits",
                document_id="doc-legacy",
                retrieval_mode="hybrid",
            )
        )

        self.assertEqual(response.sources[0].chunk_index, 1)
        self.assertEqual(response.sources[0].source_id, "S1")

    def test_citation_verification_flags_unknown_source_ids(self):
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
                page_content="Supported page context",
                metadata={
                    "source": "demo.txt",
                    "page": 1,
                    "retrieval_unit": "page",
                },
            )
        ]
        self.service.answer_content = "This cites a missing source [S9]."

        response = self.service.chat(
            ChatRequest(
                question="What is supported?",
                document_id=upload.document_id,
                retrieval_mode="page",
            )
        )

        self.assertFalse(response.citation_verification.grounded)
        self.assertFalse(response.citation_verification.all_citations_valid)
        self.assertEqual(response.citation_verification.cited_source_ids, ["S9"])
        self.assertEqual(response.citation_verification.missing_source_ids, ["S9"])

    def test_citation_verification_surfaces_unsupported_claims(self):
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
                page_content="The document describes a pilot study but gives no accuracy figure.",
                metadata={
                    "source": "demo.txt",
                    "page": 1,
                    "retrieval_unit": "page",
                },
            )
        ]
        self.service.answer_content = "The paper reports 90% accuracy [S1]."
        self.service.evaluator_content = json.dumps(
            {
                "grounded": False,
                "unsupported_claims": [
                    {
                        "claim": "The paper reports 90% accuracy",
                        "cited_source_ids": ["S1"],
                        "reason": "Source S1 does not mention an accuracy figure.",
                    }
                ],
            }
        )

        response = self.service.chat(
            ChatRequest(
                question="What accuracy does it report?",
                document_id=upload.document_id,
                retrieval_mode="page",
            )
        )

        self.assertFalse(response.citation_verification.grounded)
        self.assertEqual(len(response.citation_verification.unsupported_claims), 1)
        self.assertEqual(
            response.citation_verification.unsupported_claims[0].cited_source_ids,
            ["S1"],
        )
        self.assertIn(
            "Question:\nWhat accuracy does it report?",
            self.service.evaluator_prompt_inputs[0]["answer"],
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
