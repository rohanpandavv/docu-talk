from __future__ import annotations

from dataclasses import dataclass
import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from chromadb.config import Settings as ChromaClientSettings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

from config import Settings, get_settings
from schemas import (
    ChatRequest,
    ChatResponse,
    ChunkingStrategiesResponse,
    DEFAULT_RETRIEVAL_MODE,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    SourceSnippet,
    UploadResponse,
)
from services.chunking import (
    build_text_splitter,
    list_chunking_strategies,
    resolve_chunking_strategy,
)
from services.document_registry import DocumentRegistry
from services.errors import (
    ConfigurationError,
    DocumentNotFoundError,
    DocumentProcessingError,
    NoActiveDocumentError,
    UpstreamServiceError,
)
from services.hybrid import bm25_scores, reciprocal_rank_fusion

PROMPT_TEMPLATE = """You are answering questions about a user-uploaded document.
Use only the provided context to answer the question.
If the answer is not supported by the context, say that it is not present in the indexed document.

Context:
{context}

Question:
{question}
"""

CAG_SYSTEM_PROMPT = """You are answering questions about a user-uploaded document.
The full document has been preloaded into context.
Use only the provided document text to answer the question.
If the answer is not supported by the document, say that it is not present in the indexed document.
"""

HYBRID_VECTOR_CANDIDATE_MULTIPLIER = 4
HYBRID_VECTOR_CANDIDATE_FLOOR = 10


@dataclass(frozen=True, slots=True)
class RagQueryResult:
    answer: str
    document_id: str
    retrieved_documents: list[Document]


@dataclass(frozen=True, slots=True)
class CagContext:
    full_text: str
    page_documents: list[Document]


class RagService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger("docutalk.rag")
        self.registry = DocumentRegistry(settings.documents_registry_path)
        self.prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        self._vectorstore: Chroma | None = None
        self._llm: ChatAnthropic | None = None

    def ingest_document(
        self,
        filename: str | None,
        content_type: str | None,
        content: bytes,
        chunking_strategy_key: str | None = None,
    ) -> UploadResponse:
        safe_filename = (filename or "document").strip() or "document"
        normalized_content_type = (content_type or "").lower()
        resolved_content_type = normalized_content_type or self._content_type_from_filename(
            safe_filename
        )
        strategy = resolve_chunking_strategy(self.settings, chunking_strategy_key)

        self.logger.info(
            "Starting document ingest for %s (%s bytes, content_type=%s, strategy=%s)",
            safe_filename,
            len(content),
            resolved_content_type,
            strategy.key,
        )

        if not content:
            raise DocumentProcessingError("Uploaded file is empty.")

        if len(content) > self.settings.max_upload_size_bytes:
            raise DocumentProcessingError(
                f"Uploaded file exceeds the {self.settings.max_upload_size_bytes} byte limit."
            )

        source_documents = self._extract_documents(
            filename=safe_filename,
            content_type=resolved_content_type,
            content=content,
        )
        self.logger.info(
            "Extracted %s source section(s) from %s",
            len(source_documents),
            safe_filename,
        )

        splitter = build_text_splitter(strategy)
        chunked_documents = splitter.split_documents(source_documents)
        if not chunked_documents:
            raise DocumentProcessingError("The uploaded document did not produce any chunks.")
        self.logger.info(
            "Prepared %s page record(s) and %s chunk(s) for %s using %s (%s/%s); starting embedding/indexing",
            len(source_documents),
            len(chunked_documents),
            safe_filename,
            strategy.key,
            strategy.chunk_size,
            strategy.chunk_overlap,
        )

        document_id = str(uuid4())
        page_texts, page_metadatas, page_ids = self._build_index_records(
            documents=source_documents,
            document_id=document_id,
            filename=safe_filename,
            content_type=resolved_content_type,
            chunking_strategy=strategy.key,
            retrieval_unit="page",
        )
        chunk_texts, chunk_metadatas, chunk_ids = self._build_index_records(
            documents=chunked_documents,
            document_id=document_id,
            filename=safe_filename,
            content_type=resolved_content_type,
            chunking_strategy=strategy.key,
            retrieval_unit="chunk",
        )
        texts = [*page_texts, *chunk_texts]
        metadatas = [*page_metadatas, *chunk_metadatas]
        ids = [*page_ids, *chunk_ids]

        try:
            self._get_vectorstore().add_texts(texts=texts, metadatas=metadatas, ids=ids)
            self.registry.add_document(
                document_id=document_id,
                filename=safe_filename,
                content_type=resolved_content_type,
                page_count=len(source_documents),
                chunk_count=len(chunk_texts),
                chunking_strategy=strategy.key,
                chunk_size=strategy.chunk_size,
                chunk_overlap=strategy.chunk_overlap,
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            self.logger.exception("Failed to index document %s", safe_filename)
            try:
                if ids:
                    self._get_vectorstore().delete(ids=ids)
            except Exception:
                self.logger.exception("Failed to roll back vector store write for %s", document_id)
            raise UpstreamServiceError("Failed to index the uploaded document.") from exc

        self.logger.info(
            "Indexed document %s (%s) with %s total indexed unit(s)",
            document_id,
            safe_filename,
            len(texts),
        )

        return UploadResponse(
            message="Document indexed successfully!",
            document_id=document_id,
            filename=safe_filename,
            page_count=len(source_documents),
            chunk_count=len(chunk_texts),
            chunking_strategy=strategy.key,
            chunk_size=strategy.chunk_size,
            chunk_overlap=strategy.chunk_overlap,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        result = self.answer_with_context(request)
        sources = [self._build_source_snippet(doc) for doc in result.retrieved_documents]

        return ChatResponse(
            answer=result.answer,
            document_id=result.document_id,
            sources=sources,
        )

    def answer_with_context(self, request: ChatRequest) -> RagQueryResult:
        document_id = self._resolve_document_id(request.document_id)
        if request.retrieval_mode == "cag":
            context = self._load_cag_context(document_id)
            answer = self._generate_cag_answer(request.question, document_id, context.full_text)

            self.logger.info(
                "Answered question against document %s using CAG over %s page(s)",
                document_id,
                len(context.page_documents),
            )

            return RagQueryResult(
                answer=answer,
                document_id=document_id,
                retrieved_documents=context.page_documents,
            )
        if request.retrieval_mode == "hybrid":
            results = self._retrieve_hybrid_documents(request.question, document_id)
            answer = self._generate_answer(request.question, document_id, results)

            self.logger.info(
                "Answered question against document %s using hybrid retrieval with %s chunk(s)",
                document_id,
                len(results),
            )

            return RagQueryResult(
                answer=answer,
                document_id=document_id,
                retrieved_documents=results,
            )

        results = self._retrieve_documents(
            request.question,
            document_id,
            request.retrieval_mode,
        )
        answer = self._generate_answer(request.question, document_id, results)

        self.logger.info(
            "Answered question against document %s using %s retrieved %s unit(s)",
            document_id,
            len(results),
            request.retrieval_mode,
        )

        return RagQueryResult(
            answer=answer,
            document_id=document_id,
            retrieved_documents=results,
        )

    def list_documents(self) -> DocumentListResponse:
        return DocumentListResponse(**self.registry.list_documents())

    def list_chunking_strategies(self) -> ChunkingStrategiesResponse:
        return list_chunking_strategies(self.settings)

    def activate_document(self, document_id: str) -> DocumentSummary:
        return DocumentSummary(**self.registry.activate_document(document_id))

    def delete_document(self, document_id: str) -> DocumentDeleteResponse:
        if self.registry.get_document(document_id) is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

        try:
            vectorstore = self._get_vectorstore()
            stored = vectorstore.get(where={"document_id": document_id}, include=[])
            ids = stored.get("ids", [])
            if ids:
                vectorstore.delete(ids=ids)
        except ConfigurationError:
            raise
        except Exception as exc:
            self.logger.exception("Failed to delete document %s", document_id)
            raise UpstreamServiceError("Failed to delete document embeddings.") from exc

        self.registry.delete_document(document_id)
        self.logger.info("Deleted document %s", document_id)

        return DocumentDeleteResponse(
            message="Document deleted successfully.",
            document_id=document_id,
        )

    def _resolve_document_id(self, explicit_document_id: str | None) -> str:
        document_id = explicit_document_id or self.registry.get_active_document_id()
        if not document_id:
            raise NoActiveDocumentError(
                "No active document found. Upload a document or provide document_id in the chat request."
            )

        document = self.registry.get_document(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

        return document_id

    def _retrieve_documents(
        self,
        question: str,
        document_id: str,
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    ) -> list[Document]:
        self.logger.info("Starting %s retrieval for document %s", retrieval_mode, document_id)
        primary_filter = {
            "$and": [
                {"document_id": document_id},
                {"retrieval_unit": retrieval_mode},
            ]
        }

        try:
            results = self._get_vectorstore().similarity_search(
                question,
                k=self.settings.retrieve_k,
                filter=primary_filter,
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise UpstreamServiceError("Failed to retrieve document context.") from exc

        if not results and retrieval_mode == DEFAULT_RETRIEVAL_MODE:
            self.logger.info(
                "No %s-tagged records found for %s. Falling back to legacy chunk retrieval.",
                retrieval_mode,
                document_id,
            )
            try:
                results = self._get_vectorstore().similarity_search(
                    question,
                    k=self.settings.retrieve_k,
                    filter={"document_id": document_id},
                )
            except ConfigurationError:
                raise
            except Exception as exc:
                raise UpstreamServiceError("Failed to retrieve document context.") from exc

        if not results:
            if retrieval_mode == "page":
                raise DocumentProcessingError(
                    "Page retrieval is not available for this document yet. Re-upload it to build page indexes."
                )
            raise DocumentProcessingError(
                "No indexed chunks were found for the requested document."
            )

        return results

    def _retrieve_hybrid_documents(self, question: str, document_id: str) -> list[Document]:
        chunk_documents = self._load_chunk_documents(document_id)
        if not chunk_documents:
            raise DocumentProcessingError(
                "No indexed chunks were found for the requested document."
            )

        candidate_count = min(
            len(chunk_documents),
            max(self.settings.retrieve_k * HYBRID_VECTOR_CANDIDATE_MULTIPLIER, HYBRID_VECTOR_CANDIDATE_FLOOR),
        )
        vector_candidates = self._load_hybrid_vector_candidates(
            question,
            document_id,
            candidate_count,
        )

        lexical_scores = bm25_scores(
            question,
            [document.page_content for document in chunk_documents.values()],
        )
        lexical_ranking = [
            candidate_id
            for candidate_id, score in sorted(
                zip(chunk_documents.keys(), lexical_scores, strict=False),
                key=lambda item: item[1],
                reverse=True,
            )
            if score > 0
        ]
        vector_ranking = [candidate_id for candidate_id, _ in vector_candidates]

        fused_scores = reciprocal_rank_fusion([vector_ranking, lexical_ranking])
        if not fused_scores:
            return [document for _, document in vector_candidates[: self.settings.retrieve_k]]

        ranked_candidate_ids = [
            candidate_id
            for candidate_id, _ in sorted(
                fused_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        results: list[Document] = []
        for candidate_id in ranked_candidate_ids:
            document = chunk_documents.get(candidate_id)
            if document is None:
                continue
            results.append(document)
            if len(results) == self.settings.retrieve_k:
                break

        if results:
            return results

        return [document for _, document in vector_candidates[: self.settings.retrieve_k]]

    def _load_chunk_documents(self, document_id: str) -> dict[str, Document]:
        primary_filter = {
            "$and": [
                {"document_id": document_id},
                {"retrieval_unit": "chunk"},
            ]
        }
        stored = self._load_stored_documents(
            document_id=document_id,
            primary_filter=primary_filter,
            legacy_error_message="No indexed chunks were found for the requested document.",
        )

        chunk_documents: dict[str, Document] = {}
        for text, metadata in zip(
            stored.get("documents") or [],
            stored.get("metadatas") or [],
            strict=False,
        ):
            if not isinstance(text, str) or not text.strip():
                continue
            document = Document(page_content=text, metadata=dict(metadata or {}))
            chunk_documents[self._hybrid_candidate_key(document)] = document

        return chunk_documents

    def _load_hybrid_vector_candidates(
        self,
        question: str,
        document_id: str,
        candidate_count: int,
    ) -> list[tuple[str, Document]]:
        primary_filter = {
            "$and": [
                {"document_id": document_id},
                {"retrieval_unit": "chunk"},
            ]
        }

        try:
            scored_results = self._get_vectorstore().similarity_search_with_score(
                question,
                k=candidate_count,
                filter=primary_filter,
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise UpstreamServiceError("Failed to retrieve document context.") from exc

        if not scored_results:
            self.logger.info(
                "No hybrid chunk-tagged records found for %s. Falling back to legacy chunk retrieval.",
                document_id,
            )
            try:
                scored_results = self._get_vectorstore().similarity_search_with_score(
                    question,
                    k=candidate_count,
                    filter={"document_id": document_id},
                )
            except ConfigurationError:
                raise
            except Exception as exc:
                raise UpstreamServiceError("Failed to retrieve document context.") from exc

        return [
            (self._hybrid_candidate_key(document), document)
            for document, _ in scored_results
        ]

    def _load_stored_documents(
        self,
        *,
        document_id: str,
        primary_filter: dict[str, object],
        legacy_error_message: str,
    ) -> dict[str, object]:
        try:
            stored = self._get_vectorstore().get(
                where=primary_filter,
                include=["documents", "metadatas"],
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise UpstreamServiceError("Failed to load indexed document content.") from exc

        if stored.get("documents"):
            return stored

        self.logger.info(
            "No tagged records found for %s. Falling back to legacy stored chunks.",
            document_id,
        )
        try:
            legacy_stored = self._get_vectorstore().get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise UpstreamServiceError("Failed to load indexed document content.") from exc

        if not legacy_stored.get("documents"):
            raise DocumentProcessingError(legacy_error_message)

        return legacy_stored

    def _hybrid_candidate_key(self, document: Document) -> str:
        metadata = document.metadata or {}
        return "|".join(
            [
                str(metadata.get("document_id", "")),
                str(metadata.get("source", "")),
                str(metadata.get("page", "")),
                str(metadata.get("chunk_index", "")),
            ]
        )

    def _generate_answer(
        self,
        question: str,
        document_id: str,
        retrieved_documents: list[Document],
    ) -> str:
        context = "\n\n".join(
            doc.page_content for doc in retrieved_documents if doc.page_content.strip()
        )
        if not context:
            raise DocumentProcessingError("The retrieved context was empty.")

        try:
            response = (self.prompt | self._get_llm()).invoke(
                {"context": context, "question": question}
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            self.logger.exception("Failed to generate answer for document %s", document_id)
            raise UpstreamServiceError("Failed to generate an answer from the indexed document.") from exc

        return self._normalize_answer_content(response.content)

    def _load_cag_context(self, document_id: str) -> CagContext:
        document = self.registry.get_document(document_id) or {}
        page_count = document.get("page_count")
        if page_count is not None and int(page_count) > self.settings.cag_max_pages:
            raise DocumentProcessingError(
                "This document is too large for CAG mode. Use chunk/page retrieval or raise the CAG limits."
            )

        try:
            stored = self._get_vectorstore().get(
                where={
                    "$and": [
                        {"document_id": document_id},
                        {"retrieval_unit": "page"},
                    ]
                },
                include=["documents", "metadatas"],
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise UpstreamServiceError("Failed to load full-document context for CAG mode.") from exc

        raw_documents = stored.get("documents") or []
        raw_metadatas = stored.get("metadatas") or []
        page_documents: list[Document] = []

        for text, metadata in zip(raw_documents, raw_metadatas):
            if not isinstance(text, str) or not text.strip():
                continue

            normalized_metadata = dict(metadata or {})
            normalized_metadata["retrieval_unit"] = "cag"
            page_documents.append(
                Document(
                    page_content=text,
                    metadata=normalized_metadata,
                )
            )

        page_documents.sort(key=lambda doc: int((doc.metadata or {}).get("page", 0) or 0))
        if not page_documents:
            raise DocumentProcessingError(
                "CAG mode is not available for this document yet. Re-upload it to build page indexes."
            )

        if len(page_documents) > self.settings.cag_max_pages:
            raise DocumentProcessingError(
                "This document is too large for CAG mode. Use chunk/page retrieval or raise the CAG limits."
            )

        full_text = "\n\n".join(
            self._format_page_for_cag(document)
            for document in page_documents
            if document.page_content.strip()
        )
        if len(full_text) > self.settings.cag_max_characters:
            raise DocumentProcessingError(
                "This document is too large for CAG mode. Use chunk/page retrieval or raise the CAG limits."
            )

        return CagContext(
            full_text=full_text,
            page_documents=page_documents,
        )

    def _generate_cag_answer(
        self,
        question: str,
        document_id: str,
        full_text: str,
    ) -> str:
        cached_messages = self._build_cag_messages(question, full_text, use_prompt_cache=True)

        try:
            response = self._invoke_cag_model(cached_messages)
        except ConfigurationError:
            raise
        except Exception as exc:
            self.logger.warning(
                "Prompt-cached CAG request failed for %s; retrying without prompt caching.",
                document_id,
                exc_info=exc,
            )
            uncached_messages = self._build_cag_messages(
                question,
                full_text,
                use_prompt_cache=False,
            )
            try:
                response = self._invoke_cag_model(uncached_messages)
            except ConfigurationError:
                raise
            except Exception as retry_exc:
                self.logger.exception("Failed to generate CAG answer for document %s", document_id)
                raise UpstreamServiceError(
                    "Failed to generate an answer from the full document context."
                ) from retry_exc

        return self._normalize_answer_content(response.content)

    def _build_cag_messages(
        self,
        question: str,
        full_text: str,
        *,
        use_prompt_cache: bool,
    ) -> list[SystemMessage | HumanMessage]:
        document_block: dict[str, object] = {
            "type": "text",
            "text": f"Document context:\n{full_text}",
        }
        if use_prompt_cache:
            document_block["cache_control"] = {
                "type": "ephemeral",
                "ttl": self.settings.anthropic_prompt_cache_ttl,
            }

        return [
            SystemMessage(content=CAG_SYSTEM_PROMPT),
            HumanMessage(content=[document_block]),
            HumanMessage(content=f"Question:\n{question}"),
        ]

    def _invoke_cag_model(self, messages: list[SystemMessage | HumanMessage]) -> object:
        return self._get_llm().invoke(messages)

    def _format_page_for_cag(self, document: Document) -> str:
        page_number = (document.metadata or {}).get("page")
        if page_number is None:
            return document.page_content
        return f"[Page {page_number}]\n{document.page_content}"

    def _extract_documents(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> list[Document]:
        extension = Path(filename).suffix.lower()

        if content_type == "application/pdf" or extension == ".pdf":
            return self._extract_pdf_documents(filename, content)
        if content_type == "text/plain" or extension == ".txt":
            return self._extract_text_document(filename, content)

        raise DocumentProcessingError("Unsupported file type. Only PDF and TXT files are accepted.")

    def _extract_pdf_documents(self, filename: str, content: bytes) -> list[Document]:
        try:
            pdf = PdfReader(BytesIO(content))
        except Exception as exc:
            raise DocumentProcessingError(f"Unable to read the PDF '{filename}'.") from exc

        documents = []
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={"page": page_number},
                    )
                )

        if not documents:
            raise DocumentProcessingError(
                "The PDF does not contain extractable text. Scanned PDFs are not supported yet."
            )

        return documents

    def _extract_text_document(self, filename: str, content: bytes) -> list[Document]:
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise DocumentProcessingError(
                f"The text file '{filename}' is not valid UTF-8."
            ) from exc

        if not text:
            raise DocumentProcessingError("The uploaded text file was empty.")

        return [Document(page_content=text, metadata={"page": 1})]

    def _build_index_records(
        self,
        *,
        documents: list[Document],
        document_id: str,
        filename: str,
        content_type: str,
        chunking_strategy: str,
        retrieval_unit: str,
    ) -> tuple[list[str], list[dict[str, object]], list[str]]:
        texts: list[str] = []
        metadatas: list[dict[str, object]] = []
        ids: list[str] = []

        for index, document in enumerate(documents):
            metadata = {key: value for key, value in document.metadata.items() if value is not None}
            metadata.update(
                {
                    "document_id": document_id,
                    "source": filename,
                    "content_type": content_type,
                    "chunking_strategy": chunking_strategy,
                    "retrieval_unit": retrieval_unit,
                }
            )

            if retrieval_unit == "chunk":
                metadata["chunk_index"] = index
                record_id = f"{document_id}:chunk:{index}"
            else:
                page_number = int(metadata.get("page", index + 1))
                record_id = f"{document_id}:page:{page_number}"

            texts.append(document.page_content)
            metadatas.append(metadata)
            ids.append(record_id)

        return texts, metadatas, ids

    def _content_type_from_filename(self, filename: str) -> str:
        extension = Path(filename).suffix.lower()
        if extension == ".pdf":
            return "application/pdf"
        if extension == ".txt":
            return "text/plain"
        return "application/octet-stream"

    def _get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            if not self.settings.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY is not configured.")

            embeddings = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout_seconds,
                max_retries=self.settings.provider_max_retries,
            )
            chroma_settings = ChromaClientSettings(
                anonymized_telemetry=self.settings.chroma_anonymized_telemetry,
                is_persistent=True,
                persist_directory=str(self.settings.chroma_directory),
            )
            self._vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=str(self.settings.chroma_directory),
                client_settings=chroma_settings,
            )
            self.logger.info(
                "Initialized Chroma vector store (telemetry=%s)",
                self.settings.chroma_anonymized_telemetry,
            )

        return self._vectorstore

    def _get_llm(self) -> ChatAnthropic:
        if self._llm is None:
            if not self.settings.anthropic_api_key:
                raise ConfigurationError("ANTHROPIC_API_KEY is not configured.")

            self._llm = ChatAnthropic(
                model_name=self.settings.chat_model,
                api_key=self.settings.anthropic_api_key,
                timeout=self.settings.anthropic_timeout_seconds,
                max_retries=self.settings.provider_max_retries,
            )
            self.logger.info(
                "Initialized Anthropic chat client with timeout=%ss",
                self.settings.anthropic_timeout_seconds,
            )

        return self._llm

    def _build_source_snippet(self, document: Document) -> SourceSnippet:
        excerpt = document.page_content.strip()
        if len(excerpt) > 280:
            excerpt = f"{excerpt[:277].rstrip()}..."

        metadata = document.metadata or {}
        return SourceSnippet(
            source=str(metadata.get("source", "document")),
            page=metadata.get("page"),
            chunk_index=(
                int(metadata["chunk_index"]) if metadata.get("chunk_index") is not None else None
            ),
            retrieval_unit=str(metadata.get("retrieval_unit", DEFAULT_RETRIEVAL_MODE)),
            excerpt=excerpt,
        )

    def _normalize_answer_content(self, content: object) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    text = getattr(item, "text", None)
                    if text:
                        parts.append(str(text))
            return "\n".join(parts).strip()

        return str(content).strip()


@lru_cache
def get_rag_service() -> RagService:
    return RagService(get_settings())
