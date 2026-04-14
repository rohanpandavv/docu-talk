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
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

from config import Settings, get_settings
from schemas import (
    ChatRequest,
    ChatResponse,
    ChunkingStrategiesResponse,
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

PROMPT_TEMPLATE = """You are answering questions about a user-uploaded document.
Use only the provided context to answer the question.
If the answer is not supported by the context, say that it is not present in the indexed document.

Context:
{context}

Question:
{question}
"""


@dataclass(frozen=True, slots=True)
class RagQueryResult:
    answer: str
    document_id: str
    retrieved_documents: list[Document]


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
            "Prepared %s chunk(s) for %s using %s (%s/%s); starting embedding/indexing",
            len(chunked_documents),
            safe_filename,
            strategy.key,
            strategy.chunk_size,
            strategy.chunk_overlap,
        )

        document_id = str(uuid4())
        texts: list[str] = []
        metadatas: list[dict[str, object]] = []
        ids: list[str] = []

        for index, chunk in enumerate(chunked_documents):
            metadata = {key: value for key, value in chunk.metadata.items() if value is not None}
            metadata.update(
                {
                    "document_id": document_id,
                    "source": safe_filename,
                    "content_type": resolved_content_type,
                    "chunk_index": index,
                    "chunking_strategy": strategy.key,
                }
            )
            texts.append(chunk.page_content)
            metadatas.append(metadata)
            ids.append(f"{document_id}:{index}")

        try:
            self._get_vectorstore().add_texts(texts=texts, metadatas=metadatas, ids=ids)
            self.registry.add_document(
                document_id=document_id,
                filename=safe_filename,
                content_type=resolved_content_type,
                chunk_count=len(texts),
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
            "Indexed document %s (%s) with %s chunks",
            document_id,
            safe_filename,
            len(texts),
        )

        return UploadResponse(
            message="Document indexed successfully!",
            document_id=document_id,
            filename=safe_filename,
            chunk_count=len(texts),
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
        results = self._retrieve_documents(request.question, document_id)
        answer = self._generate_answer(request.question, document_id, results)

        self.logger.info(
            "Answered question against document %s using %s retrieved chunks",
            document_id,
            len(results),
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

    def _retrieve_documents(self, question: str, document_id: str) -> list[Document]:
        self.logger.info("Starting retrieval for document %s", document_id)

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
            raise DocumentProcessingError(
                "No indexed chunks were found for the requested document."
            )

        return results

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
            chunk_index=int(metadata.get("chunk_index", 0)),
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
