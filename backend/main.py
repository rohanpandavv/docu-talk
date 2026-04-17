from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from config import get_settings
from logging_config import configure_logging
from schemas import (
    ChatRequest,
    ChatResponse,
    ChunkingStrategiesResponse,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    ObservabilityResponse,
    UploadResponse,
)
from services.errors import (
    ConfigurationError,
    DocumentNotFoundError,
    DocumentProcessingError,
    NoActiveDocumentError,
    ServiceError,
    UpstreamServiceError,
)
from services.observability import ObservabilityService, get_observability_service
from services.rag import RagService, get_rag_service


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name)

    @app.exception_handler(ServiceError)
    async def service_error_handler(_, exc: ServiceError):
        if isinstance(exc, DocumentNotFoundError):
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        if isinstance(exc, (DocumentProcessingError, NoActiveDocumentError)):
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        if isinstance(exc, UpstreamServiceError):
            return JSONResponse(status_code=502, content={"detail": str(exc)})
        if isinstance(exc, ConfigurationError):
            return JSONResponse(status_code=500, content={"detail": str(exc)})
        return JSONResponse(status_code=500, content={"detail": "Unexpected service failure."})

    @app.get("/documents", response_model=DocumentListResponse)
    def list_documents(rag_service: RagService = Depends(get_rag_service)):
        return rag_service.list_documents()

    @app.get("/chunking-strategies", response_model=ChunkingStrategiesResponse)
    def list_chunking_strategies(rag_service: RagService = Depends(get_rag_service)):
        return rag_service.list_chunking_strategies()

    @app.post("/documents/{document_id}/activate", response_model=DocumentSummary)
    def activate_document(document_id: str, rag_service: RagService = Depends(get_rag_service)):
        return rag_service.activate_document(document_id)

    @app.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
    def delete_document(document_id: str, rag_service: RagService = Depends(get_rag_service)):
        return rag_service.delete_document(document_id)

    @app.post("/upload", response_model=UploadResponse)
    async def upload_document(
        file: UploadFile = File(...),
        chunking_strategy: str | None = Form(None),
        rag_service: RagService = Depends(get_rag_service),
    ):
        content = await file.read()
        return await run_in_threadpool(
            rag_service.ingest_document,
            file.filename,
            file.content_type,
            content,
            chunking_strategy,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, rag_service: RagService = Depends(get_rag_service)):
        return rag_service.chat(request)

    @app.get("/observability", response_model=ObservabilityResponse)
    def observability(
        observability_service: ObservabilityService = Depends(get_observability_service),
    ):
        return observability_service.snapshot()

    return app


app = create_app()
