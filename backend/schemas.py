from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return question


class SourceSnippet(BaseModel):
    source: str
    page: int | None = None
    chunk_index: int
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    document_id: str
    sources: list[SourceSnippet]


class ChunkingStrategyOption(BaseModel):
    key: str
    label: str
    description: str
    chunk_size: int
    chunk_overlap: int


class ChunkingStrategiesResponse(BaseModel):
    default_strategy: str
    strategies: list[ChunkingStrategyOption]


class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    chunk_count: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    content_type: str
    chunk_count: int
    chunking_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    created_at: datetime
    is_active: bool = False


class DocumentListResponse(BaseModel):
    active_document_id: str | None
    documents: list[DocumentSummary]


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: str
