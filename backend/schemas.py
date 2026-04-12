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


class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    chunk_count: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    content_type: str
    chunk_count: int
    created_at: datetime
    is_active: bool = False


class DocumentListResponse(BaseModel):
    active_document_id: str | None
    documents: list[DocumentSummary]


class DocumentDeleteResponse(BaseModel):
    message: str
    document_id: str
