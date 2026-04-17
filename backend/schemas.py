from datetime import datetime

from pydantic import BaseModel, Field, field_validator

DEFAULT_RETRIEVAL_MODE = "chunk"
SUPPORTED_RETRIEVAL_MODES = {"cag", "chunk", "hybrid", "page"}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return question

    @field_validator("retrieval_mode")
    @classmethod
    def validate_retrieval_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in SUPPORTED_RETRIEVAL_MODES:
            allowed = ", ".join(sorted(SUPPORTED_RETRIEVAL_MODES))
            raise ValueError(f"retrieval_mode must be one of: {allowed}.")
        return mode


class SourceSnippet(BaseModel):
    source_id: str
    source: str
    page: int | None = None
    chunk_index: int | None = None
    retrieval_unit: str = DEFAULT_RETRIEVAL_MODE
    excerpt: str


class CitationIssue(BaseModel):
    claim: str
    cited_source_ids: list[str] = Field(default_factory=list)
    reason: str


class CitationVerification(BaseModel):
    grounded: bool
    all_citations_valid: bool
    cited_source_ids: list[str] = Field(default_factory=list)
    missing_source_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[CitationIssue] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    document_id: str
    sources: list[SourceSnippet]
    citation_verification: CitationVerification | None = None


class ObservabilityRecord(BaseModel):
    request_id: str
    timestamp: datetime
    retrieval_mode: str
    document_id: str | None = None
    success: bool
    error_type: str | None = None
    total_latency_ms: float
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    estimated_cost_usd: float


class ObservabilitySummary(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    failure_rate: float
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    average_cost_usd: float
    total_cost_usd: float


class ObservabilityResponse(BaseModel):
    summary: ObservabilitySummary
    recent_requests: list[ObservabilityRecord] = Field(default_factory=list)
    cost_estimation_strategy: str


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
    page_count: int
    chunk_count: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    content_type: str
    page_count: int | None = None
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
