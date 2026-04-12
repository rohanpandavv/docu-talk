from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from schemas import ChunkingStrategiesResponse, ChunkingStrategyOption
from services.errors import DocumentProcessingError

DEFAULT_CHUNKING_STRATEGY = "research_paper"


@dataclass(frozen=True, slots=True)
class ChunkingStrategy:
    key: str
    label: str
    description: str
    chunk_size: int
    chunk_overlap: int


def _build_strategies(settings: Settings) -> dict[str, ChunkingStrategy]:
    return {
        "research_paper": ChunkingStrategy(
            key="research_paper",
            label="Research Paper",
            description="Balanced chunks for sectioned academic writing, citations, and method-heavy PDFs.",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        "general_article": ChunkingStrategy(
            key="general_article",
            label="Article / Report",
            description="Larger chunks for continuous prose like blogs, essays, and business reports.",
            chunk_size=1200,
            chunk_overlap=150,
        ),
        "notes_transcript": ChunkingStrategy(
            key="notes_transcript",
            label="Notes / Transcript",
            description="Smaller chunks for fast topic shifts, bullet points, meeting notes, or transcripts.",
            chunk_size=650,
            chunk_overlap=120,
        ),
    }


def list_chunking_strategies(settings: Settings) -> ChunkingStrategiesResponse:
    strategies = _build_strategies(settings)
    return ChunkingStrategiesResponse(
        default_strategy=DEFAULT_CHUNKING_STRATEGY,
        strategies=[
            ChunkingStrategyOption(
                key=strategy.key,
                label=strategy.label,
                description=strategy.description,
                chunk_size=strategy.chunk_size,
                chunk_overlap=strategy.chunk_overlap,
            )
            for strategy in strategies.values()
        ],
    )


def resolve_chunking_strategy(settings: Settings, strategy_key: str | None) -> ChunkingStrategy:
    strategies = _build_strategies(settings)
    normalized_key = (strategy_key or DEFAULT_CHUNKING_STRATEGY).strip() or DEFAULT_CHUNKING_STRATEGY
    strategy = strategies.get(normalized_key)
    if strategy is None:
        allowed = ", ".join(strategies.keys())
        raise DocumentProcessingError(
            f"Unsupported chunking strategy '{normalized_key}'. Supported strategies: {allowed}."
        )
    return strategy


def build_text_splitter(strategy: ChunkingStrategy) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=strategy.chunk_size,
        chunk_overlap=strategy.chunk_overlap,
    )
