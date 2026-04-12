from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean.")


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    log_level: str
    openai_api_key: str | None
    anthropic_api_key: str | None
    embedding_model: str
    chat_model: str
    chroma_directory: Path
    documents_registry_path: Path
    chunk_size: int
    chunk_overlap: int
    retrieve_k: int
    max_upload_size_bytes: int
    provider_max_retries: int
    openai_timeout_seconds: int
    anthropic_timeout_seconds: int
    chroma_anonymized_telemetry: bool

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be between 0 and CHUNK_SIZE - 1.")
        if self.retrieve_k <= 0:
            raise ValueError("RETRIEVE_K must be greater than 0.")
        if self.max_upload_size_bytes <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be greater than 0.")
        if self.provider_max_retries < 0:
            raise ValueError("PROVIDER_MAX_RETRIES must be 0 or greater.")
        if self.openai_timeout_seconds <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be greater than 0.")
        if self.anthropic_timeout_seconds <= 0:
            raise ValueError("ANTHROPIC_TIMEOUT_SECONDS must be greater than 0.")

        self.chroma_directory.mkdir(parents=True, exist_ok=True)
        self.documents_registry_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "DocuTalk API"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5-20251001"),
        chroma_directory=BASE_DIR / "chroma_db",
        documents_registry_path=BASE_DIR / "data" / "documents.json",
        chunk_size=_read_int("CHUNK_SIZE", 1000),
        chunk_overlap=_read_int("CHUNK_OVERLAP", 200),
        retrieve_k=_read_int("RETRIEVE_K", 3),
        max_upload_size_bytes=_read_int("MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024),
        provider_max_retries=_read_int("PROVIDER_MAX_RETRIES", 2),
        openai_timeout_seconds=_read_int("OPENAI_TIMEOUT_SECONDS", 30),
        anthropic_timeout_seconds=_read_int("ANTHROPIC_TIMEOUT_SECONDS", 30),
        chroma_anonymized_telemetry=_read_bool("CHROMA_ANONYMIZED_TELEMETRY", False),
    )
