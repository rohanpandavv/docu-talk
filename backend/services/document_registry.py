from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from services.errors import DocumentNotFoundError


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._ensure_file()

    def list_documents(self) -> dict[str, Any]:
        state = self._read_state()
        active_document_id = state["active_document_id"]

        documents = []
        for metadata in state["documents"].values():
            item = dict(metadata)
            item["is_active"] = item["document_id"] == active_document_id
            documents.append(item)

        documents.sort(key=lambda item: item["created_at"], reverse=True)

        return {
            "active_document_id": active_document_id,
            "documents": documents,
        }

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        state = self._read_state()
        metadata = state["documents"].get(document_id)
        if metadata is None:
            return None

        item = dict(metadata)
        item["is_active"] = document_id == state["active_document_id"]
        return item

    def get_active_document_id(self) -> str | None:
        return self._read_state()["active_document_id"]

    def add_document(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        chunk_count: int,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict[str, Any]:
        document = {
            "document_id": document_id,
            "filename": filename,
            "content_type": content_type,
            "chunk_count": chunk_count,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with self._lock:
            state = self._read_state()
            state["documents"][document_id] = document
            state["active_document_id"] = document_id
            self._write_state(state)

        return {**document, "is_active": True}

    def activate_document(self, document_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._read_state()
            document = state["documents"].get(document_id)
            if document is None:
                raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

            state["active_document_id"] = document_id
            self._write_state(state)

        return {**document, "is_active": True}

    def delete_document(self, document_id: str) -> tuple[dict[str, Any], str | None]:
        with self._lock:
            state = self._read_state()
            document = state["documents"].pop(document_id, None)
            if document is None:
                raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

            if state["active_document_id"] == document_id:
                remaining_documents = sorted(
                    state["documents"].values(),
                    key=lambda item: item["created_at"],
                    reverse=True,
                )
                state["active_document_id"] = (
                    remaining_documents[0]["document_id"] if remaining_documents else None
                )

            self._write_state(state)

        return document, state["active_document_id"]

    def _ensure_file(self) -> None:
        if self.path.exists():
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._default_state(), indent=2),
            encoding="utf-8",
        )

    def _default_state(self) -> dict[str, Any]:
        return {
            "active_document_id": None,
            "documents": {},
        }

    def _read_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_state()

        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_state(self, state: dict[str, Any]) -> None:
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp_path.replace(self.path)
