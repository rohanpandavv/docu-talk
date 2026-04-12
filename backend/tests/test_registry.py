import tempfile
import unittest
from pathlib import Path

from services.document_registry import DocumentRegistry


class DocumentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = DocumentRegistry(Path(self.temp_dir.name) / "documents.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_document_marks_latest_document_active(self):
        self.registry.add_document(
            document_id="doc-1",
            filename="first.txt",
            content_type="text/plain",
            chunk_count=2,
        )
        self.registry.add_document(
            document_id="doc-2",
            filename="second.txt",
            content_type="text/plain",
            chunk_count=4,
        )

        listing = self.registry.list_documents()

        self.assertEqual(listing["active_document_id"], "doc-2")
        self.assertEqual([doc["document_id"] for doc in listing["documents"]], ["doc-2", "doc-1"])

    def test_delete_active_document_promotes_next_document(self):
        self.registry.add_document(
            document_id="doc-1",
            filename="first.txt",
            content_type="text/plain",
            chunk_count=2,
        )
        self.registry.add_document(
            document_id="doc-2",
            filename="second.txt",
            content_type="text/plain",
            chunk_count=4,
        )

        _, active_document_id = self.registry.delete_document("doc-2")

        self.assertEqual(active_document_id, "doc-1")
        self.assertEqual(self.registry.get_active_document_id(), "doc-1")
