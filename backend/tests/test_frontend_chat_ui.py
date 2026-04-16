import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from frontend.chat_ui import (  # noqa: E402
    build_assistant_message,
    build_chat_payload,
    format_source_label,
    summarize_citation_verification,
)


class FrontendChatUiTests(unittest.TestCase):
    def test_build_chat_payload_includes_active_document_and_mode(self):
        payload = build_chat_payload(
            question="Summarize the findings",
            active_document_id="doc-123",
            retrieval_mode="page",
        )

        self.assertEqual(
            payload,
            {
                "question": "Summarize the findings",
                "document_id": "doc-123",
                "retrieval_mode": "page",
            },
        )

    def test_build_assistant_message_preserves_sources_and_verification(self):
        message = build_assistant_message(
            {
                "answer": "The results improved [S1].",
                "sources": [
                    {
                        "source_id": "S1",
                        "source": "demo.txt",
                        "page": 2,
                        "chunk_index": 4,
                        "excerpt": "Improved results are reported in the pilot.",
                    }
                ],
                "citation_verification": {
                    "grounded": True,
                    "all_citations_valid": True,
                    "cited_source_ids": ["S1"],
                    "missing_source_ids": [],
                    "unsupported_claims": [],
                },
            },
            fallback_answer="Fallback answer",
        )

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "The results improved [S1].")
        self.assertEqual(message["sources"][0]["source_id"], "S1")
        self.assertTrue(message["citation_verification"]["grounded"])

    def test_source_and_verification_helpers_format_for_display(self):
        label = format_source_label(
            {
                "source_id": "S2",
                "source": "paper.pdf",
                "page": 7,
                "chunk_index": 1,
            }
        )
        summary = summarize_citation_verification(
            {
                "grounded": False,
                "all_citations_valid": False,
                "missing_source_ids": ["S9"],
                "unsupported_claims": [{"claim": "Unsupported claim"}],
            }
        )

        self.assertEqual(label, "S2 | paper.pdf | page 7 | chunk 1")
        self.assertEqual(
            summary,
            "Citation check: unknown source IDs: S9; 1 unsupported or weakly supported claim(s).",
        )


if __name__ == "__main__":
    unittest.main()
