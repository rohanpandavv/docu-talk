import unittest

from config import get_settings
from services.chunking import (
    DEFAULT_CHUNKING_STRATEGY,
    list_chunking_strategies,
    resolve_chunking_strategy,
)
from services.errors import DocumentProcessingError


class ChunkingStrategyTests(unittest.TestCase):
    def setUp(self):
        self.settings = get_settings()

    def test_default_strategy_matches_existing_research_paper_settings(self):
        strategy = resolve_chunking_strategy(self.settings, None)

        self.assertEqual(strategy.key, DEFAULT_CHUNKING_STRATEGY)
        self.assertEqual(strategy.chunk_size, self.settings.chunk_size)
        self.assertEqual(strategy.chunk_overlap, self.settings.chunk_overlap)

    def test_invalid_strategy_raises_processing_error(self):
        with self.assertRaises(DocumentProcessingError):
            resolve_chunking_strategy(self.settings, "unknown")

    def test_strategy_list_exposes_expected_presets(self):
        response = list_chunking_strategies(self.settings)

        self.assertEqual(response.default_strategy, "research_paper")
        self.assertEqual(
            [strategy.key for strategy in response.strategies],
            ["research_paper", "general_article", "notes_transcript"],
        )
