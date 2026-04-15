import unittest
from collections import Counter

from config import BASE_DIR
from evals.dataset import load_eval_dataset, resolve_document_path


class BenchmarkDatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset_path = BASE_DIR / "evals" / "datasets" / "docutalk_benchmark_v1.jsonl"
        self.allowed_strategies = {
            "research_paper",
            "general_article",
            "notes_transcript",
        }

    def test_benchmark_dataset_resolves_documents_and_covers_all_strategies(self):
        samples = load_eval_dataset(self.dataset_path)

        self.assertGreaterEqual(len(samples), 12)
        self.assertEqual(
            {sample.chunking_strategy for sample in samples},
            self.allowed_strategies,
        )

        for sample in samples:
            self.assertIn(sample.chunking_strategy, self.allowed_strategies)
            resolved_path = resolve_document_path(self.dataset_path, sample.document_path)
            self.assertTrue(resolved_path.is_file())
            self.assertTrue(resolved_path.read_text(encoding="utf-8").strip())
            self.assertIn("benchmark_v1", sample.tags)

    def test_benchmark_dataset_includes_cross_strategy_comparisons(self):
        samples = load_eval_dataset(self.dataset_path)
        pair_counts = Counter((sample.document_path, sample.question) for sample in samples)
        duplicated_pairs = [pair for pair, count in pair_counts.items() if count > 1]

        self.assertGreaterEqual(len(duplicated_pairs), 3)

        for document_path, question in duplicated_pairs:
            strategies = {
                sample.chunking_strategy
                for sample in samples
                if sample.document_path == document_path and sample.question == question
            }
            self.assertGreater(len(strategies), 1)


if __name__ == "__main__":
    unittest.main()
