import unittest

from services.hybrid import bm25_scores, reciprocal_rank_fusion, tokenize_for_bm25


class HybridScoringTests(unittest.TestCase):
    def test_tokenize_for_bm25_normalizes_basic_terms(self):
        self.assertEqual(
            tokenize_for_bm25("BM25 + Vector_Retrieval, v2!"),
            ["bm25", "vector_retrieval", "v2"],
        )

    def test_bm25_scores_reward_keyword_overlap(self):
        scores = bm25_scores(
            "bm25 hybrid",
            [
                "dense vector retrieval only",
                "bm25 keyword retrieval",
                "hybrid bm25 and vector retrieval",
            ],
        )

        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[1], scores[0])

    def test_reciprocal_rank_fusion_boosts_consensus_candidates(self):
        fused = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["b", "a", "d"],
            ]
        )

        self.assertGreater(fused["b"], fused["c"])
        self.assertGreater(fused["a"], fused["d"])
