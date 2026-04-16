from __future__ import annotations

from collections import Counter
import math
import re

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_RRF_K = 60


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
) -> list[float]:
    if not documents:
        return []

    query_terms = tokenize_for_bm25(query)
    if not query_terms:
        return [0.0] * len(documents)

    tokenized_documents = [tokenize_for_bm25(document) for document in documents]
    document_lengths = [len(tokens) for tokens in tokenized_documents]
    average_document_length = sum(document_lengths) / len(document_lengths)
    query_term_frequencies = Counter(query_terms)

    document_frequencies: dict[str, int] = {}
    for term in query_term_frequencies:
        document_frequencies[term] = sum(1 for tokens in tokenized_documents if term in tokens)

    scores: list[float] = []
    for tokens, document_length in zip(tokenized_documents, document_lengths):
        token_frequencies = Counter(tokens)
        score = 0.0

        for term, query_term_frequency in query_term_frequencies.items():
            term_frequency = token_frequencies.get(term, 0)
            if term_frequency == 0:
                continue

            document_frequency = document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            normalization = k1 * (
                1 - b + b * (document_length / average_document_length if average_document_length else 0)
            )
            score += (
                query_term_frequency
                * inverse_document_frequency
                * ((term_frequency * (k1 + 1)) / (term_frequency + normalization))
            )

        scores.append(score)

    return scores


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, float]:
    fused_scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, candidate_id in enumerate(ranking, start=1):
            fused_scores[candidate_id] = fused_scores.get(candidate_id, 0.0) + (1.0 / (rrf_k + rank))

    return fused_scores
