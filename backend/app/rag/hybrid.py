"""
hybrid.py
─────────
Reciprocal Rank Fusion (RRF) to combine BM25 + FAISS results.

RRF is a simple, effective method to merge ranked lists from different
retrieval systems. It doesn't require score normalization — it only
uses the rank position of each result.

Formula: RRF_score(doc) = Σ  1 / (k + rank_i)
         where k = 60 (standard constant)
"""

from collections import defaultdict


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[int, float]],
    k: int = 60,
    top_n: int = 15,
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        *ranked_lists: Variable number of ranked lists.
                       Each list contains (chunk_index, score) tuples,
                       assumed to be sorted by score descending.
        k:             RRF constant (default 60, as in the original paper)
        top_n:         Number of results to return after fusion

    Returns:
        List of (chunk_index, rrf_score) tuples, sorted by RRF score descending.
    """
    rrf_scores: dict[int, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (chunk_idx, _original_score) in enumerate(ranked_list):
            # RRF formula: each result gets 1/(k + rank + 1)
            # rank is 0-based, so we add 1 to make it 1-based
            rrf_scores[chunk_idx] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return sorted_results[:top_n]


def hybrid_search(
    faiss_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60,
    top_n: int = 15,
) -> list[tuple[int, float]]:
    """
    Convenience wrapper: fuse FAISS dense + BM25 sparse results.

    Args:
        faiss_results: (chunk_index, cosine_score) from FAISS
        bm25_results:  (chunk_index, bm25_score) from BM25
        k:             RRF constant
        top_n:         Number of fused results

    Returns:
        Fused (chunk_index, rrf_score) list, sorted by score descending.
    """
    return reciprocal_rank_fusion(faiss_results, bm25_results, k=k, top_n=top_n)
