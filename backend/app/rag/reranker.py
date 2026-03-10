"""
reranker.py
───────────
CrossEncoder reranker using ms-marco-MiniLM-L-6-v2.

Why reranking?
  Initial retrieval (FAISS + BM25) is fast but imprecise.
  A CrossEncoder takes (query, passage) pairs and produces a more
  accurate relevance score, at the cost of being slower (can't be
  used for initial retrieval). We use it as a second stage on the
  top ~15 candidates from RRF fusion.
"""

from sentence_transformers import CrossEncoder

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Singleton model loader ────────────────────────────────────────────────────

_reranker_cache: dict[str, CrossEncoder] = {}


def load_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    """Load the CrossEncoder reranker model (cached singleton)."""
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[tuple[int, str, float]],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[tuple[int, str, float]]:
    """
    Rerank candidate chunks using the CrossEncoder.

    Args:
        query:      The user's query string
        candidates: List of (chunk_index, chunk_text, initial_score) tuples
        reranker:   CrossEncoder model
        top_k:      Number of results to return after reranking

    Returns:
        List of (chunk_index, chunk_text, reranker_score) tuples,
        sorted by reranker score descending.
    """
    if not candidates:
        return []

    # Prepare (query, passage) pairs for the CrossEncoder
    pairs = [(query, chunk_text) for _, chunk_text, _ in candidates]

    # Score all pairs
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Combine with chunk metadata
    reranked = []
    for (chunk_idx, chunk_text, _), score in zip(candidates, scores):
        reranked.append((chunk_idx, chunk_text, float(score)))

    # Sort by reranker score (descending) and take top-k
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked[:top_k]
