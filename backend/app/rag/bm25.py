"""
bm25.py
───────
BM25 sparse retrieval for keyword-based matching.

Why BM25 alongside FAISS?
  Dense embeddings capture semantic meaning but can miss exact keyword matches
  (names, numbers, technical terms). BM25 excels at these exact matches.
  Combining both via Reciprocal Rank Fusion gives the best of both worlds.
"""

import os
import re
import pickle
from rank_bm25 import BM25Okapi


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases and removes non-alphanumeric characters (except Odia script).
    """
    text = text.lower()
    # Keep word characters (includes Unicode letters for Odia)
    tokens = re.findall(r'\w+', text, re.UNICODE)
    # Filter very short tokens
    return [t for t in tokens if len(t) > 1]


# ── BM25 Index ────────────────────────────────────────────────────────────────

class BM25Index:
    """Wrapper around rank_bm25.BM25Okapi with save/load support."""

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []
        self._chunk_count: int = 0

    def build(self, chunk_texts: list[str]):
        """
        Build the BM25 index from a list of chunk texts.

        Args:
            chunk_texts: List of raw text strings
        """
        self._corpus_tokens = [_tokenize(text) for text in chunk_texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._chunk_count = len(chunk_texts)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Search the BM25 index.

        Args:
            query: Raw query string
            top_k: Number of results to return

        Returns:
            List of (chunk_index, bm25_score) tuples, sorted by score descending.
        """
        if self._bm25 is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def save(self, path: str):
        """Save the BM25 index to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "corpus_tokens": self._corpus_tokens,
                "chunk_count": self._chunk_count,
            }, f)

    def load(self, path: str) -> bool:
        """
        Load the BM25 index from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._corpus_tokens = data["corpus_tokens"]
        self._chunk_count = data["chunk_count"]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        return True

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def chunk_count(self) -> int:
        return self._chunk_count
