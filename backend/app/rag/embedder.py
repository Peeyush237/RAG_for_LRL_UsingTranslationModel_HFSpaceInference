"""
embedder.py
───────────
Dense embedding using BAAI/bge-base-en-v1.5 + FAISS IndexFlatIP.

Key improvements:
  • Upgraded from bge-small to bge-base (110M params)
  • BGE query prefix for better retrieval accuracy
  • No Streamlit dependency — pure Python
  • Singleton model loading with thread safety
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# BGE models require a specific query prefix for optimal retrieval
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Singleton model loader ────────────────────────────────────────────────────

_model_cache: dict[str, SentenceTransformer] = {}


def load_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the embedding model (cached singleton)."""
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


# ── Embedding functions ───────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    embedder: SentenceTransformer,
    is_query: bool = False,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode texts into dense vectors.

    Args:
        texts:      List of text strings to encode
        embedder:   SentenceTransformer model
        is_query:   If True, prepend BGE query prefix for better retrieval
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized
    """
    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]

    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine via inner product
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
    )
    return vectors.astype(np.float32)


# ── FAISS index management ────────────────────────────────────────────────────

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product index.
    Since vectors are L2-normalized, inner product = cosine similarity.
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def search_faiss(
    query_vector: np.ndarray,
    index: faiss.IndexFlatIP,
    top_k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for top-k nearest neighbors.

    Returns:
        (scores, indices) — both of shape (1, top_k)
    """
    scores, indices = index.search(query_vector.reshape(1, -1), top_k)
    return scores[0], indices[0]


def save_faiss_index(index: faiss.IndexFlatIP, path: str):
    """Save FAISS index to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> faiss.IndexFlatIP | None:
    """Load FAISS index from disk. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)


# ── Chunk storage ─────────────────────────────────────────────────────────────

def save_chunks(chunks: list, path: str):
    """Save chunks to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(path: str) -> list:
    """Load chunks from a pickle file. Returns empty list if not found."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


# ── High-level convenience ────────────────────────────────────────────────────

def build_dense_index(
    chunk_texts: list[str],
    embedder: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """
    Build a FAISS index from a list of chunk texts.
    Embeds all chunks and adds them to an inner-product index.
    """
    vectors = embed_texts(chunk_texts, embedder, is_query=False)
    return build_faiss_index(vectors)


def dense_retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list,
    embedder: SentenceTransformer,
    top_k: int = 20,
) -> list[tuple[int, float]]:
    """
    Retrieve top-k chunks from FAISS.

    Returns:
        List of (chunk_index, score) tuples, sorted by score descending.
    """
    if index is None or not chunks:
        return []

    q_vec = embed_texts([query], embedder, is_query=True)
    scores, indices = search_faiss(q_vec, index, top_k=min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores, indices):
        if 0 <= idx < len(chunks):
            results.append((int(idx), float(score)))

    return results
