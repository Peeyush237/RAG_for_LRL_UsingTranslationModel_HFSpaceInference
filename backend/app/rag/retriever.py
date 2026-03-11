"""
retriever.py
────────────
Unified retriever that orchestrates the full advanced RAG pipeline:
  1. FAISS dense retrieval  → top-20 candidates
  2. BM25 sparse retrieval  → top-20 candidates
  3. RRF fusion             → top-15 merged candidates
  4. CrossEncoder reranking → top-K final chunks
"""

from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss

from app.rag.embedder import dense_retrieve
from app.rag.bm25 import BM25Index
from app.rag.hybrid import hybrid_search
from app.rag.reranker import rerank


@dataclass
class ChunkResult:
    """A retrieved chunk with its scores."""
    index: int
    text: str
    score: float           # final reranker score
    source: str = ""


class Retriever:
    """
    Unified retriever combining FAISS + BM25 + RRF + CrossEncoder.

    Usage:
        retriever = Retriever(embedder, reranker_model, faiss_index, bm25_index, chunks)
        results = retriever.retrieve("your query", top_k=5)
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        reranker,  # CrossEncoder or None
        faiss_index: faiss.IndexFlatIP,
        bm25_index: BM25Index,
        chunk_texts: list[str],
        chunk_sources: list[str] | None = None,
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunk_texts = chunk_texts
        self.chunk_sources = chunk_sources or [""] * len(chunk_texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        faiss_top_n: int = 20,
        bm25_top_n: int = 20,
        rrf_top_n: int = 15,
        rrf_k: int = 60,
    ) -> list[ChunkResult]:
        """
        Run the full retrieval pipeline.

        Args:
            query:       The search query
            top_k:       Final number of chunks to return (after reranking)
            faiss_top_n: Candidates from FAISS dense retrieval
            bm25_top_n:  Candidates from BM25 sparse retrieval
            rrf_top_n:   Candidates after RRF fusion (input to reranker)
            rrf_k:       RRF constant

        Returns:
            List of ChunkResult objects, sorted by relevance.
        """
        if not self.chunk_texts:
            return []

        # Step 1: FAISS dense retrieval
        faiss_results = dense_retrieve(
            query=query,
            index=self.faiss_index,
            chunks=self.chunk_texts,
            embedder=self.embedder,
            top_k=faiss_top_n,
        )

        # Step 2: BM25 sparse retrieval
        bm25_results = self.bm25_index.search(query, top_k=bm25_top_n)

        # Step 3: RRF fusion
        fused_results = hybrid_search(
            faiss_results=faiss_results,
            bm25_results=bm25_results,
            k=rrf_k,
            top_n=rrf_top_n,
        )

        # Step 4: Prepare candidates for reranking
        candidates = []
        for chunk_idx, rrf_score in fused_results:
            if 0 <= chunk_idx < len(self.chunk_texts):
                candidates.append((
                    chunk_idx,
                    self.chunk_texts[chunk_idx],
                    rrf_score,
                ))

        # Step 5: Rerank if CrossEncoder is available, otherwise use RRF scores
        if self.reranker is not None:
            reranked = rerank(
                query=query,
                candidates=candidates,
                reranker=self.reranker,
                top_k=top_k,
            )
        else:
            # Use RRF fusion scores directly
            reranked = candidates[:top_k]

        # Step 6: Build ChunkResult objects
        results = []
        for chunk_idx, chunk_text, score in reranked:
            source = self.chunk_sources[chunk_idx] if chunk_idx < len(self.chunk_sources) else ""
            results.append(ChunkResult(
                index=chunk_idx,
                text=chunk_text,
                score=score,
                source=source,
            ))

        return results

    def retrieve_dense_only(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[ChunkResult]:
        """
        Fallback: FAISS-only retrieval without BM25 or reranking.
        Useful when BM25 index isn't available.
        """
        faiss_results = dense_retrieve(
            query=query,
            index=self.faiss_index,
            chunks=self.chunk_texts,
            embedder=self.embedder,
            top_k=top_k,
        )

        results = []
        for chunk_idx, score in faiss_results:
            if 0 <= chunk_idx < len(self.chunk_texts):
                source = self.chunk_sources[chunk_idx] if chunk_idx < len(self.chunk_sources) else ""
                results.append(ChunkResult(
                    index=chunk_idx,
                    text=self.chunk_texts[chunk_idx],
                    score=score,
                    source=source,
                ))

        return results
