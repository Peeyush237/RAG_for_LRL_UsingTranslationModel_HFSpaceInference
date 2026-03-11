"""
main.py
───────
FastAPI application entry point for the LinguaBridge backend.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import HealthResponse
from app.routers import query, ingest, translate
from app.rag.embedder import load_embedder, load_faiss_index, load_chunks
from app.rag.bm25 import BM25Index
from app.rag.reranker import load_reranker
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# ── App State ─────────────────────────────────────────────────────────────────
# Mutable shared state for the application (retrievers, chunks, etc.)
app_state: dict = {}


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indexes on startup."""
    logger.info("🚀 Starting LinguaBridge backend...")

    # Set HF token for authenticated downloads (avoids rate limits)
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN
        logger.info("✅ HuggingFace token configured")

    # Ensure data directory exists
    os.makedirs(settings.DATA_DIR, exist_ok=True)

    # Load embedding model
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    embedder = load_embedder(settings.EMBEDDING_MODEL)
    app_state["embedder"] = embedder

    # Load reranker model (optional — disabled on free tier to save memory)
    reranker = None
    if settings.ENABLE_RERANKER:
        logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
        reranker = load_reranker(settings.RERANKER_MODEL)
        app_state["reranker"] = reranker
    else:
        logger.info("ℹ️ Reranker disabled (ENABLE_RERANKER=false)")

    # Try to load saved English index
    en_faiss = load_faiss_index(settings.EN_FAISS_PATH)
    en_chunks = load_chunks(settings.EN_CHUNKS_PATH)
    if en_faiss and en_chunks:
        en_bm25 = BM25Index()
        if not en_bm25.load(settings.EN_BM25_PATH):
            # Rebuild BM25 from chunks if saved index not found
            en_bm25.build(en_chunks)
        app_state["en_retriever"] = Retriever(
            embedder=embedder,
            reranker=reranker,
            faiss_index=en_faiss,
            bm25_index=en_bm25,
            chunk_texts=en_chunks,
        )
        app_state["en_chunks"] = en_chunks
        logger.info(f"✅ English index loaded: {len(en_chunks)} chunks")

    # Try to load saved Odia index
    od_faiss = load_faiss_index(settings.OD_FAISS_PATH)
    od_chunks = load_chunks(settings.OD_CHUNKS_PATH)
    if od_faiss and od_chunks:
        od_bm25 = BM25Index()
        if not od_bm25.load(settings.OD_BM25_PATH):
            od_bm25.build(od_chunks)
        app_state["od_retriever"] = Retriever(
            embedder=embedder,
            reranker=reranker,
            faiss_index=od_faiss,
            bm25_index=od_bm25,
            chunk_texts=od_chunks,
        )
        app_state["od_chunks"] = od_chunks
        logger.info(f"✅ Odia index loaded: {len(od_chunks)} chunks")

    logger.info("✅ LinguaBridge backend ready!")
    yield

    # Cleanup
    app_state.clear()
    logger.info("Shutting down LinguaBridge backend.")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LinguaBridge API",
    description="RAG-powered QA system for low-resource languages (Odia ↔ English)",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(translate.router)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status."""
    return HealthResponse(
        status="ok",
        embedder_loaded="embedder" in app_state,
        reranker_loaded="reranker" in app_state,
        en_index_loaded="en_retriever" in app_state,
        od_index_loaded="od_retriever" in app_state,
        en_chunks_count=len(app_state.get("en_chunks", [])),
        od_chunks_count=len(app_state.get("od_chunks", [])),
    )


@app.get("/")
async def root():
    """Root endpoint — redirect to docs."""
    return {
        "name": "LinguaBridge API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
