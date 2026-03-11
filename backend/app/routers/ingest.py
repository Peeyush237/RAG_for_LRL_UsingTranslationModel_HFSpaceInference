"""
ingest.py
─────────
POST /api/ingest — Upload documents, chunk, embed, and build indexes.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.schemas import IngestResponse
from app.rag.chunker import chunk_text, Chunk
from app.rag.embedder import load_embedder, build_dense_index, save_faiss_index, save_chunks
from app.rag.bm25 import BM25Index
from app.rag.reranker import load_reranker
from app.rag.retriever import Retriever
from app.utils.file_parser import parse_file
from app.config import settings

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] = File(...),
    lang: str = Form(default="en", description="Language: 'en' or 'od'"),
):
    """
    Upload PDF/TXT files, chunk them, build FAISS + BM25 indexes.
    """
    from app.main import app_state

    if lang not in ("en", "od"):
        raise HTTPException(status_code=400, detail="lang must be 'en' or 'od'")

    # Parse all files
    all_chunks: list[Chunk] = []
    for file in files:
        content = await file.read()
        try:
            text = parse_file(file.filename, content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if text:
            chunks = chunk_text(
                text=text,
                source=file.filename,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP,
            )
            all_chunks.extend(chunks)

    if not all_chunks:
        return IngestResponse(
            success=False,
            lang=lang,
            num_files=len(files),
            num_chunks=0,
            message="No text could be extracted from the uploaded files.",
        )

    # Extract text and source info
    chunk_texts = [c.text for c in all_chunks]
    chunk_sources = [c.source for c in all_chunks]

    # Use pre-loaded models from app state
    embedder = app_state.get("embedder") or load_embedder(settings.EMBEDDING_MODEL)
    reranker = app_state.get("reranker")  # May be None if ENABLE_RERANKER=False

    # Build FAISS index
    faiss_index = build_dense_index(chunk_texts, embedder)

    # Build BM25 index
    bm25_index = BM25Index()
    bm25_index.build(chunk_texts)

    # Save to disk
    if lang == "en":
        faiss_path = settings.EN_FAISS_PATH
        chunks_path = settings.EN_CHUNKS_PATH
        bm25_path = settings.EN_BM25_PATH
    else:
        faiss_path = settings.OD_FAISS_PATH
        chunks_path = settings.OD_CHUNKS_PATH
        bm25_path = settings.OD_BM25_PATH

    save_faiss_index(faiss_index, faiss_path)
    save_chunks(chunk_texts, chunks_path)
    bm25_index.save(bm25_path)

    # Create retriever and store in app state
    retriever = Retriever(
        embedder=embedder,
        reranker=reranker,
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunk_texts=chunk_texts,
        chunk_sources=chunk_sources,
    )

    if lang == "en":
        app_state["en_retriever"] = retriever
        app_state["en_chunks"] = chunk_texts
    else:
        app_state["od_retriever"] = retriever
        app_state["od_chunks"] = chunk_texts

    return IngestResponse(
        success=True,
        lang=lang,
        num_files=len(files),
        num_chunks=len(all_chunks),
        message=f"Successfully indexed {len(all_chunks)} chunks from {len(files)} file(s).",
    )
