"""
query.py
────────
POST /api/query — Runs the RAG pipeline(s) and returns answers.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QueryRequest, QueryResponse, PipelineMode,
)
from app.services.pipeline import run_linguabridge, run_traditional_rag
from app.rag.generator import llm_evaluate
from app.config import settings

router = APIRouter(prefix="/api", tags=["query"])


def _get_retriever(lang: str):
    """Get the retriever for the specified language from app state."""
    from app.main import app_state
    if lang == "en":
        return app_state.get("en_retriever")
    else:
        return app_state.get("od_retriever")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run RAG pipeline(s) and return the answer(s).
    Supports LinguaBridge, Traditional, or both modes.
    """
    groq_key = request.groq_api_key or settings.GROQ_API_KEY
    if not groq_key:
        raise HTTPException(status_code=400, detail="Groq API key is required.")

    hf_url = settings.HF_SPACES_URL

    response = QueryResponse()

    # Run LinguaBridge pipeline
    if request.pipeline_mode in (PipelineMode.LINGUABRIDGE, PipelineMode.BOTH):
        en_retriever = _get_retriever("en")
        if not en_retriever:
            raise HTTPException(
                status_code=400,
                detail="English knowledge base not indexed. Please ingest documents first."
            )
        if not hf_url:
            raise HTTPException(
                status_code=400,
                detail="HF_SPACES_URL not configured. Translation service unavailable."
            )

        try:
            response.linguabridge = await run_linguabridge(
                odia_question=request.question,
                retriever=en_retriever,
                groq_api_key=groq_key,
                hf_spaces_url=hf_url,
                top_k=request.top_k,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LinguaBridge pipeline error: {str(e)}")

    # Run Traditional RAG pipeline
    if request.pipeline_mode in (PipelineMode.TRADITIONAL, PipelineMode.BOTH):
        od_retriever = _get_retriever("od")
        if not od_retriever:
            raise HTTPException(
                status_code=400,
                detail="Odia knowledge base not indexed. Please ingest documents first."
            )

        try:
            response.traditional = run_traditional_rag(
                odia_question=request.question,
                retriever=od_retriever,
                groq_api_key=groq_key,
                top_k=request.top_k,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Traditional RAG error: {str(e)}")

    # Run LLM evaluation if both pipelines ran
    if response.linguabridge and response.traditional:
        try:
            response.evaluation = llm_evaluate(
                original_question_odia=request.question,
                lb_answer_odia=response.linguabridge.answer,
                trad_answer=response.traditional.answer,
                groq_api_key=groq_key,
            )
        except Exception as e:
            response.evaluation = f"Evaluation failed: {str(e)}"

    return response
