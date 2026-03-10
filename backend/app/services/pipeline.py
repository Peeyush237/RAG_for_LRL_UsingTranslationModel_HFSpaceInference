"""
pipeline.py
───────────
Orchestrates both pipelines (LinguaBridge + Traditional RAG).
Refactored from the original to use the advanced RAG retriever
and remote translation service.
"""

import time
import logging

from app.rag.retriever import Retriever, ChunkResult
from app.rag.generator import generate_answer, llm_evaluate
from app.utils.entity_guard import load_entity_dict, protect_entities, restore_entities
from app.services.translation import translate_odia_to_english, translate_english_to_odia
from app.models.schemas import PipelineResult, RetrievedChunk, DebugStep

logger = logging.getLogger(__name__)


async def run_linguabridge(
    odia_question: str,
    retriever: Retriever,
    groq_api_key: str,
    hf_spaces_url: str,
    top_k: int = 5,
) -> PipelineResult:
    """
    Full LinguaBridge pipeline:
      Odia Q → entity protect → translate to EN → entity restore
      → hybrid retrieve (EN) → Groq answer (EN) → translate to Odia
    """
    debug_steps = []
    t_start = time.time()

    # Step 1 — Original input
    debug_steps.append(DebugStep(step_name="Original Odia Input", value=odia_question))

    # Step 2 — Entity Protection
    entity_dict = load_entity_dict()
    protected_text, entity_mapping = protect_entities(odia_question, entity_dict)
    debug_steps.append(DebugStep(step_name="After Entity Protection", value=protected_text))
    if entity_mapping:
        mapping_str = ", ".join(f"{k} → {v}" for k, v in entity_mapping.items())
        debug_steps.append(DebugStep(step_name="Entity Mapping", value=mapping_str))

    # Step 3 — Odia → English translation
    translated_en = await translate_odia_to_english(protected_text, hf_spaces_url)
    debug_steps.append(DebugStep(step_name="Translated to English", value=translated_en))

    # Step 4 — Entity Restoration
    restored_en = restore_entities(translated_en, entity_mapping)
    debug_steps.append(DebugStep(step_name="After Entity Restore", value=restored_en))

    # Step 5 — Advanced hybrid retrieval
    results: list[ChunkResult] = retriever.retrieve(restored_en, top_k=top_k)
    retrieved_chunks_schema = [
        RetrievedChunk(
            text=r.text, score=r.score, source=r.source, index=r.index
        )
        for r in results
    ]

    chunks_debug = "\n".join(
        f"[Chunk {i+1} | Score: {r.score:.4f}] {r.text[:100]}..."
        for i, r in enumerate(results)
    )
    debug_steps.append(DebugStep(step_name="Retrieved Chunks (English)", value=chunks_debug))

    # Step 6 — Groq answer in English
    chunk_tuples = [(r.text, r.score) for r in results]
    en_answer = generate_answer(restored_en, chunk_tuples, groq_api_key, mode="en")
    debug_steps.append(DebugStep(step_name="English Answer from Groq", value=en_answer))

    # Step 7 — English → Odia translation
    odia_answer = await translate_english_to_odia(en_answer, hf_spaces_url)
    debug_steps.append(DebugStep(step_name="Final Odia Answer", value=odia_answer))

    elapsed = round(time.time() - t_start, 2)
    avg_score = (
        round(sum(r.score for r in results) / len(results), 4)
        if results else 0.0
    )

    return PipelineResult(
        answer=odia_answer,
        answer_en=en_answer,
        retrieved_chunks=retrieved_chunks_schema,
        debug_steps=debug_steps,
        retrieval_score=avg_score,
        response_time=elapsed,
    )


def run_traditional_rag(
    odia_question: str,
    retriever: Retriever,
    groq_api_key: str,
    top_k: int = 5,
) -> PipelineResult:
    """
    Traditional RAG pipeline — no translation, Odia query on Odia docs.
    """
    debug_steps = []
    t_start = time.time()

    debug_steps.append(DebugStep(step_name="Original Odia Input", value=odia_question))

    # Retrieve directly with Odia query
    results: list[ChunkResult] = retriever.retrieve(odia_question, top_k=top_k)
    retrieved_chunks_schema = [
        RetrievedChunk(
            text=r.text, score=r.score, source=r.source, index=r.index
        )
        for r in results
    ]

    chunks_debug = "\n".join(
        f"[Chunk {i+1} | Score: {r.score:.4f}] {r.text[:100]}..."
        for i, r in enumerate(results)
    )
    debug_steps.append(DebugStep(step_name="Retrieved Chunks (Odia)", value=chunks_debug))

    # Generate answer in Odia
    chunk_tuples = [(r.text, r.score) for r in results]
    answer = generate_answer(odia_question, chunk_tuples, groq_api_key, mode="od")
    debug_steps.append(DebugStep(step_name="Final Answer", value=answer))

    elapsed = round(time.time() - t_start, 2)
    avg_score = (
        round(sum(r.score for r in results) / len(results), 4)
        if results else 0.0
    )

    return PipelineResult(
        answer=answer,
        retrieved_chunks=retrieved_chunks_schema,
        debug_steps=debug_steps,
        retrieval_score=avg_score,
        response_time=elapsed,
    )
