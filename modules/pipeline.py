"""
pipeline.py
───────────
Orchestrates both end-to-end pipelines and returns structured results
including all intermediate steps for the debug panel.

LinguaBridge pipeline:
  Odia Q → entity protect → Odia→EN translate → entity restore
  → FAISS retrieve (EN) → Groq answer (EN) → EN→Odia translate → Odia answer

Traditional RAG pipeline:
  Odia Q → FAISS retrieve (Odia) → Groq answer (Odia/EN) → answer
"""

import time
from modules.entity_guard import load_entity_dict, protect_entities, restore_entities
from modules.translator import odia_to_english, english_to_odia
from modules.rag import retrieve, generate_answer


def run_linguabridge(
    odia_question: str,
    en_index,
    en_chunks: list,
    embedder,
    translator_components: dict,
    groq_api_key: str,
    top_k: int = 3,
) -> dict:
    """
    Full LinguaBridge pipeline.
    Returns a dict with the final answer + all intermediate steps.
    """
    debug_steps = {}
    t_start = time.time()

    # Step 1 — Original input
    debug_steps["1_original_odia"] = odia_question

    # Step 2 — Entity Protection
    entity_dict = load_entity_dict()
    protected_text, entity_mapping = protect_entities(odia_question, entity_dict)
    debug_steps["2_after_entity_protect"] = protected_text
    debug_steps["2_entity_mapping"] = entity_mapping

    # Step 3 — Odia → English translation
    translated_en = odia_to_english(protected_text, translator_components)
    debug_steps["3_translated_to_english"] = translated_en

    # Step 4 — Entity Restoration
    restored_en = restore_entities(translated_en, entity_mapping)
    debug_steps["4_after_entity_restore"] = restored_en

    # Step 5 — FAISS retrieval on English index
    retrieved = retrieve(restored_en, en_index, en_chunks, embedder, top_k=top_k)
    debug_steps["5_retrieved_chunks"] = retrieved   # list of (chunk, score)

    # Step 6 — Groq answer in English
    en_answer = generate_answer(restored_en, retrieved, groq_api_key, mode="en")
    debug_steps["6_english_answer"] = en_answer

    # Step 7 — English → Odia translation
    odia_answer = english_to_odia(en_answer, translator_components)
    debug_steps["7_final_odia_answer"] = odia_answer

    elapsed = round(time.time() - t_start, 2)

    avg_retrieval_score = (
        round(sum(s for _, s in retrieved) / len(retrieved), 4)
        if retrieved else 0.0
    )

    return {
        "answer":              odia_answer,
        "answer_en":           en_answer,
        "debug":               debug_steps,
        "retrieval_score":     avg_retrieval_score,
        "response_time":       elapsed,
        "retrieved_chunks":    retrieved,
    }


def run_traditional_rag(
    odia_question: str,
    od_index,
    od_chunks: list,
    embedder,
    groq_api_key: str,
    top_k: int = 3,
) -> dict:
    """
    Traditional RAG pipeline — no translation, Odia query on Odia docs.
    Returns a dict with the final answer + debug steps.
    """
    debug_steps = {}
    t_start = time.time()

    debug_steps["1_original_odia"] = odia_question

    # Retrieve directly with Odia query against Odia docs
    retrieved = retrieve(odia_question, od_index, od_chunks, embedder, top_k=top_k)
    debug_steps["2_retrieved_chunks"] = retrieved

    # Generate answer (ask LLM to respond in Odia)
    answer = generate_answer(odia_question, retrieved, groq_api_key, mode="od")
    debug_steps["3_final_answer"] = answer

    elapsed = round(time.time() - t_start, 2)

    avg_retrieval_score = (
        round(sum(s for _, s in retrieved) / len(retrieved), 4)
        if retrieved else 0.0
    )

    return {
        "answer":           answer,
        "debug":            debug_steps,
        "retrieval_score":  avg_retrieval_score,
        "response_time":    elapsed,
        "retrieved_chunks": retrieved,
    }