"""
generator.py
────────────
Groq LLM answer generation — extracted from the original rag.py
with no Streamlit dependency.

Improvements:
  • Includes chunk relevance scores in the prompt
  • Better prompt engineering for factual answers
  • Supports both English and Odia answer modes
"""

from groq import Groq


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(
    question: str,
    retrieved_chunks: list[tuple[str, float]],
    mode: str = "en",
) -> str:
    """
    Build the LLM prompt with retrieved context and relevance scores.

    Args:
        question:         The user's question
        retrieved_chunks: List of (chunk_text, relevance_score) tuples
        mode:             "en" for English answers, "od" for Odia answers

    Returns:
        Formatted prompt string
    """
    context = "\n\n".join(
        f"[Chunk {i+1} | Relevance: {score:.3f}]: {text}"
        for i, (text, score) in enumerate(retrieved_chunks)
    )

    if mode == "en":
        return f"""You are a precise, helpful assistant. Answer the question using ONLY the context provided below.

Rules:
- If the answer is not in the context, say "I could not find relevant information in the provided documents."
- Be concise, factual, and cite which chunk(s) your answer comes from.
- Higher relevance scores indicate more trustworthy chunks.

Context:
{context}

Question: {question}

Answer:"""
    else:
        return f"""You are a helpful assistant. The question is in Odia language.
Answer the question in Odia using ONLY the context provided below.
If the answer is not in the context, say you could not find the information.
Be concise and factual.

Context:
{context}

Question: {question}

Answer:"""


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    retrieved_chunks: list[tuple[str, float]],
    groq_api_key: str,
    mode: str = "en",
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Generate an answer using Groq LLM with retrieved context.

    Args:
        question:         The user's question
        retrieved_chunks: List of (chunk_text, score) tuples
        groq_api_key:     Groq API key
        mode:             "en" or "od"
        model:            Groq model name
        temperature:      Sampling temperature
        max_tokens:       Maximum response tokens

    Returns:
        Generated answer string
    """
    if not groq_api_key:
        return "⚠️ No Groq API key provided."
    if not retrieved_chunks:
        return "⚠️ No relevant documents found in the knowledge base."

    client = Groq(api_key=groq_api_key)
    prompt = build_prompt(question, retrieved_chunks, mode=mode)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ── LLM Self-Evaluation ──────────────────────────────────────────────────────

def llm_evaluate(
    original_question_odia: str,
    lb_answer_odia: str,
    trad_answer: str,
    groq_api_key: str,
    model: str = DEFAULT_GROQ_MODEL,
) -> str:
    """
    Ask Groq to compare LinguaBridge answer vs Traditional RAG answer.
    Returns a verdict string.
    """
    if not groq_api_key:
        return "⚠️ No Groq API key provided."

    prompt = f"""You are an impartial evaluator for two question-answering systems.

Original Question (in Odia): {original_question_odia}

System A (LinguaBridge - translates query to English, retrieves from English docs, translates answer back to Odia):
Answer: {lb_answer_odia}

System B (Traditional RAG - uses Odia query directly on Odia docs):
Answer: {trad_answer}

Evaluate both answers on:
1. Relevance (does it answer the question?)
2. Completeness (is the answer complete?)
3. Fluency (is it well-formed?)

Give each system a score out of 10 and declare a winner. Be concise.
Format your response as:
System A Score: X/10
System B Score: X/10
Winner: [System A / System B / Tie]
Reason: [1-2 sentences]"""

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()
