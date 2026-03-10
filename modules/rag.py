"""
rag.py
──────
Handles:
  1. Document ingestion  — PDF + TXT parsing
  2. Chunking            — overlapping word-level chunks
  3. Embedding           — sentence-transformers (BAAI/bge-small-en-v1.5)
  4. FAISS index         — build + save + load
  5. Retrieval           — top-k semantic search with scores
  6. Answer generation   — Groq LLM with retrieved context
"""

import os
import re
import pickle
import time
import numpy as np
import faiss
import streamlit as st
from groq import Groq

from sentence_transformers import SentenceTransformer
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    EMBEDDING_MODEL,
    EN_FAISS_PATH, OD_FAISS_PATH,
    EN_CHUNKS_PATH, OD_CHUNKS_PATH,
    GROQ_MODEL,
)


# ── PDF / TXT parsing ─────────────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file) -> str:
    """Extract raw text from a Streamlit UploadedFile (PDF or TXT)."""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        try:
            import pdfplumber
            import io
            text_parts = []
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            return "\n".join(text_parts)
        except ImportError:
            st.error("pdfplumber not installed. Run: pip install pdfplumber")
            return ""
    else:
        st.warning(f"Unsupported file type: {uploaded_file.name}")
        return ""


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    Preserves sentence boundaries where possible.
    """
    # Split into sentences first
    sentences = re.split(r'(?<=[।\.!\?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_words = []

    for sentence in sentences:
        words = sentence.split()
        current_words.extend(words)

        if len(current_words) >= chunk_size:
            chunks.append(" ".join(current_words[:chunk_size]))
            # Keep last `overlap` words for next chunk
            current_words = current_words[chunk_size - overlap:]

    # Remainder
    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if len(c.strip()) > 20]


# ── Embedding model ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str], embedder) -> np.ndarray:
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# ── FAISS index management ────────────────────────────────────────────────────

def build_index(chunks: list[str], embedder) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index (cosine sim on normalized vectors)."""
    vectors = embed(chunks, embedder)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index


def save_index(index, chunks: list[str], faiss_path: str, chunks_path: str):
    faiss.write_index(index, faiss_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)


def load_index(faiss_path: str, chunks_path: str):
    if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
        return None, []
    index = faiss.read_index(faiss_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def ingest_documents(uploaded_files, lang: str, embedder) -> tuple:
    """
    Parse, chunk, embed and index a list of uploaded files.

    lang: "en" for English docs, "od" for Odia docs
    Returns: (index, chunks, num_chunks)
    """
    all_chunks = []
    for uf in uploaded_files:
        text = parse_uploaded_file(uf)
        if text:
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

    if not all_chunks:
        return None, [], 0

    index = build_index(all_chunks, embedder)

    faiss_path  = EN_FAISS_PATH  if lang == "en" else OD_FAISS_PATH
    chunks_path = EN_CHUNKS_PATH if lang == "en" else OD_CHUNKS_PATH
    save_index(index, all_chunks, faiss_path, chunks_path)

    return index, all_chunks, len(all_chunks)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, index, chunks: list[str], embedder, top_k: int = TOP_K):
    """
    Retrieve top-k most relevant chunks for a query.

    Returns list of (chunk_text, score) tuples.
    """
    if index is None or not chunks:
        return []

    q_vec = embed([query], embedder).astype(np.float32)
    scores, indices = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append((chunks[idx], float(score)))

    return results


# ── Groq LLM answer generation ────────────────────────────────────────────────

def build_prompt(question: str, retrieved_chunks: list, mode: str = "en") -> str:
    """
    Build the LLM prompt.
    mode: "en" = question is in English (LinguaBridge)
          "od" = question is in Odia (Traditional RAG)
    """
    context = "\n\n".join(
        [f"[Chunk {i+1}]: {chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)]
    )

    if mode == "en":
        return f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find relevant information."
Be concise and factual.

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


def generate_answer(question: str, retrieved_chunks: list, groq_api_key: str, mode: str = "en") -> str:
    """Call Groq API and return the generated answer."""
    if not groq_api_key:
        return "⚠️ No Groq API key provided."
    if not retrieved_chunks:
        return "⚠️ No relevant documents found in the knowledge base."

    client = Groq(api_key=groq_api_key)
    prompt = build_prompt(question, retrieved_chunks, mode=mode)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


# ── LLM Self-Evaluation ───────────────────────────────────────────────────────

def llm_evaluate(
    original_question_odia: str,
    lb_answer_odia: str,
    trad_answer: str,
    groq_api_key: str,
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
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()