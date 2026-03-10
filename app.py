"""
app.py
──────
LinguaBridge — RAG for Low-Resource Languages
Main Streamlit application.

Run with:
    streamlit run app.py
"""

import time
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="LinguaBridge",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from modules.rag import load_embedder, ingest_documents, load_index, llm_evaluate
from modules.translator import load_translator
from modules.pipeline import run_linguabridge, run_traditional_rag
from config import (
    EN_FAISS_PATH, EN_CHUNKS_PATH,
    OD_FAISS_PATH, OD_CHUNKS_PATH,
    TOP_K,
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .subtitle {
        color: #555;
        font-size: 1rem;
        margin-top: 0;
    }
    .answer-box {
        background: #f8f9ff;
        border-left: 4px solid #4a6fa5;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 1.05rem;
        line-height: 1.7;
        min-height: 80px;
    }
    .answer-box-trad {
        background: #fff8f0;
        border-left: 4px solid #e07b39;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 1.05rem;
        line-height: 1.7;
        min-height: 80px;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
    }
    .verdict-box {
        background: #eafaf1;
        border-left: 4px solid #27ae60;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.95rem;
    }
    .step-label {
        font-weight: 600;
        color: #4a6fa5;
        margin-bottom: 2px;
    }
    .chunk-box {
        background: #f4f4f4;
        border-radius: 6px;
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
        margin-bottom: 6px;
        border-left: 3px solid #aaa;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🌉 LinguaBridge</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Retrieval-Augmented QA System for Low-Resource Languages • '
    'Odia ↔ English</p>', unsafe_allow_html=True
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Groq API Key
    groq_api_key = st.text_input(
        "Groq API Key", type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com"
    )

    st.divider()

    # RAG settings
    st.subheader("RAG Settings")
    top_k = st.slider("Top-K chunks to retrieve", 1, 8, TOP_K)
    show_en_answer = st.checkbox("Show English answer (LinguaBridge)", value=False)

    st.divider()

    # Document upload — English (for LinguaBridge)
    st.subheader("📄 English Knowledge Base")
    st.caption("Used by LinguaBridge pipeline")
    en_files = st.file_uploader(
        "Upload English PDFs/TXTs",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="en_uploader"
    )

    if en_files:
        if st.button("🔨 Build English Index", use_container_width=True):
            embedder = load_embedder()
            with st.spinner("Ingesting English documents..."):
                idx, chunks, n = ingest_documents(en_files, lang="en", embedder=embedder)
                st.session_state["en_index"]  = idx
                st.session_state["en_chunks"] = chunks
            st.success(f"✅ {n} chunks indexed from {len(en_files)} file(s)")

    st.divider()

    # Document upload — Odia (for Traditional RAG)
    st.subheader("📄 Odia Knowledge Base")
    st.caption("Used by Traditional RAG pipeline")
    od_files = st.file_uploader(
        "Upload Odia PDFs/TXTs",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="od_uploader"
    )

    if od_files:
        if st.button("🔨 Build Odia Index", use_container_width=True):
            embedder = load_embedder()
            with st.spinner("Ingesting Odia documents..."):
                idx, chunks, n = ingest_documents(od_files, lang="od", embedder=embedder)
                st.session_state["od_index"]  = idx
                st.session_state["od_chunks"] = chunks
            st.success(f"✅ {n} chunks indexed from {len(od_files)} file(s)")

    st.divider()

    # Load model button
    if st.button("🤖 Load Translation Model", use_container_width=True):
        with st.spinner("Loading IndicTrans2 models (~2 min)..."):
            components = load_translator()
            st.session_state["translator"] = components
        st.success("✅ Translation model ready!")

    # Load saved indexes from disk
    if st.button("📂 Load Saved Indexes", use_container_width=True):
        embedder = load_embedder()
        en_idx, en_ch = load_index(EN_FAISS_PATH, EN_CHUNKS_PATH)
        od_idx, od_ch = load_index(OD_FAISS_PATH, OD_CHUNKS_PATH)
        if en_idx:
            st.session_state["en_index"]  = en_idx
            st.session_state["en_chunks"] = en_ch
            st.success(f"✅ English index: {len(en_ch)} chunks")
        if od_idx:
            st.session_state["od_index"]  = od_idx
            st.session_state["od_chunks"] = od_ch
            st.success(f"✅ Odia index: {len(od_ch)} chunks")
        if not en_idx and not od_idx:
            st.warning("No saved indexes found. Please upload and build first.")

    # Status indicators
    st.divider()
    st.subheader("Status")
    st.write("🤖 Translation model:",
             "✅ Ready" if "translator" in st.session_state else "❌ Not loaded")
    st.write("📘 English index:",
             f"✅ {len(st.session_state.get('en_chunks', []))} chunks"
             if "en_index" in st.session_state else "❌ Not built")
    st.write("📗 Odia index:",
             f"✅ {len(st.session_state.get('od_chunks', []))} chunks"
             if "od_index" in st.session_state else "❌ Not built")


# ── Main query area ───────────────────────────────────────────────────────────
st.markdown('<p class="section-header">💬 Ask a Question in Odia</p>',
            unsafe_allow_html=True)

odia_question = st.text_area(
    label="odia_input",
    label_visibility="collapsed",
    placeholder="ଏଠାରେ ଆପଣଙ୍କ ପ୍ରଶ୍ନ ଲେଖନ୍ତୁ... (Type your question in Odia here)",
    height=100,
)

run_btn = st.button("🔍 Search & Answer", type="primary", use_container_width=False)

# ── Guard checks ──────────────────────────────────────────────────────────────
if run_btn:
    if not odia_question.strip():
        st.warning("Please enter a question.")
        st.stop()
    if "translator" not in st.session_state:
        st.error("Please load the translation model first (sidebar).")
        st.stop()
    if "en_index" not in st.session_state:
        st.error("Please build the English knowledge base index first (sidebar).")
        st.stop()
    if "od_index" not in st.session_state:
        st.error("Please build the Odia knowledge base index first (sidebar).")
        st.stop()
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    embedder    = load_embedder()
    translator  = st.session_state["translator"]
    en_index    = st.session_state["en_index"]
    en_chunks   = st.session_state["en_chunks"]
    od_index    = st.session_state["od_index"]
    od_chunks   = st.session_state["od_chunks"]

    # ── Run both pipelines ────────────────────────────────────────────────────
    with st.spinner("Running LinguaBridge pipeline..."):
        lb_result = run_linguabridge(
            odia_question, en_index, en_chunks, embedder,
            translator, groq_api_key, top_k=top_k
        )

    with st.spinner("Running Traditional RAG pipeline..."):
        trad_result = run_traditional_rag(
            odia_question, od_index, od_chunks, embedder,
            groq_api_key, top_k=top_k
        )

    # ── Side-by-side answers ──────────────────────────────────────────────────
    st.divider()
    col_lb, col_trad = st.columns(2, gap="large")

    with col_lb:
        st.markdown("### 🌉 LinguaBridge Answer")
        st.success(lb_result["answer"])
        if show_en_answer:
            st.caption("English answer (intermediate):")
            st.info(lb_result["answer_en"])

    with col_trad:
        st.markdown("### 📚 Traditional RAG Answer")
        st.warning(trad_result["answer"])

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">📊 Performance Metrics</p>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "⏱ LinguaBridge Time",
            f"{lb_result['response_time']}s"
        )
    with m2:
        st.metric(
            "⏱ Traditional RAG Time",
            f"{trad_result['response_time']}s"
        )
    with m3:
        st.metric(
            "🎯 LB Retrieval Score",
            f"{lb_result['retrieval_score']:.3f}",
            delta=f"{lb_result['retrieval_score'] - trad_result['retrieval_score']:+.3f}",
            help="Average cosine similarity of top-k chunks to query. Higher = better retrieval."
        )
    with m4:
        st.metric(
            "🎯 Trad. Retrieval Score",
            f"{trad_result['retrieval_score']:.3f}",
        )

    # ── LLM Self-Evaluation Verdict ───────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">🏆 LLM Evaluation Verdict</p>',
                unsafe_allow_html=True)

    with st.spinner("Asking Groq to evaluate both answers..."):
        verdict = llm_evaluate(
            odia_question,
            lb_result["answer"],
            trad_result["answer"],
            groq_api_key,
        )

    st.success(verdict)

    # ── Debug Panel ───────────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔍 Debug Panel — LinguaBridge Pipeline", expanded=False):
        d = lb_result["debug"]

        st.markdown("**Step 1 — Original Odia Input**")
        st.code(d.get("1_original_odia", ""), language=None)

        st.markdown("**Step 2 — After Entity Protection**")
        st.code(d.get("2_after_entity_protect", ""), language=None)
        if d.get("2_entity_mapping"):
            st.caption("Entity mapping:")
            for ph, en in d["2_entity_mapping"].items():
                st.code(f"{ph}  →  {en}", language=None)
        else:
            st.caption("No entities detected in this query.")

        st.markdown("**Step 3 — Translated to English**")
        st.code(d.get("3_translated_to_english", ""), language=None)

        st.markdown("**Step 4 — After Entity Restore**")
        st.code(d.get("4_after_entity_restore", ""), language=None)

        st.markdown("**Step 5 — Retrieved Chunks (English)**")
        chunks_lb = lb_result.get("retrieved_chunks") or []
        for i, item in enumerate(chunks_lb):
            chunk, score = item[0], item[1]
            st.markdown(f"**Chunk {i+1}** (score: {score:.4f})")
            st.info(chunk)

        st.markdown("**Step 6 — English Answer from Groq**")
        st.info(d.get("6_english_answer", ""))

        st.markdown("**Step 7 — Final Odia Answer**")
        st.success(d.get("7_final_odia_answer", ""))

    with st.expander("🔍 Debug Panel — Traditional RAG Pipeline", expanded=False):
        d2 = trad_result["debug"]

        st.markdown("**Step 1 — Original Odia Input (no translation)**")
        st.code(d2.get("1_original_odia", ""), language=None)

        chunks_trad = trad_result.get("retrieved_chunks") or []
        for i, item in enumerate(chunks_trad):
            chunk, score = item[0], item[1]
            st.markdown(f"**Chunk {i+1}** (score: {score:.4f})")
            st.info(chunk)

        st.markdown("**Step 3 — Final Answer**")
        st.info(d2.get("3_final_answer", ""))


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "LinguaBridge • IIIT Nagpur • Semester VI CSA • "
    "Peeyush Mishra | Divyal Surse | Sandesh Charhate • "
    "Guide: Mr. Amol Bhopale"
)