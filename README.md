<div align="center">
  <h1>🌉 LinguaBridge</h1>
  <p><strong>A Cross-Lingual RAG System for Low-Resource Languages</strong></p>
  
  <p>
    <a href="https://rag-for-lrl-using-translation-model.vercel.app/" target="_blank">View Live Demo</a>
    ·
    <a href="#architecture">View Architecture</a>
    ·
    <a href="#features">Features</a>
  </p>
</div>

---

LinguaBridge is a specialized, open-source Retrieval-Augmented Generation (RAG) system built to serve **Odia (ଓଡ଼ିଆ)**, a low-resource Indic language. Since high-quality LLMs and embeddings struggle with low-resource languages natively, this project introduces a robust translation-layer pipeline that queries an English knowledge base using Odia, unlocking global information without sacrificing accuracy.

## 🚀 Live Demo

**[👉 Try LinguaBridge Here](https://rag-for-lrl-using-translation-model.vercel.app/)**

> **Test it out:** We've provided an English PDF and an Odia PDF directly on the application's sidebar. Download them, upload them back into the system, and try the example query provided in the UI!

## 📸 UI Screenshots

### Query Interface

![LinguaBridge Query Interface](C:\Users\LENOVO\OneDrive\Desktop\LinguaBridge\image copy.png)

### LLM Evaluation Result

![LinguaBridge LLM Evaluation](C:\Users\LENOVO\OneDrive\Desktop\LinguaBridge\image.png)

---

## 🏛️ System Architecture

LinguaBridge operates across a multi-node cloud architecture to bypass free-tier memory and Compute restrictions.

### 1. The RAG Engine (`Backend — Render`)
A **FastAPI** orchestrator running PyTorch (CPU-optimized) that handles document ingestion and the heavy lifting of modern retrieval:
*   **Vector Search (FAISS):** Dense retrieval using `BAAI/bge-small-en-v1.5`.
*   **Keyword Search (BM25Okapi):** Sparse retrieval for exact-match terminology.
*   **Reciprocal Rank Fusion (RRF):** Intelligently merges Vector and Keyword scores.
*   **CrossEncoder Reranking:** Re-scores the fused hybrid results using `ms-marco-MiniLM-L-6-v2` for maximum contextual relevance.
*   **Generation (Groq LLM):** Passes the top chunks to `Llama-3-70b-versatile` for extremely fast, hardware-accelerated answer synthesis.

### 2. The Translation API (`HuggingFace Spaces`)
We fine-tuned the state-of-the-art **IndicTrans2** model using custom LoRA (Low-Rank Adaptation) adapters for English ↔ Odia translation. 
*   Hosted entirely on a HuggingFace Space (T4/CPU tier).
*   Allows the backend to asynchronously translate Odia queries to English, and English LLM responses back to Odia.

### 3. The User Interface (`Vercel Edge`)
A modern **Next.js** React application offering a beautiful, responsive chat interface.
*   Implements **Server-Side API Proxying** (`/api/*`) to seamlessly route requests from the browser to Render, entirely bypassing strict CORS restrictions.
*   Features an advanced side-by-side comparison mode allowing users to benchmark the "LinguaBridge" approach against "Traditional RAG".

---

## 🛠️ Tech Stack 

| Layer | Technologies Used |
| :--- | :--- |
| **Frontend** | Next.js, React, CSS Modules, Vercel Edge Serverless |
| **Backend** | Python, FastAPI, PyTorch (CPU wheel), Uvicorn, Render |
| **AI / NLP** | HuggingFace `gradio_client`, FAISS, Groq (Llama-3), SentenceTransformers |
| **Custom Models** | IndicTrans2 (IT2), LoRA Adapters (PEFT) |

---

## 💡 The Pipeline: How it Works

When a user submits an Odia query, the LinguaBridge pipeline executes the following sequence:

1.  **Odia Query Submission:** The user asks a question in Odia.
2.  **Entity Guard (Optional):** Protects specific named entities from being translated incorrectly.
3.  **Forward Translation (Odia → English):** The query is securely sent to our HuggingFace Space and translated into high-quality English.
4.  **Hybrid Retrieval:** The English query searches the English FAISS Vector database AND the BM25 sparse index simultaneously.
5.  **Reranking:** A CrossEncoder re-evaluates the top documents from both sources to pick the absolute best context chunks.
6.  **LLM Generation:** The Groq API generates a precise English answer based *only* on the retrieved chunks.
7.  **Backward Translation (English → Odia):** The English answer is translated back into Odia via HuggingFace Spaces.
8.  **Delivery:** The user receives a native Odia response in the UI, complete with performance metrics and an exact pipeline trace!

---

## 🔧 Local Development

To run this heavily segmented architecture locally:

### 1. Environment Setup
Clone the repository and configure your `.env` variables inside the `/backend` folder. You will need a Groq API key and HuggingFace Token.
```bash
# Example
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...
HF_SPACES_URL=https://peeyush237-linguabridge-translate.hf.space
```

### 2. Start the Backend (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Start the Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:3000` to interact with the application locally.

---

