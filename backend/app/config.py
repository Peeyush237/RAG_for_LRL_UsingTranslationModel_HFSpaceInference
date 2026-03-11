"""
config.py
─────────
Backend configuration using Pydantic BaseSettings.
Loads from environment variables or .env file.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── API Keys ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    HF_SPACES_URL: str = ""  # URL of the HuggingFace Spaces translation API
    HF_TOKEN: str = ""       # HuggingFace token for authenticated model downloads

    # ── Models ────────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"  # ~130MB, fits in 512MB free tier
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ENABLE_RERANKER: bool = False  # Disable by default for free tier (saves ~80MB)
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ── RAG Settings ──────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 200           # words per chunk
    CHUNK_OVERLAP: int = 40         # overlap between chunks
    TOP_K: int = 5                  # final chunks to return
    FAISS_TOP_N: int = 20           # FAISS initial retrieval count
    BM25_TOP_N: int = 20            # BM25 initial retrieval count
    RRF_TOP_N: int = 15             # candidates after RRF fusion
    RRF_K: int = 60                 # RRF constant

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    EN_FAISS_PATH: str = ""
    OD_FAISS_PATH: str = ""
    EN_CHUNKS_PATH: str = ""
    OD_CHUNKS_PATH: str = ""
    EN_BM25_PATH: str = ""
    OD_BM25_PATH: str = ""
    ENTITY_DICT_PATH: str = ""

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://*.vercel.app"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set default paths relative to DATA_DIR if not explicitly set
        if not self.EN_FAISS_PATH:
            self.EN_FAISS_PATH = os.path.join(self.DATA_DIR, "en_faiss.index")
        if not self.OD_FAISS_PATH:
            self.OD_FAISS_PATH = os.path.join(self.DATA_DIR, "od_faiss.index")
        if not self.EN_CHUNKS_PATH:
            self.EN_CHUNKS_PATH = os.path.join(self.DATA_DIR, "en_chunks.pkl")
        if not self.OD_CHUNKS_PATH:
            self.OD_CHUNKS_PATH = os.path.join(self.DATA_DIR, "od_chunks.pkl")
        if not self.EN_BM25_PATH:
            self.EN_BM25_PATH = os.path.join(self.DATA_DIR, "en_bm25.pkl")
        if not self.OD_BM25_PATH:
            self.OD_BM25_PATH = os.path.join(self.DATA_DIR, "od_bm25.pkl")
        if not self.ENTITY_DICT_PATH:
            self.ENTITY_DICT_PATH = os.path.join(self.DATA_DIR, "entity_dict.json")


# Singleton
settings = Settings()
