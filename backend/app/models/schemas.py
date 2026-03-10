"""
schemas.py
──────────
Pydantic request/response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class PipelineMode(str, Enum):
    LINGUABRIDGE = "linguabridge"
    TRADITIONAL = "traditional"
    BOTH = "both"


class Language(str, Enum):
    EN = "en"
    OD = "od"


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask (in Odia for LinguaBridge)")
    pipeline_mode: PipelineMode = Field(
        default=PipelineMode.BOTH,
        description="Which pipeline to run"
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Number of chunks to retrieve")
    groq_api_key: str = Field(default="", description="Groq API key (optional if set in env)")


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source: str = ""
    index: int = 0


class DebugStep(BaseModel):
    step_name: str
    value: str


class PipelineResult(BaseModel):
    answer: str
    answer_en: str = ""
    retrieved_chunks: list[RetrievedChunk] = []
    debug_steps: list[DebugStep] = []
    retrieval_score: float = 0.0
    response_time: float = 0.0


class QueryResponse(BaseModel):
    linguabridge: PipelineResult | None = None
    traditional: PipelineResult | None = None
    evaluation: str = ""


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    success: bool
    lang: str
    num_files: int
    num_chunks: int
    message: str


# ── Translate ─────────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_lang: Language = Field(..., description="Source language")
    target_lang: Language = Field(..., description="Target language")


class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    embedder_loaded: bool = False
    reranker_loaded: bool = False
    en_index_loaded: bool = False
    od_index_loaded: bool = False
    en_chunks_count: int = 0
    od_chunks_count: int = 0
