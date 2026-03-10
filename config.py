# ─────────────────────────────────────────────
#  LinguaBridge — Central Configuration
# ─────────────────────────────────────────────

# ── Model paths ──────────────────────────────
EN_TO_OR_ADAPTER = "/content/drive/MyDrive/it2_en2or_lora_adapter"  # English → Odia
OR_TO_EN_ADAPTER = "/content/drive/MyDrive/it2_or2en_lora_adapter"  # Odia → English
EN_INDIC_BASE     = "ai4bharat/indictrans2-en-indic-1B"
INDIC_EN_BASE     = "ai4bharat/indictrans2-indic-en-1B"

# ── Language codes ────────────────────────────
SRC_LANG_EN  = "eng_Latn"
TGT_LANG_OD  = "ory_Orya"   # Odia script — NOT ory_Deva

# ── RAG settings ──────────────────────────────
CHUNK_SIZE        = 400      # words per chunk
CHUNK_OVERLAP     = 50       # overlap between chunks
TOP_K             = 3        # number of chunks to retrieve
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"

# ── Groq ──────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"   # free tier, fast

# ── Paths ─────────────────────────────────────
ENTITY_DICT_PATH  = "data/entity_dict.json"
EN_FAISS_PATH     = "data/en_faiss.index"
OD_FAISS_PATH     = "data/od_faiss.index"
EN_CHUNKS_PATH    = "data/en_chunks.pkl"
OD_CHUNKS_PATH    = "data/od_chunks.pkl"
