"""
LinguaBridge Translation API — HuggingFace Spaces
──────────────────────────────────────────────────
Hosts the IndicTrans2 model with LoRA adapters on a free T4 GPU.
Exposes Odia↔English translation as a Gradio app + API.

Deployment:
  1. Create a new HuggingFace Space (Gradio SDK, T4 GPU)
  2. Upload your LoRA adapters to HuggingFace
  3. Push this file + requirements.txt to the Space
  4. The Space will auto-deploy
"""

import torch
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

# ── Configuration ─────────────────────────────────────────────────────────────

# IMPORTANT: Update these paths after uploading your adapters to HuggingFace
# Option A: Upload to a HuggingFace model repo
#   EN_TO_OR_ADAPTER = "your-username/it2-en2or-lora-adapter"
#   OR_TO_EN_ADAPTER = "your-username/it2-or2en-lora-adapter"
# Option B: Upload directly to the Space's repository
#   EN_TO_OR_ADAPTER = "./it2_en2or_lora_adapter"
#   OR_TO_EN_ADAPTER = "./it2_or2en_lora_adapter"

EN_TO_OR_ADAPTER = "your-username/it2-en2or-lora-adapter"  # UPDATE THIS
OR_TO_EN_ADAPTER = "your-username/it2-or2en-lora-adapter"  # UPDATE THIS

EN_INDIC_BASE = "ai4bharat/indictrans2-en-indic-1B"
INDIC_EN_BASE = "ai4bharat/indictrans2-indic-en-1B"

SRC_LANG_EN = "eng_Latn"
TGT_LANG_OD = "ory_Orya"

# ── Patch for transformers version compatibility ──────────────────────────────

def _patch_indictrans_toolkit():
    """Fix import issues in IndicTransToolkit with newer transformers."""
    try:
        import importlib.util, pathlib
        pkg_path = pathlib.Path(
            importlib.util.find_spec('IndicTransToolkit').origin
        ).parent
        collator_path = pkg_path / 'collator.py'
        content = collator_path.read_text()
        if 'from transformers.tokenization_utils import PreTrainedTokenizerBase' in content:
            content = content.replace(
                'from transformers.tokenization_utils import PreTrainedTokenizerBase',
                'from transformers import PreTrainedTokenizerBase'
            )
            collator_path.write_text(content)
    except Exception:
        pass

_patch_indictrans_toolkit()

try:
    from IndicTransToolkit import IndicProcessor
except ImportError:
    from IndicTransToolkit.IndicTransToolkit import IndicProcessor


# ── Model Loading ─────────────────────────────────────────────────────────────

print("Loading models... This may take 2-3 minutes on first run.")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# English → Odia
print("Loading English → Odia model...")
en_indic_base = AutoModelForSeq2SeqLM.from_pretrained(
    EN_INDIC_BASE, trust_remote_code=True, torch_dtype=dtype,
)
en_indic_model = PeftModel.from_pretrained(en_indic_base, EN_TO_OR_ADAPTER)
en_indic_model = en_indic_model.merge_and_unload().to(device).eval()

# Odia → English
print("Loading Odia → English model...")
indic_en_base = AutoModelForSeq2SeqLM.from_pretrained(
    INDIC_EN_BASE, trust_remote_code=True, torch_dtype=dtype,
)
indic_en_model = PeftModel.from_pretrained(indic_en_base, OR_TO_EN_ADAPTER)
indic_en_model = indic_en_model.merge_and_unload().to(device).eval()

# Tokenizers
en_indic_tok = AutoTokenizer.from_pretrained(EN_INDIC_BASE, trust_remote_code=True)
indic_en_tok = AutoTokenizer.from_pretrained(INDIC_EN_BASE, trust_remote_code=True)

# IndicProcessor
ip = IndicProcessor(inference=True)

print("✅ All models loaded!")


# ── Translation Functions ─────────────────────────────────────────────────────

def _translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Core translation function."""
    if src_lang == SRC_LANG_EN:  # English → Odia
        model = en_indic_model
        tokenizer = en_indic_tok
    else:  # Odia → English
        model = indic_en_model
        tokenizer = indic_en_tok

    forced_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    preprocessed = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)

    inputs = tokenizer(
        preprocessed,
        truncation=True,
        padding="longest",
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_length=256,
            forced_bos_token_id=forced_id,
        )

    decoded = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    result = ip.postprocess_batch(decoded, lang=tgt_lang)
    return result[0] if result else ""


def odia_to_english(text: str) -> str:
    """Translate Odia → English."""
    if not text.strip():
        return ""
    return _translate(text, src_lang=TGT_LANG_OD, tgt_lang=SRC_LANG_EN)


def english_to_odia(text: str) -> str:
    """Translate English → Odia."""
    if not text.strip():
        return ""
    return _translate(text, src_lang=SRC_LANG_EN, tgt_lang=TGT_LANG_OD)


# ── Gradio Interface ──────────────────────────────────────────────────────────

with gr.Blocks(title="LinguaBridge Translation API") as demo:
    gr.Markdown("# 🌉 LinguaBridge Translation API")
    gr.Markdown("IndicTrans2 fine-tuned with LoRA adapters for Odia ↔ English translation.")

    with gr.Tab("Odia → English"):
        or2en_input = gr.Textbox(
            label="Odia Text",
            placeholder="ଏଠାରେ ଓଡ଼ିଆ ଲେଖନ୍ତୁ...",
            lines=4,
        )
        or2en_output = gr.Textbox(label="English Translation", lines=4)
        or2en_btn = gr.Button("Translate", variant="primary")
        or2en_btn.click(odia_to_english, inputs=or2en_input, outputs=or2en_output)

    with gr.Tab("English → Odia"):
        en2or_input = gr.Textbox(
            label="English Text",
            placeholder="Type English text here...",
            lines=4,
        )
        en2or_output = gr.Textbox(label="Odia Translation", lines=4)
        en2or_btn = gr.Button("Translate", variant="primary")
        en2or_btn.click(english_to_odia, inputs=en2or_input, outputs=en2or_output)

    gr.Markdown("""
    ---
    **API Usage**: This Space also provides an API. Use the following endpoints:
    - `POST /call/odia_to_english` with `{"data": ["your odia text"]}`
    - `POST /call/english_to_odia` with `{"data": ["your english text"]}`

    Built for the LinguaBridge project — IIIT Nagpur
    """)

demo.launch()
