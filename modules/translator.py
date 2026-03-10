"""
translator.py
─────────────
Loads two separate fine-tuned IndicTrans2 LoRA adapters:
  • it2_or2en_lora_adapter  ->  Odia -> English  (query translation)
  • it2_en2or_lora_adapter  ->  English -> Odia  (answer translation)
"""

import torch
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

try:
    from IndicTransToolkit import IndicProcessor
except ImportError:
    from IndicTransToolkit.IndicTransToolkit import IndicProcessor

from config import (
    EN_TO_OR_ADAPTER, OR_TO_EN_ADAPTER,
    EN_INDIC_BASE, INDIC_EN_BASE,
    SRC_LANG_EN, TGT_LANG_OD
)


# ── Patch for transformers version compatibility ──────────────────────────────
def _patch_indictrans_toolkit():
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


@st.cache_resource(show_spinner=False)
def load_translator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress = st.progress(0, text="Loading English → Odia model (1/2)...")

    # English -> Odia
    en_indic_base = AutoModelForSeq2SeqLM.from_pretrained(
        EN_INDIC_BASE, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    progress.progress(25, text="Applying EN→OR LoRA adapter...")
    en_indic_model = PeftModel.from_pretrained(en_indic_base, EN_TO_OR_ADAPTER)
    en_indic_model = en_indic_model.merge_and_unload().to(device).eval()
    progress.progress(50, text="Loading Odia → English model (2/2)...")

    # Odia -> English
    indic_en_base = AutoModelForSeq2SeqLM.from_pretrained(
        INDIC_EN_BASE, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    progress.progress(75, text="Applying OR→EN LoRA adapter...")
    indic_en_model = PeftModel.from_pretrained(indic_en_base, OR_TO_EN_ADAPTER)
    indic_en_model = indic_en_model.merge_and_unload().to(device).eval()
    progress.progress(90, text="Loading tokenizers...")

    en_indic_tok = AutoTokenizer.from_pretrained(EN_INDIC_BASE, trust_remote_code=True)
    indic_en_tok = AutoTokenizer.from_pretrained(INDIC_EN_BASE, trust_remote_code=True)
    ip_inference = IndicProcessor(inference=True)

    progress.progress(100, text="✅ Both models ready!")
    progress.empty()

    return {
        "en_indic_model": en_indic_model,
        "indic_en_model": indic_en_model,
        "en_indic_tok":   en_indic_tok,
        "indic_en_tok":   indic_en_tok,
        "ip":             ip_inference,
        "device":         device,
    }


def _translate(text: str, src_lang: str, tgt_lang: str, components: dict) -> str:
    """Core translation function used internally."""
    ip     = components["ip"]
    device = components["device"]

    # Pick correct model + tokenizer based on direction
    if src_lang == SRC_LANG_EN:         # English -> Odia
        model     = components["en_indic_model"]
        tokenizer = components["en_indic_tok"]
    else:                               # Odia -> English
        model     = components["indic_en_model"]
        tokenizer = components["indic_en_tok"]

    forced_id    = tokenizer.convert_tokens_to_ids(tgt_lang)
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
            forced_bos_token_id=forced_id,  # forces correct output script
        )

    decoded = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    result = ip.postprocess_batch(decoded, lang=tgt_lang)
    return result[0] if result else ""


def odia_to_english(text: str, components: dict) -> str:
    """Translate Odia -> English using or2en adapter."""
    return _translate(text, src_lang=TGT_LANG_OD, tgt_lang=SRC_LANG_EN, components=components)


def english_to_odia(text: str, components: dict) -> str:
    """Translate English -> Odia using en2or adapter."""
    return _translate(text, src_lang=SRC_LANG_EN, tgt_lang=TGT_LANG_OD, components=components)