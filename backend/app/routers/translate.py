"""
translate.py
────────────
POST /api/translate — Proxy translation requests to HuggingFace Spaces.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import TranslateRequest, TranslateResponse, Language
from app.services.translation import translate_odia_to_english, translate_english_to_odia
from app.config import settings

router = APIRouter(prefix="/api", tags=["translate"])


@router.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """
    Translate text between Odia and English.
    Proxies the request to HuggingFace Spaces.
    """
    hf_url = settings.HF_SPACES_URL
    if not hf_url:
        raise HTTPException(
            status_code=400,
            detail="HF_SPACES_URL not configured. Translation service unavailable."
        )

    try:
        if request.source_lang == Language.OD and request.target_lang == Language.EN:
            translated = await translate_odia_to_english(request.text, hf_url)
        elif request.source_lang == Language.EN and request.target_lang == Language.OD:
            translated = await translate_english_to_odia(request.text, hf_url)
        else:
            raise HTTPException(
                status_code=400,
                detail="Only Odia↔English translation is supported."
            )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return TranslateResponse(
        translated_text=translated,
        source_lang=request.source_lang.value,
        target_lang=request.target_lang.value,
    )
