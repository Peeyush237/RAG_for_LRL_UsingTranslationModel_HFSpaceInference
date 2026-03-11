"""
translation.py
──────────────
Client for calling the HuggingFace Spaces translation API.

The IndicTrans2 model runs on HuggingFace Spaces (free T4 GPU).
This service uses the official `gradio_client` block to handle SSE
and API negotiation seamlessly, avoiding direct httpx requests.
"""

import logging
import asyncio
from gradio_client import Client

logger = logging.getLogger(__name__)

# Cache the client to avoid re-initializing it for every request,
# which requires fetching the API schema each time.
_client_instance = None


def get_client(hf_spaces_url: str) -> Client:
    global _client_instance
    if _client_instance is None:
        logger.info(f"Initializing Gradio Client for {hf_spaces_url}")
        _client_instance = Client(hf_spaces_url)
    return _client_instance


def _do_predict_sync(text: str, api_name: str, hf_spaces_url: str) -> str:
    """Synchronous translation call to be run in a thread."""
    client = get_client(hf_spaces_url)
    result = client.predict(text, api_name=api_name)
    return str(result)


async def _call_hf_spaces(
    text: str,
    api_name: str,
    hf_spaces_url: str,
) -> str:
    """
    Call a Gradio API endpoint on HuggingFace Spaces using asyncio.to_thread
    to prevent the synchronous client from blocking the FastAPI event loop.

    Args:
        text:           Input text to translate
        api_name:       Gradio API name (e.g., "/odia_to_english")
        hf_spaces_url:  Base URL of the HuggingFace Space

    Returns:
        Translated text string
    """
    try:
        # Run the synchronous gradio_client in a worker thread
        result = await asyncio.to_thread(
            _do_predict_sync, 
            text, 
            api_name, 
            hf_spaces_url
        )
        return result
    except Exception as e:
        logger.error(f"HF Spaces API error using gradio_client: {e}")
        raise RuntimeError(f"Translation API error: {str(e)}")


async def translate_odia_to_english(text: str, hf_spaces_url: str) -> str:
    """Translate Odia text to English via HuggingFace Spaces."""
    return await _call_hf_spaces(text, "/odia_to_english", hf_spaces_url)


async def translate_english_to_odia(text: str, hf_spaces_url: str) -> str:
    """Translate English text to Odia via HuggingFace Spaces."""
    return await _call_hf_spaces(text, "/english_to_odia", hf_spaces_url)
