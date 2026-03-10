"""
translation.py
──────────────
Client for calling the HuggingFace Spaces translation API.

The IndicTrans2 model runs on HuggingFace Spaces (free T4 GPU).
This service makes HTTP calls to the Gradio API endpoint.
"""

import httpx
import logging

logger = logging.getLogger(__name__)

# Timeout for HF Spaces calls (cold start can take 30-60s)
HF_TIMEOUT = 120.0


async def _call_hf_spaces(
    text: str,
    api_name: str,
    hf_spaces_url: str,
) -> str:
    """
    Call a Gradio API endpoint on HuggingFace Spaces.

    Args:
        text:           Input text to translate
        api_name:       Gradio API name (e.g., "/odia_to_english")
        hf_spaces_url:  Base URL of the HuggingFace Space

    Returns:
        Translated text string
    """
    url = f"{hf_spaces_url.rstrip('/')}/api/predict"

    payload = {
        "fn_index": 0 if api_name == "/odia_to_english" else 1,
        "data": [text],
    }

    # Try the Gradio REST endpoint
    # Different Gradio versions use different endpoint formats
    gradio_url = f"{hf_spaces_url.rstrip('/')}/call{api_name}"

    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        try:
            # Step 1: Submit the request
            response = await client.post(
                gradio_url,
                json={"data": [text]},
            )
            response.raise_for_status()
            event_id = response.json().get("event_id")

            if event_id:
                # Step 2: Get the result using SSE
                result_url = f"{hf_spaces_url.rstrip('/')}/call{api_name}/{event_id}"
                result_response = await client.get(result_url)
                result_response.raise_for_status()

                # Parse SSE response
                for line in result_response.text.split("\n"):
                    if line.startswith("data:"):
                        import json
                        data = json.loads(line[5:].strip())
                        if isinstance(data, list) and len(data) > 0:
                            return data[0]
                        return str(data)

            # Fallback: direct response
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]

            return str(result)

        except httpx.HTTPStatusError as e:
            logger.error(f"HF Spaces API error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Translation API error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("HF Spaces API timeout — the Space may be cold-starting")
            raise RuntimeError(
                "Translation API timeout. The HuggingFace Space may be starting up. "
                "Please try again in 30-60 seconds."
            )
        except Exception as e:
            logger.error(f"HF Spaces API error: {e}")
            raise RuntimeError(f"Translation API error: {str(e)}")


async def translate_odia_to_english(text: str, hf_spaces_url: str) -> str:
    """Translate Odia text to English via HuggingFace Spaces."""
    return await _call_hf_spaces(text, "/odia_to_english", hf_spaces_url)


async def translate_english_to_odia(text: str, hf_spaces_url: str) -> str:
    """Translate English text to Odia via HuggingFace Spaces."""
    return await _call_hf_spaces(text, "/english_to_odia", hf_spaces_url)
