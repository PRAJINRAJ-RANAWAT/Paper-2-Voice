"""
Text-to-Speech functionality using MeloTTS with exponential backoff and jitter retry handling.
"""

import logging
from typing import Optional, Tuple
from gradio_client import Client

from ..core.config import (
    BASE_RETRY_DELAY,
    MAX_RETRIES,
    MAX_RETRY_DELAY,
    TTS_MODEL,
)
from ..utils.retry import retry_with_backoff

logger = logging.getLogger("paper_to_voice.audio.tts")


def parse_speaker_and_text(text: str) -> Optional[Tuple[str, str, float, str]]:
    """
    Parse speaker formatting and return (speaker_label, clean_text, speed, accent).
    Returns None if text does not match a supported speaker tag.
    """
    stripped = text.strip()
    if "**Jane:**" in stripped:
        clean = stripped.replace("**Jane:**", "").strip()
        return ("Jane", clean, 0.9, "EN-US")
    elif "**Dr. Sharma:**" in stripped:
        clean = stripped.split("**")[-1].strip()
        return ("Dr. Sharma", clean, 0.9, "EN_INDIA")
    return None


def get_text_to_voice(
    text: str,
    speed: float = 0.9,
    accent: str = "EN-US",
    language: str = "EN",
) -> str:
    """
    Convert text to speech using MeloTTS via Gradio Client.
    
    Args:
        text: Text to convert to speech.
        speed: Speaking speed (default: 0.9).
        accent: Voice accent / speaker profile (default: "EN-US").
        language: Language code (default: "EN").
        
    Returns:
        Path to the generated audio file.
    """
    hf_client = Client(TTS_MODEL)
    file_path = hf_client.predict(
        text=text,
        language=language,
        speaker=accent,
        speed=speed,
        api_name="/synthesize",
    )
    return file_path


def generate_podcast_audio(text: str, language: str = "EN") -> str:
    """
    Generate podcast audio with appropriate voice settings for different speakers.
    Uses exponential backoff with randomized jitter to handle transient failures.
    
    Args:
        text: Formatted dialogue text containing speaker prefix.
        language: Language code (default: "EN").
        
    Returns:
        Path to generated audio file, or 'Empty Text' if text contains no recognized speaker.
        
    Raises:
        RetryExhaustedError: When transient errors persist across all retries.
        Exception: On non-retryable / permanent errors.
    """
    parsed = parse_speaker_and_text(text)
    if not parsed:
        return 'Empty Text'

    speaker, clean_text, speed, accent = parsed
    if not clean_text:
        return 'Empty Text'

    def _call():
        return get_text_to_voice(clean_text, speed=speed, accent=accent, language=language)

    logger.info("Synthesizing audio for speaker '%s' (accent=%s, len=%d)", speaker, accent, len(clean_text))
    file_path = retry_with_backoff(
        _call,
        operation_name=f"tts_synthesis_{speaker}",
        max_retries=MAX_RETRIES,
        base_delay=BASE_RETRY_DELAY,
        max_delay=MAX_RETRY_DELAY,
    )
    return file_path
