"""Audio processing and TTS module for Paper to Voice Assistant"""

from .processor import (
    cleanup_audio_files,
    consolidate_voice,
    store_voice,
    synthesize_segment,
    verify_audio_metadata,
)
from .tts import generate_podcast_audio, get_text_to_voice, parse_speaker_and_text

__all__ = [
    "cleanup_audio_files",
    "consolidate_voice",
    "store_voice",
    "synthesize_segment",
    "verify_audio_metadata",
    "generate_podcast_audio",
    "get_text_to_voice",
    "parse_speaker_and_text",
]
