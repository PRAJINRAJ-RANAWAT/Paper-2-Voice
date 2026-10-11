"""
Audio processing, bounded concurrent TTS synthesis, and audio consolidation functionality.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from pydub import AudioSegment
from pydub.generators import Sine

from .tts import generate_podcast_audio
from ..core.config import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    LIGHT_GUITAR_FREQ,
    AMBIENT_GUITAR_FREQ,
    GUITAR_DURATION,
    MAX_CONCURRENCY,
)

logger = logging.getLogger("paper_to_voice.audio.processor")


def synthesize_segment(
    idx: int,
    text: str,
    language: str = "EN",
) -> Dict[str, Any]:
    """
    Synthesize an individual dialogue segment with structured result reporting.
    """
    clean = text.strip()
    if not clean:
        return {"index": idx, "status": "empty", "file_path": None, "text": clean}

    try:
        audio_path = generate_podcast_audio(clean, language=language)
        if audio_path == "Empty Text":
            return {"index": idx, "status": "skipped", "file_path": None, "text": clean}
        return {"index": idx, "status": "success", "file_path": audio_path, "text": clean}
    except Exception as exc:
        logger.error("Failed to synthesize segment %d ('%s'): %s", idx, clean[:40], exc)
        return {"index": idx, "status": "failed", "file_path": None, "text": clean, "error": str(exc)}


def store_voice(
    topic_dialog: dict,
    language: str = "EN",
    max_concurrency: int = MAX_CONCURRENCY,
) -> List[str]:
    """
    Generate voice files for all dialogue parts across all topics using bounded concurrency.
    Preserves exact sequential order of dialogue parts.
    
    Args:
        topic_dialog: Dictionary containing topic dialogues (keyed by index or topic name).
        language: Language code.
        max_concurrency: Maximum number of worker threads for in-process parallel I/O.
        
    Returns:
        List of generated audio file paths (or 'Empty Text' / None if skipped/failed).
    """
    all_lines: List[str] = []
    for topic, dialog in topic_dialog.items():
        if isinstance(dialog, str):
            lines = dialog.split("\n")
        elif isinstance(dialog, list):
            lines = dialog
        else:
            continue

        for line in lines:
            line_str = str(line).strip()
            if line_str:
                all_lines.append(line_str)

    if not all_lines:
        logger.warning("No dialogue lines found in topic_dialog.")
        return []

    logger.info(
        "Synthesizing %d dialogue segments with bounded concurrency (limit=%d)",
        len(all_lines),
        max_concurrency,
    )

    results: List[Optional[Dict[str, Any]]] = [None] * len(all_lines)

    # Use bounded ThreadPoolExecutor for I/O-bound TTS requests
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        future_to_idx = {
            executor.submit(synthesize_segment, idx, line, language): idx
            for idx, line in enumerate(all_lines)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results[idx] = res
            except Exception as exc:
                logger.error("Segment synthesis exception at index %d: %s", idx, exc)
                results[idx] = {
                    "index": idx,
                    "status": "failed",
                    "file_path": None,
                    "text": all_lines[idx],
                    "error": str(exc),
                }

    audio_paths: List[str] = []
    successful_count = 0
    failed_count = 0

    for item in results:
        if item is not None and item.get("status") == "success" and item.get("file_path"):
            audio_paths.append(item["file_path"])
            successful_count += 1
        else:
            if item and item.get("status") == "failed":
                failed_count += 1
            audio_paths.append("Empty Text")

    logger.info(
        "Voice synthesis completed: %d succeeded, %d failed out of %d total segments",
        successful_count,
        failed_count,
        len(all_lines),
    )
    return audio_paths


def verify_audio_metadata(file_path: str) -> Dict[str, Any]:
    """
    Inspect actual exported audio file properties to verify sample rate and channel count.
    """
    segment = AudioSegment.from_file(file_path)
    return {
        "file_path": file_path,
        "sample_rate": segment.frame_rate,
        "channels": segment.channels,
        "duration_seconds": len(segment) / 1000.0,
    }


def consolidate_voice(
    audio_paths: List[str],
    voice_dir: str,
    target_sample_rate: int = AUDIO_SAMPLE_RATE,
    target_channels: int = AUDIO_CHANNELS,
) -> Optional[str]:
    """
    Consolidate multiple audio files into a single master podcast file.
    Validates audio files, normalizes to 44,100 Hz stereo, and verifies the final output.
    
    Args:
        audio_paths: List of synthesized audio file paths.
        voice_dir: Directory to store background audio and temporary outputs.
        target_sample_rate: Target frame rate in Hz (default: 44,100).
        target_channels: Target channel count (default: 2 for stereo).
        
    Returns:
        Path to the verified consolidated MP3 file, or None if no valid segments exist.
    """
    os.makedirs(voice_dir, exist_ok=True)
    valid_paths = [p for p in audio_paths if p and p != "Empty Text"]

    # Verify that candidate audio files actually exist on disk
    existing_paths = []
    missing_paths = []
    for p in valid_paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            existing_paths.append(p)
        else:
            missing_paths.append(p)

    if missing_paths:
        logger.warning(
            "Found %d missing or zero-byte audio segments before consolidation: %s",
            len(missing_paths),
            missing_paths,
        )

    if not existing_paths:
        logger.error("No valid audio segments available for consolidation.")
        return None

    audio_segments: List[AudioSegment] = []

    # Generate or load intro/outro guitar audio
    light_guitar_path = os.path.join(voice_dir, "light-guitar.wav")
    ambient_guitar_path = os.path.join(voice_dir, "ambient-guitar.wav")

    if not os.path.exists(light_guitar_path):
        sine = Sine(LIGHT_GUITAR_FREQ).to_audio_segment(duration=GUITAR_DURATION)
        sine = sine.set_frame_rate(target_sample_rate).set_channels(target_channels)
        sine.export(light_guitar_path, format="wav")

    if not os.path.exists(ambient_guitar_path):
        sine = Sine(AMBIENT_GUITAR_FREQ).to_audio_segment(duration=GUITAR_DURATION)
        sine = sine.set_frame_rate(target_sample_rate).set_channels(target_channels)
        sine.export(ambient_guitar_path, format="wav")

    # Add intro
    intro_segment = AudioSegment.from_file(light_guitar_path).set_frame_rate(target_sample_rate).set_channels(target_channels)
    audio_segments.append(intro_segment)

    # Add voice segments with normalized sample rate and stereo channels
    loaded_voice_segments = 0
    for path in existing_paths:
        try:
            seg = AudioSegment.from_file(path)
            seg = seg.set_frame_rate(target_sample_rate).set_channels(target_channels)
            audio_segments.append(seg)
            loaded_voice_segments += 1
        except Exception as exc:
            logger.warning("Failed to decode audio segment %s: %s", path, exc)

    if loaded_voice_segments == 0:
        logger.error("None of the voice segments could be loaded into AudioSegments.")
        return None

    # Add outro
    outro_segment = AudioSegment.from_file(ambient_guitar_path).set_frame_rate(target_sample_rate).set_channels(target_channels)
    audio_segments.append(outro_segment)

    # Combine all segments
    combined_audio = sum(audio_segments)
    combined_audio = combined_audio.set_frame_rate(target_sample_rate).set_channels(target_channels)

    tmp_dir = os.path.join(voice_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    with NamedTemporaryFile(dir=tmp_dir, delete=False, suffix=".mp3") as temp_file:
        output_path = temp_file.name

    combined_audio.export(
        output_path,
        format="mp3",
        parameters=["-ar", str(target_sample_rate), "-ac", str(target_channels)],
    )

    # Verify actual exported file metadata
    metadata = verify_audio_metadata(output_path)
    logger.info(
        "Consolidated podcast exported and verified: sample_rate=%d, channels=%d, duration=%.2fs, file=%s",
        metadata["sample_rate"],
        metadata["channels"],
        metadata["duration_seconds"],
        output_path,
    )

    if metadata["sample_rate"] != target_sample_rate:
        logger.warning(
            "Exported sample rate (%d) does not match target (%d)",
            metadata["sample_rate"],
            target_sample_rate,
        )

    if metadata["channels"] != target_channels:
        logger.warning(
            "Exported channels (%d) does not match target (%d)",
            metadata["channels"],
            target_channels,
        )

    return output_path


def cleanup_audio_files(paths: List[str]) -> None:
    """Safely remove a list of temporary audio files if they exist."""
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as exc:
            logger.debug("Failed to remove temporary file %s: %s", p, exc)
