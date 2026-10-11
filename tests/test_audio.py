"""
Tests for audio processing, speaker parsing, consolidation, 44.1kHz stereo verification,
and cleanup.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from pydub import AudioSegment
from pydub.generators import Sine

from src.paper_to_voice.audio.tts import (
    generate_podcast_audio,
    parse_speaker_and_text,
)
from src.paper_to_voice.audio.processor import (
    cleanup_audio_files,
    consolidate_voice,
    verify_audio_metadata,
)


def test_speaker_tag_parsing():
    """Verify parsing of Jane and Dr. Sharma speaker tags."""
    res_jane = parse_speaker_and_text("**Jane:** Welcome to the podcast discussion!")
    assert res_jane is not None
    assert res_jane[0] == "Jane"
    assert res_jane[1] == "Welcome to the podcast discussion!"
    assert res_jane[2] == 0.9
    assert res_jane[3] == "EN-US"

    res_sharma = parse_speaker_and_text("**Dr. Sharma:** Thank you, excited to discuss our paper.")
    assert res_sharma is not None
    assert res_sharma[0] == "Dr. Sharma"
    assert res_sharma[1] == "Thank you, excited to discuss our paper."
    assert res_sharma[3] == "EN_INDIA"

    res_none = parse_speaker_and_text("Just some text without speaker tag.")
    assert res_none is None


def test_tts_returns_empty_text_for_unrecognized_speaker():
    """Verify generate_podcast_audio returns 'Empty Text' for lines without speakers."""
    assert generate_podcast_audio("No speaker header here") == "Empty Text"
    assert generate_podcast_audio("") == "Empty Text"


def test_tts_retry_on_transient_failure():
    """Verify that transient failures during TTS invoke retry and succeed."""
    attempts = 0

    def mock_predict(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("504 Gateway Timeout from HuggingFace space")
        return "/mock/generated_voice.wav"

    mock_client = MagicMock()
    mock_client.predict.side_effect = mock_predict

    with patch("src.paper_to_voice.audio.tts.Client", return_value=mock_client):
        res = generate_podcast_audio("**Jane:** Testing retry mechanism.")

    assert res == "/mock/generated_voice.wav"
    assert attempts == 2


def test_consolidate_voice_and_verify_44100_stereo():
    """Verify that consolidated audio is exported to 44,100 Hz, 2 channels (stereo)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create 2 valid sine wave audio files simulating speech segments
        seg1_path = os.path.join(tmp_dir, "segment1.wav")
        seg2_path = os.path.join(tmp_dir, "segment2.wav")

        Sine(300).to_audio_segment(duration=200).export(seg1_path, format="wav")
        Sine(500).to_audio_segment(duration=200).export(seg2_path, format="wav")

        audio_paths = [seg1_path, "Empty Text", seg2_path]
        final_mp3 = consolidate_voice(audio_paths, voice_dir=tmp_dir)

        assert final_mp3 is not None
        assert os.path.exists(final_mp3)

        # Inspect actual exported file properties
        metadata = verify_audio_metadata(final_mp3)
        assert metadata["sample_rate"] == 44100
        assert metadata["channels"] == 2
        assert metadata["duration_seconds"] > 0


def test_consolidate_voice_no_valid_segments_returns_none():
    """Verify consolidate_voice returns None if all paths are empty or missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = consolidate_voice(["Empty Text", "/nonexistent/file.wav"], voice_dir=tmp_dir)
        assert result is None


def test_cleanup_audio_files_safely_removes_files():
    """Verify cleanup_audio_files deletes existing files and handles missing files without error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        f1 = os.path.join(tmp_dir, "temp1.wav")
        with open(f1, "w") as f:
            f.write("data")
        assert os.path.exists(f1)

        cleanup_audio_files([f1, os.path.join(tmp_dir, "nonexistent.wav"), None])
        assert not os.path.exists(f1)
