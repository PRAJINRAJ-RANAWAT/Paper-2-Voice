"""
End-to-end integration tests for Paper to Voice Assistant with mocked external services.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from pydub.generators import Sine

from src.paper_to_voice.workflow.orchestrator import create_podcast_workflow
from src.paper_to_voice.audio.processor import store_voice, consolidate_voice, verify_audio_metadata


def test_full_langgraph_workflow_mocked(monkeypatch):
    """
    Test the entire LangGraph workflow end-to-end using mocked Gemini responses.
    Verifies that state progresses through:
    START -> generate_steps -> markdown_to_json -> parse_json -> solve_substeps -> generate_dialog -> END
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test-mock-api-key")

    mock_steps_content = "Research paper analysis content."
    mock_json_content = """```json
    [
        {
            "step": "Identify architectural novelties",
            "substeps": [
                {"key": "Attention Mechanism", "value": "How does linear attention scale with sequence length?"}
            ]
        },
        {
            "step": "Analyze empirical benchmarks",
            "substeps": [
                {"key": "Dataset Evaluation", "value": "What metrics are reported on ImageNet and COCO?"}
            ]
        }
    ]
    ```"""
    mock_solution_content = "Question: How does linear attention scale? \n Answer: It scales linearly O(N)."
    mock_dialog_content = """## Podcast Script
**Jane:** Welcome to the podcast! Today we discuss linear attention mechanisms.
**Dr. Sharma:** Linear attention allows processing long sequences with O(N) complexity instead of quadratic."""

    call_count = 0

    def mock_invoke(messages, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        msg_str = str(messages)

        if "expert in understanding research papers" in msg_str:
            resp.content = mock_steps_content
        elif "schema strictly" in msg_str:
            resp.content = mock_json_content
        elif "analyze research papers" in msg_str:
            resp.content = mock_solution_content
        elif "podcast producer" in msg_str:
            resp.content = mock_dialog_content
        else:
            resp.content = "Generic mocked LLM response"
        return resp

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_invoke

    with patch("src.paper_to_voice.workflow.steps.get_llm", return_value=mock_llm), \
         patch("src.paper_to_voice.workflow.dialog.get_llm", return_value=mock_llm), \
         patch("src.paper_to_voice.workflow.orchestrator.get_llm", return_value=mock_llm):

        app, _ = create_podcast_workflow()
        initial_state = {"image_path": ["test_page_1.jpg", "test_page_2.jpg"]}

        events = list(app.stream(initial_state))

    assert len(events) > 0
    # Collect all generate_dialog outputs
    dialog_outputs = []
    for event in events:
        if "generate_dialog" in event:
            dialog_outputs.append(event["generate_dialog"])

    # 2 steps fanned out => 2 dialogue tasks produced
    assert len(dialog_outputs) == 2
    for d in dialog_outputs:
        assert "Dialog" in d
        assert len(d["Dialog"]) == 1
        assert "**Jane:**" in d["Dialog"][0]
        assert "**Dr. Sharma:**" in d["Dialog"][0]


def test_audio_pipeline_integration(monkeypatch):
    """
    Test audio synthesis and consolidation pipeline with mocked MeloTTS.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock audio synthesizer that generates a tiny valid WAV file
        mock_audio_counter = 0

        def mock_generate_audio(text, language):
            nonlocal mock_audio_counter
            mock_audio_counter += 1
            path = os.path.join(tmp_dir, f"mock_speech_{mock_audio_counter}.wav")
            Sine(400).to_audio_segment(duration=150).export(path, format="wav")
            return path

        topic_dialog = {
            "step_0": """**Jane:** Hello listeners, welcome to Paper to Voice!
**Dr. Sharma:** Great to be here to break down this research.""",
            "step_1": """**Jane:** What were the key conclusions of the study?
**Dr. Sharma:** The model achieved state-of-the-art accuracy with half the parameters.""",
        }

        with patch("src.paper_to_voice.audio.processor.generate_podcast_audio", side_effect=mock_generate_audio):
            audio_paths = store_voice(topic_dialog, max_concurrency=2)
            assert len(audio_paths) == 4
            assert all(os.path.exists(p) for p in audio_paths)

            final_mp3 = consolidate_voice(audio_paths, voice_dir=tmp_dir)
            assert final_mp3 is not None
            assert os.path.exists(final_mp3)

            metadata = verify_audio_metadata(final_mp3)
            assert metadata["sample_rate"] == 44100
            assert metadata["channels"] == 2
            assert metadata["duration_seconds"] > 0
