"""
Tests for LangGraph Send-based fan-out, out-of-order deterministic aggregation,
and bounded concurrency.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

from src.paper_to_voice.workflow.orchestrator import (
    continue_to_substeps,
    continue_to_substeps_voice,
)
from src.paper_to_voice.audio.processor import store_voice


def test_one_task_per_research_step():
    """Verify continue_to_substeps creates exactly one Send task per research step."""
    plan = [
        {"step": "Step 1", "substeps": ["q1", "q2"]},
        {"step": "Step 2", "substeps": ["q3"]},
        {"step": "Step 3", "substeps": ["q4", "q5"]},
    ]
    state = {"plan": plan, "image_path": ["img1.jpg", "img2.jpg"]}
    tasks = continue_to_substeps(state)

    assert len(tasks) == 3
    for idx, send_task in enumerate(tasks):
        assert send_task.node == "solve_substeps"
        assert send_task.arg["step"] == plan[idx]
        assert send_task.arg["image_path"] == ["img1.jpg", "img2.jpg"]
        assert send_task.arg["step_index"] == idx


def test_empty_or_invalid_plan_raises_error():
    """Verify continue_to_substeps rejects empty or missing plans."""
    with pytest.raises(ValueError, match="empty, missing, or invalid"):
        continue_to_substeps({"plan": []})

    with pytest.raises(ValueError, match="empty, missing, or invalid"):
        continue_to_substeps({})


def test_one_dialogue_task_per_valid_step_solution_pair():
    """Verify continue_to_substeps_voice creates a dialogue task per valid pair."""
    state = {
        "indexed_solutions": [
            {"index": 0, "step": "Step 1", "solution": "Sol 1"},
            {"index": 1, "step": "Step 2", "solution": "Sol 2"},
        ],
        "tone": "Formal",
        "length": "Short (1-2 min)",
        "language": "EN",
    }
    tasks = continue_to_substeps_voice(state)

    assert len(tasks) == 2
    assert tasks[0].node == "generate_dialog"
    assert tasks[0].arg["step"] == "Step 1"
    assert tasks[0].arg["text"] == "Sol 1"
    assert tasks[0].arg["step_index"] == 0
    assert tasks[1].arg["step"] == "Step 2"
    assert tasks[1].arg["text"] == "Sol 2"
    assert tasks[1].arg["step_index"] == 1


def test_out_of_order_task_completion_deterministic_ordering():
    """Verify that tasks completing out-of-order are sorted back deterministically by index."""
    # Tasks completed in jumbled order: step 2 first, then step 0, then step 1
    state = {
        "indexed_solutions": [
            {"index": 2, "step": "Step 3", "solution": "Sol 3"},
            {"index": 0, "step": "Step 1", "solution": "Sol 1"},
            {"index": 1, "step": "Step 2", "solution": "Sol 2"},
        ],
    }
    tasks = continue_to_substeps_voice(state)

    # Verification: Tasks MUST be dispatched in ordered step sequence [0, 1, 2]
    assert len(tasks) == 3
    assert tasks[0].arg["step_index"] == 0
    assert tasks[0].arg["step"] == "Step 1"
    assert tasks[1].arg["step_index"] == 1
    assert tasks[1].arg["step"] == "Step 2"
    assert tasks[2].arg["step_index"] == 2
    assert tasks[2].arg["step"] == "Step 3"


def test_mismatched_step_and_solution_counts_handled_gracefully():
    """Verify that mismatched step and solution counts do not crash or silently lose data."""
    state = {
        "steps": ["Step 1", "Step 2", "Step 3"],
        "solutions": ["Sol 1", "Sol 2"],  # One solution missing
    }
    tasks = continue_to_substeps_voice(state)

    assert len(tasks) == 2
    assert tasks[0].arg["step"] == "Step 1"
    assert tasks[1].arg["step"] == "Step 2"


def test_individual_task_failure_does_not_corrupt_remaining_results():
    """Verify that if an individual task fails, valid remaining tasks still proceed."""
    state = {
        "indexed_solutions": [
            {"index": 0, "step": "Step 1", "solution": "Sol 1"},
            {"index": 1, "step": "Step 2", "solution": None},  # Failed task
            {"index": 2, "step": "Step 3", "solution": "Sol 3"},
        ],
    }
    tasks = continue_to_substeps_voice(state)

    assert len(tasks) == 2
    assert tasks[0].arg["step_index"] == 0
    assert tasks[1].arg["step_index"] == 2


def test_concurrency_limit_enforcement():
    """Verify that parallel processing enforces the configured concurrency limit."""
    max_concurrency = 2
    active_threads = 0
    max_active_observed = 0

    def mock_generate_podcast_audio(text, language):
        nonlocal active_threads, max_active_observed
        active_threads += 1
        if active_threads > max_active_observed:
            max_active_observed = active_threads
        time.sleep(0.05)
        active_threads -= 1
        return f"/mock/path/{text[:5]}.mp3"

    topic_dialog = {
        "topic1": [
            "**Jane:** Line 1",
            "**Dr. Sharma:** Line 2",
            "**Jane:** Line 3",
            "**Dr. Sharma:** Line 4",
        ]
    }

    with patch("src.paper_to_voice.audio.processor.generate_podcast_audio", side_effect=mock_generate_podcast_audio):
        paths = store_voice(topic_dialog, max_concurrency=max_concurrency)

    assert len(paths) == 4
    # Max concurrency observed must NOT exceed configured max_workers
    assert max_active_observed <= max_concurrency
    assert all(p.startswith("/mock/path/") for p in paths)
