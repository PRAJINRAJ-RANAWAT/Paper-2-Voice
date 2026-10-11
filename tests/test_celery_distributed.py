"""
Tests for Celery distributed task processing, worker execution, idempotency,
retries, task status tracking, and result aggregation.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.paper_to_voice.tasks.celery_app import celery_app
from src.paper_to_voice.tasks.tracking import TaskStatus, TaskTracker, tracker
from src.paper_to_voice.tasks.workers import (
    generate_dialog_task,
    solve_substeps_task,
)
from src.paper_to_voice.workflow.steps import solve_substeps
from src.paper_to_voice.workflow.dialog import generate_dialog


@pytest.fixture(autouse=True)
def configure_celery_eager():
    """Ensure Celery runs in eager mode during testing to execute tasks locally in-process."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_eager_propagates = False


def test_celery_configuration():
    """Verify Celery application routing, serialization, and safety configurations."""
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.task_acks_late is True
    assert "solve_substeps_task" in celery_app.conf.task_routes
    assert celery_app.conf.task_routes["solve_substeps_task"]["queue"] == "research_steps"
    assert celery_app.conf.task_routes["generate_dialog_task"]["queue"] == "dialog_tasks"


def test_solve_substeps_task_execution():
    """Verify worker execution of solve_substeps_task and status tracking."""
    mock_resp = MagicMock()
    mock_resp.content = "Answer: Attention scales with sequence length quadratically in standard self-attention."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp

    step_data = {
        "step": "Analyze attention scalability",
        "substeps": ["How does attention scale with length?"],
    }

    with patch("src.paper_to_voice.tasks.workers.get_llm", return_value=mock_llm):
        async_res = solve_substeps_task.apply_async(
            kwargs={
                "step_data": step_data,
                "image_paths": ["page1.jpg"],
                "step_index": 0,
                "workflow_id": "test_wf_1",
            },
            task_id="workflow_test_wf_1_step_0",
        )
        res = async_res.get()

    assert res["index"] == 0
    assert res["step"] == "Analyze attention scalability"
    assert "Attention scales" in res["solution"]

    # Verify task status was recorded as SUCCEEDED
    status_record = tracker.get_task_status("workflow_test_wf_1_step_0")
    assert status_record is not None
    assert status_record["status"] == TaskStatus.SUCCEEDED


def test_generate_dialog_task_execution():
    """Verify worker execution of generate_dialog_task and status tracking."""
    mock_resp = MagicMock()
    mock_resp.content = "**Jane:** Welcome!\n**Dr. Sharma:** Glad to be here."
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp

    with patch("src.paper_to_voice.tasks.workers.get_llm", return_value=mock_llm):
        async_res = generate_dialog_task.apply_async(
            kwargs={
                "step_desc": "Step 1",
                "finding_text": "Attention scales linearly.",
                "tone": "Formal",
                "length": "Short (1-2 min)",
                "language": "EN",
                "step_index": 1,
                "workflow_id": "test_wf_2",
            },
            task_id="workflow_test_wf_2_dialog_1",
        )
        res = async_res.get()

    assert res["index"] == 1
    assert res["step"] == "Step 1"
    assert "**Jane:**" in res["dialog"]

    status_record = tracker.get_task_status("workflow_test_wf_2_dialog_1")
    assert status_record is not None
    assert status_record["status"] == TaskStatus.SUCCEEDED


def test_task_idempotency_duplicate_delivery():
    """Verify that redelivering a task with an existing stable ID returns the cached result without duplicate LLM calls."""
    mock_resp = MagicMock()
    mock_resp.content = "Original Computed Solution"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp

    stable_task_id = "workflow_idemp_step_0"
    step_data = {"step": "Idempotent Step", "substeps": ["q1"]}

    with patch("src.paper_to_voice.tasks.workers.get_llm", return_value=mock_llm):
        # First execution
        res1 = solve_substeps_task.apply_async(
            kwargs={"step_data": step_data, "image_paths": [], "step_index": 0, "workflow_id": "idemp"},
            task_id=stable_task_id,
        ).get()
        assert mock_llm.invoke.call_count == 1

        # Second delivery (e.g. duplicate network delivery / worker retry)
        res2 = solve_substeps_task.apply_async(
            kwargs={"step_data": step_data, "image_paths": [], "step_index": 0, "workflow_id": "idemp"},
            task_id=stable_task_id,
        ).get()

        # LLM should NOT have been called a second time
        assert mock_llm.invoke.call_count == 1
        assert res2 == res1


def test_task_tracker_lifecycle():
    """Verify TaskTracker handles PENDING, RUNNING, SUCCEEDED, and FAILED states cleanly."""
    custom_tracker = TaskTracker(redis_url="redis://nonexistent:9999/0")  # In-memory fallback
    task_id = "test_lifecycle_task"

    # Pending
    custom_tracker.set_task_status(task_id, TaskStatus.PENDING, workflow_id="wf_1", stage="research", step_index=0)
    st = custom_tracker.get_task_status(task_id)
    assert st["status"] == TaskStatus.PENDING

    # Running
    custom_tracker.set_task_status(task_id, TaskStatus.RUNNING, workflow_id="wf_1", stage="research", step_index=0)
    st = custom_tracker.get_task_status(task_id)
    assert st["status"] == TaskStatus.RUNNING

    # Succeeded
    custom_tracker.store_idempotent_result(task_id, {"result": "success"})
    custom_tracker.set_task_status(task_id, TaskStatus.SUCCEEDED, workflow_id="wf_1", stage="research", step_index=0)
    is_done, cached = custom_tracker.is_idempotent_complete(task_id)
    assert is_done is True
    assert cached["result"] == "success"

    # Failed
    failed_id = "test_failed_task"
    custom_tracker.set_task_status(failed_id, TaskStatus.FAILED, error="Connection timeout")
    st_failed = custom_tracker.get_task_status(failed_id)
    assert st_failed["status"] == TaskStatus.FAILED
    assert st_failed["error"] == "Connection timeout"


def test_worker_failure_explicitly_recorded():
    """Verify worker failures update status to FAILED and raise explicit error."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("Fatal worker error")

    task_id = "workflow_fail_step_0"
    with patch("src.paper_to_voice.tasks.workers.get_llm", return_value=mock_llm):
        with pytest.raises(RuntimeError, match="Fatal worker error"):
            solve_substeps_task.apply_async(
                kwargs={
                    "step_data": {"step": "Failing step", "substeps": []},
                    "image_paths": [],
                    "step_index": 0,
                    "workflow_id": "fail",
                },
                task_id=task_id,
            ).get()

    status_record = tracker.get_task_status(task_id)
    assert status_record is not None
    assert status_record["status"] == TaskStatus.FAILED
    assert "Fatal worker error" in status_record["error"]


def test_solve_substeps_node_with_celery_enabled(monkeypatch):
    """Verify LangGraph solve_substeps node seamlessly dispatches to Celery when USE_CELERY=True."""
    monkeypatch.setenv("USE_CELERY", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-test-key")

    mock_resp = MagicMock()
    mock_resp.content = "Celery processed solution"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp

    state = {
        "step": {"step": "Celery Step", "substeps": ["q1"]},
        "image_path": ["img.jpg"],
        "step_index": 0,
        "workflow_id": "wf_celery",
    }

    with patch("src.paper_to_voice.tasks.workers.get_llm", return_value=mock_llm):
        out = solve_substeps(state)

    assert out["steps"] == ["Celery Step"]
    assert out["solutions"] == ["Celery processed solution"]
    assert len(out["indexed_solutions"]) == 1
    assert out["indexed_solutions"][0]["index"] == 0


def test_generate_dialog_node_with_celery_enabled(monkeypatch):
    """Verify LangGraph generate_dialog node seamlessly dispatches to Celery when USE_CELERY=True."""
    monkeypatch.setenv("USE_CELERY", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-test-key")

    mock_resp = MagicMock()
    mock_resp.content = "**Jane:** Dialogue\n**Dr. Sharma:** Indeed"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp

    state = {
        "step": "Step 1",
        "text": "Finding text",
        "tone": "Formal",
        "length": "Short (1-2 min)",
        "language": "EN",
        "step_index": 0,
        "workflow_id": "wf_celery_dialog",
    }

    with patch("src.paper_to_voice.tasks.workers.get_llm", return_value=mock_llm):
        out = generate_dialog(state)

    assert out["Step"] == ["Step 1"]
    assert "**Jane:**" in out["Dialog"][0]
    assert len(out["indexed_dialogs"]) == 1
    assert out["indexed_dialogs"][0]["index"] == 0
