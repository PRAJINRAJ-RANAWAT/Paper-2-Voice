"""
Tests for Pydantic schema validation of research plans and bounded recovery strategy.
"""

import json
import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock

from src.paper_to_voice.core.models import ResearchPlan, ResearchStep, SubStepItem
from src.paper_to_voice.workflow.steps import (
    clean_json_markdown,
    parse_and_validate_plan,
    parse_json,
)


def test_valid_research_plan():
    """Verify that valid research plans parse and validate successfully."""
    valid_json = """
    [
        {
            "step": "Analyze dataset characteristics and preprocessing pipeline",
            "substeps": [
                {
                    "key": "Data normalization",
                    "value": "How are raw feature distributions scaled across train and test splits?"
                },
                {
                    "key": "Missing values",
                    "value": "What imputation strategy is used for sparse sensor readings?"
                }
            ]
        },
        {
            "step": "Evaluate model architecture and latency trade-offs",
            "substeps": [
                {
                    "key": "Inference speed",
                    "value": "What is the FLOP count and latency profile on edge devices?"
                }
            ]
        }
    ]
    """
    plan, workflow_plan = parse_and_validate_plan(valid_json)
    assert len(plan.steps) == 2
    assert plan.steps[0].step == "Analyze dataset characteristics and preprocessing pipeline"
    assert len(plan.steps[0].substeps) == 2
    assert len(workflow_plan) == 2
    assert workflow_plan[0]["step"] == plan.steps[0].step
    assert len(workflow_plan[0]["substeps"]) == 2


def test_markdown_code_fence_extraction():
    """Verify that markdown code fences (```json ... ```) are cleanly stripped."""
    wrapped_content = """Here is the research plan:
```json
[
    {
        "step": "Step 1: Literature review",
        "substeps": [
            {"key": "Key 1", "value": "Question 1"}
        ]
    }
]
```
Please let me know if you need modifications.
"""
    cleaned = clean_json_markdown(wrapped_content)
    assert cleaned.startswith("[")
    assert cleaned.endswith("]")
    plan, workflow_plan = parse_and_validate_plan(wrapped_content)
    assert len(plan.steps) == 1
    assert plan.steps[0].step == "Step 1: Literature review"


def test_malformed_json_raises_error():
    """Verify that malformed JSON raises an error."""
    malformed = """
    [
        {"step": "Incomplete JSON", "substeps": [{"key": "k1", "value": "v1"}
    """
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_and_validate_plan(malformed)


def test_missing_fields_validation_error():
    """Verify that missing required fields ('step' or 'substeps') trigger validation errors."""
    missing_step = """
    [
        {"substeps": [{"key": "k1", "value": "v1"}]}
    ]
    """
    with pytest.raises((ValidationError, ValueError)):
        parse_and_validate_plan(missing_step)

    missing_substeps = """
    [
        {"step": "Step without substeps"}
    ]
    """
    with pytest.raises((ValidationError, ValueError)):
        parse_and_validate_plan(missing_substeps)


def test_incorrect_types_validation_error():
    """Verify that incorrect data types trigger validation errors."""
    wrong_substeps_type = """
    [
        {"step": "Step with string substeps", "substeps": "not-a-list"}
    ]
    """
    with pytest.raises((ValidationError, ValueError)):
        parse_and_validate_plan(wrong_substeps_type)


def test_empty_plan_validation_error():
    """Verify that an empty research plan list is rejected."""
    with pytest.raises(ValueError, match="empty"):
        parse_and_validate_plan("[]")


def test_empty_step_or_substeps_rejected():
    """Verify that whitespace-only steps or empty substep lists are rejected."""
    empty_step_str = """
    [
        {"step": "   ", "substeps": [{"key": "k1", "value": "v1"}]}
    ]
    """
    with pytest.raises((ValidationError, ValueError)):
        parse_and_validate_plan(empty_step_str)

    empty_substeps_list = """
    [
        {"step": "Valid step", "substeps": []}
    ]
    """
    with pytest.raises((ValidationError, ValueError)):
        parse_and_validate_plan(empty_substeps_list)


def test_bounded_recovery_success():
    """Verify that parse_json successfully repairs invalid JSON via corrective model call."""
    invalid_content = "Here is some non-json text without proper formatting."
    mock_repair_response = MagicMock()
    mock_repair_response.content = """```json
    [
        {
            "step": "Recovered Step",
            "substeps": [{"key": "sub1", "value": "query1"}]
        }
    ]
    ```"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_repair_response

    state = {"content": invalid_content}
    result = parse_json(state, llm_for_recovery=mock_llm)

    assert "plan" in result
    assert len(result["plan"]) == 1
    assert result["plan"][0]["step"] == "Recovered Step"
    assert mock_llm.invoke.called


def test_recovery_exhaustion_raises_explicit_failure():
    """Verify that when recovery also fails, an explicit failure is raised (not an empty plan)."""
    invalid_content = "completely invalid text"
    mock_bad_response = MagicMock()
    mock_bad_response.content = "still invalid and not json"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_bad_response

    state = {"content": invalid_content}
    with pytest.raises(ValueError, match="Failed to produce a valid research plan"):
        parse_json(state, llm_for_recovery=mock_llm)
