"""
Data models and Pydantic validation schemas for Paper to Voice Assistant
"""

import operator
from typing import Annotated, Any, List, TypedDict, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class SubStepItem(BaseModel):
    """A single research substep with key and value strings."""
    key: str = Field(..., min_length=1, description="Title or key for the substep")
    value: str = Field(..., min_length=1, description="Description or query for the substep")

    @field_validator("key", "value")
    @classmethod
    def check_non_whitespace(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Substep key and value must not be empty or whitespace only")
        return v.strip()


class ResearchStep(BaseModel):
    """A research step containing a description and at least one substep."""
    step: str = Field(..., min_length=1, description="Main description of the research step")
    substeps: List[SubStepItem] = Field(..., min_length=1, description="List of substeps (at least one required)")

    @field_validator("step")
    @classmethod
    def check_non_whitespace_step(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Step description must not be empty or whitespace only")
        return v.strip()


class ResearchPlan(BaseModel):
    """Top-level model for the complete research plan."""
    steps: List[ResearchStep] = Field(..., min_length=1, description="List of research steps (at least one required)")

    @classmethod
    def from_raw(cls, data: Any) -> "ResearchPlan":
        """
        Parse raw data into a ResearchPlan, accepting either a list of step dicts
        or a top-level dictionary containing 'steps' or 'plan'.
        """
        if isinstance(data, list):
            if not data:
                raise ValueError("Research plan list cannot be empty")
            return cls(steps=data)
        elif isinstance(data, dict):
            if "steps" in data and isinstance(data["steps"], list):
                if not data["steps"]:
                    raise ValueError("Research plan steps cannot be empty")
                return cls(steps=data["steps"])
            elif "plan" in data and isinstance(data["plan"], list):
                if not data["plan"]:
                    raise ValueError("Research plan steps cannot be empty")
                return cls(steps=data["plan"])
            raise ValueError("Dictionary must contain a non-empty 'steps' or 'plan' list")
        raise ValueError(f"Expected a list or dict for research plan, got {type(data).__name__}")

    def to_workflow_plan(self) -> List[dict]:
        """
        Convert to the downstream format expected by solve_substeps:
        [{'step': step_description, 'substeps': [substep_val1, substep_val2, ...]}]
        """
        return [
            {
                "step": step.step,
                "substeps": [sub.value for sub in step.substeps],
            }
            for step in self.steps
        ]


# LangGraph State definitions
class State(TypedDict, total=False):
    image_path: Any
    steps: Annotated[list, operator.add]
    substeps: Annotated[list, operator.add]
    solutions: Annotated[list, operator.add]
    content: str
    plan: list
    Dialog: Annotated[list, operator.add]
    indexed_solutions: Annotated[list, operator.add]
    indexed_dialogs: Annotated[list, operator.add]


class StepState(TypedDict, total=False):
    step: Any
    image_path: Any
    solutions: str
    Dialog: str
    step_index: int


class DialogTaskState(TypedDict, total=False):
    step: Any
    text: str
    tone: str
    length: str
    language: str
    step_index: int


# Legacy / helper models kept for backward compatibility
class Task(BaseModel):
    task: str


class SubStep(BaseModel):
    substep: str
