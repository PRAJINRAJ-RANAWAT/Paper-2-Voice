"""
Workflow steps for processing research papers with schema validation and retry handling.
"""

import json
import logging
import os
import re
from typing import Any, List, Optional, Tuple, Union
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from ..core.models import ResearchPlan, State, StepState
from ..core.config import get_llm
from ..utils.retry import retry_with_backoff

logger = logging.getLogger("paper_to_voice.workflow.steps")


def clean_json_markdown(text: str) -> str:
    """
    Remove markdown code fences, triple quotes, and leading/trailing noise from JSON text.
    """
    if not isinstance(text, str):
        text = str(text)

    stripped = text.strip()

    # Match code blocks ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Match triple single quotes ''' ... '''
    match_single = re.search(r"'''(?:json)?\s*([\s\S]*?)\s*'''", stripped, re.IGNORECASE)
    if match_single:
        return match_single.group(1).strip()

    # Remove fence markers line-by-line as fallback
    lines = [
        line for line in stripped.splitlines()
        if not line.strip().startswith("```") and not line.strip().startswith("'''")
    ]
    cleaned = "\n".join(lines).strip()

    # Find the outermost JSON array or object
    start_bracket = cleaned.find("[")
    start_brace = cleaned.find("{")

    if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
        end_bracket = cleaned.rfind("]")
        if end_bracket != -1:
            return cleaned[start_bracket : end_bracket + 1].strip()
    elif start_brace != -1:
        end_brace = cleaned.rfind("}")
        if end_brace != -1:
            return cleaned[start_brace : end_brace + 1].strip()

    return cleaned


def parse_and_validate_plan(content: Union[str, list, dict]) -> Tuple[ResearchPlan, List[dict]]:
    """
    Parse and validate raw content into a ResearchPlan using Pydantic.
    
    Returns:
        Tuple of (ResearchPlan, workflow_plan_list)
        
    Raises:
        json.JSONDecodeError: When raw string is not valid JSON.
        ValidationError: When structure violates Pydantic schema constraints.
        ValueError: For empty or invalid data.
    """
    if isinstance(content, (list, dict)):
        raw_data = content
    else:
        cleaned = clean_json_markdown(content)
        if not cleaned:
            raise ValueError("Content contains no JSON data after stripping markdown fences.")
        raw_data = json.loads(cleaned)

    plan = ResearchPlan.from_raw(raw_data)
    return plan, plan.to_workflow_plan()


def attempt_plan_recovery(
    raw_content: str,
    error_message: str,
    llm: Optional[Any] = None,
) -> Tuple[ResearchPlan, List[dict]]:
    """
    Bounded corrective model retry: Prompt the LLM once to repair malformed JSON
    or schema violations into valid JSON conforming to the ResearchPlan schema.
    """
    if llm is None:
        try:
            llm = get_llm()
        except Exception as e:
            logger.error("Cannot attempt recovery: LLM unavailable (%s)", e)
            raise ValueError(f"Research plan validation failed and LLM is unavailable for recovery: {error_message}") from e

    logger.warning("Attempting bounded corrective model recovery for validation error: %s", error_message)

    corrective_prompt = f"""You previously returned research steps that could not be parsed or validated against the required schema.
Validation error:
{error_message}

Your previous output was:
{raw_content}

Please fix the output and return strictly valid JSON matching this schema:
[
  {{
    "step": "Clear non-empty description of step",
    "substeps": [
      {{
        "key": "Title or question for substep 1",
        "value": "Detailed question or instruction for substep 1"
      }}
    ]
  }}
]
Rules:
1. Return ONLY the JSON array inside a ```json ... ``` block.
2. Every step must have a non-empty "step" string and at least one substep.
3. Every substep must have non-empty "key" and "value" strings.
"""

    def _call_recovery():
        return llm.invoke([corrective_prompt])

    response = retry_with_backoff(_call_recovery, operation_name="corrective_plan_recovery", max_retries=1)
    repaired_content = response.content if hasattr(response, "content") else str(response)

    return parse_and_validate_plan(repaired_content)


def generate_steps(state: State) -> dict:
    """
    Generate research steps from paper images with exponential backoff retry.
    """
    llm = get_llm()
    prompt = """
    Consider you are a research scientist in artificial intelligence who is expert in understanding research papers.
    You will be given a research paper and you need to identify all the steps a researcher need to perform.
    Identify each steps and their substeps.
    """
    image_paths = state.get('image_path', [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    message = HumanMessage(content=[
        {'type': 'text', 'text': prompt},
        *[{"type": 'image_url', 'image_url': img} for img in image_paths]
    ])

    def _call():
        return llm.invoke([message])

    response = retry_with_backoff(_call, operation_name="generate_steps")
    return {"content": [response.content], "image_path": image_paths}


def markdown_to_json(state: State) -> dict:
    """
    Convert markdown content to JSON format with exponential backoff retry.
    """
    llm = get_llm()
    raw_content = state.get('content', '')
    if isinstance(raw_content, list):
        raw_content = "\n".join(str(c) for c in raw_content)

    prompt = """
    You are given a markdown content and you need to parse this data into json format. Follow correctly key and value
    pairs for each bullet point.
    Follow following schema strictly.

    schema:
    [
    {
      "step": "description of step 1 ",
      "substeps": [
        {
          "key": "title of sub step 1 of step 1",
          "value": "description of sub step 1 of step 1"
        },
        {
          "key": "title of sub step 2 of step 1",
          "value": "description of sub step 2 of step 1"
        }]},
        {
      "step": "description of step 2",
      "substeps": [
        {
          "key": "title of sub step 1 of step 2",
          "value": "description of sub step 1 of step 2"
        },
        {
          "key": "title of sub step 2 of step 2",
          "value": "description of sub step 2 of step 2"
        }]}]

    Content:
    %s
    """ % raw_content

    def _call():
        return llm.invoke([prompt])

    str_response = retry_with_backoff(_call, operation_name="markdown_to_json")
    return {'content': str_response.content, "image_path": state.get('image_path', [])}


def parse_json(state: State, llm_for_recovery: Optional[Any] = None) -> dict:
    """
    Parse and validate research plan JSON from state['content'].
    
    Explicitly handles json.JSONDecodeError and pydantic.ValidationError.
    Applies bounded corrective model recovery when validation fails.
    Never silently converts invalid output into an empty plan.
    """
    raw_content = state.get('content', '')
    if isinstance(raw_content, list):
        raw_content = "\n".join(str(c) for c in raw_content)

    try:
        plan, workflow_plan = parse_and_validate_plan(raw_content)
        logger.info("Successfully validated research plan with %d steps", len(plan.steps))
        return {"plan": workflow_plan}
    except (json.JSONDecodeError, ValidationError, ValueError) as err:
        err_msg = f"{err.__class__.__name__}: {str(err)}"
        logger.warning("Research plan JSON parsing/validation error: %s", err_msg)

        # Attempt bounded corrective model retry
        try:
            plan, workflow_plan = attempt_plan_recovery(
                raw_content=raw_content,
                error_message=err_msg,
                llm=llm_for_recovery,
            )
            logger.info("Corrective recovery succeeded: validated %d steps", len(plan.steps))
            return {"plan": workflow_plan}
        except Exception as recovery_err:
            logger.error("Corrective recovery exhausted or failed: %s", recovery_err)
            raise ValueError(
                f"Failed to produce a valid research plan after validation and corrective recovery. Reason: {err_msg}"
            ) from recovery_err


def solve_substeps(state: StepState) -> dict:
    """
    Solve substeps for an individual research step.
    Supports both local execution and distributed Celery worker execution when USE_CELERY is enabled.
    Maintains step_index for deterministic aggregation.
    """
    inp = state['step']
    step_idx = state.get('step_index', 0)
    workflow_id = state.get('workflow_id', 'default')

    if isinstance(inp, dict):
        step_title = inp.get('step', '')
        substeps = inp.get('substeps', [])
    else:
        step_title = str(inp)
        substeps = []

    image_paths = state.get('image_path', [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    from ..core.config import CELERY_TASK_TIMEOUT, USE_CELERY

    use_celery = os.getenv("USE_CELERY", "false").lower() in ("true", "1", "yes") or USE_CELERY
    if use_celery:
        try:
            from ..tasks.workers import solve_substeps_task
            from ..tasks.tracking import TaskStatus, tracker

            stable_task_id = f"workflow_{workflow_id}_step_{step_idx}"
            tracker.set_task_status(
                task_id=stable_task_id,
                status=TaskStatus.PENDING,
                workflow_id=workflow_id,
                stage="research_step",
                step_index=step_idx,
            )

            async_result = solve_substeps_task.apply_async(
                kwargs={
                    "step_data": inp,
                    "image_paths": image_paths,
                    "step_index": step_idx,
                    "workflow_id": workflow_id,
                },
                task_id=stable_task_id,
                queue="research_steps",
            )
            result = async_result.get(timeout=CELERY_TASK_TIMEOUT)
            return {
                "steps": [result["step"]],
                "solutions": [result["solution"]],
                "indexed_solutions": [result],
            }
        except Exception as exc:
            logger.error("Distributed Celery solve_substeps_task failed (step %d): %s", step_idx, exc)
            raise exc

    # Local in-process execution fallback
    llm = get_llm()
    qanda = ' '.join([
        f'\n Question: {substep} \n Answer:' 
        for substep in substeps
    ])
    
    prompt = f"""You will be given instruction to analyze research papers. You need to understand the
    instruction and solve all the questions mentioned in the list.
    Keep the pair of Question and its answer in your response. Your response should be next to the keyword "Answer"

    Instruction:
    {step_title}
    Questions:
    {qanda}
    """

    message = HumanMessage(content=[
        {'type': 'text', 'text': prompt},
        *[{"type": 'image_url', 'image_url': img} for img in image_paths]
    ])

    def _call():
        return llm.invoke([message])

    response = retry_with_backoff(_call, operation_name=f"solve_substeps_step_{step_idx}")

    return {
        "steps": [step_title],
        "solutions": [response.content],
        "indexed_solutions": [{
            "index": step_idx,
            "step": step_title,
            "solution": response.content,
        }],
    }
