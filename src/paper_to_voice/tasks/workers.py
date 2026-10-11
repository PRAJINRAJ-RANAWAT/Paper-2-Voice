"""
Distributed Celery worker tasks for Paper to Voice research solving and dialogue generation.
"""

import logging
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage

from .celery_app import celery_app
from .tracking import TaskStatus, tracker
from ..core.config import get_llm
from ..utils.retry import retry_with_backoff

logger = logging.getLogger("paper_to_voice.tasks.workers")


@celery_app.task(
    bind=True,
    name="solve_substeps_task",
    acks_late=True,
    max_retries=3,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=30,
    retry_jitter=True,
)
def solve_substeps_task(
    self,
    step_data: Dict[str, Any],
    image_paths: List[str],
    step_index: int,
    workflow_id: str = "default",
) -> Dict[str, Any]:
    """
    Distributed Celery task for solving research substeps.
    Idempotent: uses stable task ID and checks for previously computed results.
    """
    stable_task_id = self.request.id or f"workflow_{workflow_id}_step_{step_index}"
    logger.info("Executing solve_substeps_task (task_id=%s, step_index=%d)", stable_task_id, step_index)

    # 1. Idempotency check: avoid duplicate work on redelivery
    is_complete, cached = tracker.is_idempotent_complete(stable_task_id)
    if is_complete and cached:
        logger.info("Returning cached idempotent result for task %s", stable_task_id)
        return cached

    # 2. Update status to RUNNING
    tracker.set_task_status(
        task_id=stable_task_id,
        status=TaskStatus.RUNNING,
        workflow_id=workflow_id,
        stage="research_step",
        step_index=step_index,
    )

    try:
        llm = get_llm()
        if isinstance(step_data, dict):
            step_title = step_data.get("step", "")
            substeps = step_data.get("substeps", [])
        else:
            step_title = str(step_data)
            substeps = []

        qanda = " ".join([f"\n Question: {s} \n Answer:" for s in substeps])
        prompt = f"""You will be given instruction to analyze research papers. You need to understand the
instruction and solve all the questions mentioned in the list.
Keep the pair of Question and its answer in your response. Your response should be next to the keyword "Answer"

Instruction:
{step_title}
Questions:
{qanda}
"""
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                *[{"type": "image_url", "image_url": img} for img in image_paths],
            ]
        )

        def _call():
            return llm.invoke([message])

        response = retry_with_backoff(_call, operation_name=f"celery_solve_step_{step_index}")
        result = {
            "index": step_index,
            "step": step_title,
            "solution": response.content,
        }

        # 3. Store result for idempotency and mark SUCCEEDED
        tracker.store_idempotent_result(stable_task_id, result)
        tracker.set_task_status(
            task_id=stable_task_id,
            status=TaskStatus.SUCCEEDED,
            workflow_id=workflow_id,
            stage="research_step",
            step_index=step_index,
            result_metadata=result,
        )
        return result

    except Exception as exc:
        logger.error("solve_substeps_task failed (task_id=%s): %s", stable_task_id, exc)
        tracker.set_task_status(
            task_id=stable_task_id,
            status=TaskStatus.FAILED,
            workflow_id=workflow_id,
            stage="research_step",
            step_index=step_index,
            error=str(exc),
        )
        raise exc


@celery_app.task(
    bind=True,
    name="generate_dialog_task",
    acks_late=True,
    max_retries=3,
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=30,
    retry_jitter=True,
)
def generate_dialog_task(
    self,
    step_desc: str,
    finding_text: str,
    tone: str = "Formal",
    length: str = "Short (1-2 min)",
    language: str = "EN",
    step_index: int = 0,
    workflow_id: str = "default",
) -> Dict[str, Any]:
    """
    Distributed Celery task for dialogue generation.
    Idempotent: uses stable task ID and prevents duplicate result writes.
    """
    stable_task_id = self.request.id or f"workflow_{workflow_id}_dialog_{step_index}"
    logger.info("Executing generate_dialog_task (task_id=%s, step_index=%d)", stable_task_id, step_index)

    # 1. Idempotency check
    is_complete, cached = tracker.is_idempotent_complete(stable_task_id)
    if is_complete and cached:
        logger.info("Returning cached idempotent result for dialogue task %s", stable_task_id)
        return cached

    # 2. Update status to RUNNING
    tracker.set_task_status(
        task_id=stable_task_id,
        status=TaskStatus.RUNNING,
        workflow_id=workflow_id,
        stage="dialogue_generation",
        step_index=step_index,
    )

    try:
        from ..workflow.dialog import SYSTEM_PROMPT

        llm = get_llm()
        modified_prompt = SYSTEM_PROMPT + "\n\nPLEASE paraphrase the following TEXT in dialog format."
        if tone:
            modified_prompt += f"\n\nTONE: The tone of the podcast should be {tone}."
        if length:
            length_instructions = {
                "Short (1-2 min)": "Keep the podcast brief, around 1-2 minutes long.",
                "Medium (3-5 min)": "Aim for a moderate length, about 3-5 minutes.",
            }
            instr = length_instructions.get(length, "Keep the podcast brief, around 1-2 minutes long.")
            modified_prompt += f"\n\nLENGTH: {instr}"
        if language:
            modified_prompt += f"\n\nOUTPUT LANGUAGE <IMPORTANT>: The podcast should be {language}."

        messages = modified_prompt + "\nTEXT: " + finding_text

        def _call():
            return llm.invoke([messages])

        response = retry_with_backoff(_call, operation_name=f"celery_dialog_step_{step_index}")
        result = {
            "index": step_index,
            "step": step_desc,
            "finding": finding_text,
            "dialog": response.content,
        }

        # 3. Store result for idempotency and mark SUCCEEDED
        tracker.store_idempotent_result(stable_task_id, result)
        tracker.set_task_status(
            task_id=stable_task_id,
            status=TaskStatus.SUCCEEDED,
            workflow_id=workflow_id,
            stage="dialogue_generation",
            step_index=step_index,
            result_metadata=result,
        )
        return result

    except Exception as exc:
        logger.error("generate_dialog_task failed (task_id=%s): %s", stable_task_id, exc)
        tracker.set_task_status(
            task_id=stable_task_id,
            status=TaskStatus.FAILED,
            workflow_id=workflow_id,
            stage="dialogue_generation",
            step_index=step_index,
            error=str(exc),
        )
        raise exc
