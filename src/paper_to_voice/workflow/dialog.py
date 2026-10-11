"""
Dialog generation for podcast creation with retry handling and deterministic ordering.
"""

import logging
import os
from ..core.config import get_llm
from ..utils.retry import retry_with_backoff

logger = logging.getLogger("paper_to_voice.workflow.dialog")

SYSTEM_PROMPT = """
You are a world-class podcast producer tasked with transforming the provided input text into an engaging and informative podcast script. The input may be unstructured or messy, sourced from PDFs or web pages. Your goal is to extract the most interesting and insightful content for a compelling podcast discussion.

# Steps to Follow:

1. **Analyze the Input:**
   Carefully examine the text, identifying key topics, points, and interesting facts or anecdotes that
   could drive an engaging podcast conversation. Disregard irrelevant information or formatting issues.

2. **Brainstorm Ideas:**
   In the `<scratchpad>`, creatively brainstorm ways to present the key points engagingly. Consider:
   - Analogies, storytelling techniques, or hypothetical scenarios to make content relatable
   - Ways to make complex topics accessible to a general audience
   - Thought-provoking questions to explore during the podcast
   - Creative approaches to fill any gaps in the information

3. **Craft the Dialogue:**
   Develop a natural, conversational flow between the host (Jane) and Dr. Sharma (the author or an expert on the topic). Incorporate:
   - The best ideas from your brainstorming session
   - Clear explanations of complex topics
   - An engaging and lively tone to captivate listeners
   - A balance of information and entertainment

   Rules for the dialogue:
   - The host (Jane) always initiates the conversation and interviews Dr. Sharma
   - Include thoughtful questions from the host to guide the discussion
   - Incorporate natural speech patterns, including occasional verbal fillers (e.g., "um," "well," "you know")
   - Allow for natural interruptions and back-and-forth between host and Dr. Sharma
   - Ensure Dr. Sharma's responses are substantiated by the input text, avoiding unsupported claims
   - Maintain a PG-rated conversation appropriate for all audiences
   - Avoid any marketing or self-promotional content from Dr. Sharma
   - The host concludes the conversation

4. **Summarize Key Insights:**
   Naturally weave a summary of key points into the closing part of the dialogue. This should feel like a casual conversation rather than a formal recap, reinforcing the main takeaways before signing off.

5. **Maintain Authenticity:**
   Throughout the script, strive for authenticity in the conversation. Include:
   - Moments of genuine curiosity or surprise from the host
   - Instances where Dr. Sharma might briefly struggle to articulate a complex idea
   - Light-hearted moments or humor when appropriate
   - Brief personal anecdotes or examples that relate to the topic (within the bounds of the input text)

6. **Consider Pacing and Structure:**
   Ensure the dialogue has a natural ebb and flow:
   - Start with a strong hook to grab the listener's attention
   - Gradually build complexity as the conversation progresses
   - Include brief "breather" moments for listeners to absorb complex information
   - End on a high note, perhaps with a thought-provoking question or a call-to-action for listeners
"""


def generate_dialog(state: dict) -> dict:
    """
    Generate podcast dialog from research content with exponential backoff retry.
    Supports both local execution and distributed Celery worker execution when USE_CELERY is enabled.
    """
    text = state.get('text', '')
    tone = state.get('tone', 'Formal')
    length = state.get('length', 'Short (1-2 min)')
    language = state.get('language', 'EN')
    step_idx = state.get('step_index', 0)
    workflow_id = state.get('workflow_id', 'default')

    from ..core.config import CELERY_TASK_TIMEOUT, USE_CELERY

    use_celery = os.getenv("USE_CELERY", "false").lower() in ("true", "1", "yes") or USE_CELERY
    if use_celery:
        try:
            from ..tasks.workers import generate_dialog_task
            from ..tasks.tracking import TaskStatus, tracker

            stable_task_id = f"workflow_{workflow_id}_dialog_{step_idx}"
            tracker.set_task_status(
                task_id=stable_task_id,
                status=TaskStatus.PENDING,
                workflow_id=workflow_id,
                stage="dialogue_generation",
                step_index=step_idx,
            )

            async_result = generate_dialog_task.apply_async(
                kwargs={
                    "step_desc": state.get('step', ''),
                    "finding_text": text,
                    "tone": tone,
                    "length": length,
                    "language": language,
                    "step_index": step_idx,
                    "workflow_id": workflow_id,
                },
                task_id=stable_task_id,
                queue="dialog_tasks",
            )
            result = async_result.get(timeout=CELERY_TASK_TIMEOUT)
            return {
                "Step": [result["step"]],
                "Finding": [result["finding"]],
                "Dialog": [result["dialog"]],
                "indexed_dialogs": [result],
            }
        except Exception as exc:
            logger.error("Distributed Celery generate_dialog_task failed (step %d): %s", step_idx, exc)
            raise exc

    # Local in-process execution fallback
    llm = get_llm()
    modified_system_prompt = SYSTEM_PROMPT
    modified_system_prompt += "\n\nPLEASE paraphrase the following TEXT in dialog format."

    if tone:
        modified_system_prompt += f"\n\nTONE: The tone of the podcast should be {tone}."
    if length:
        length_instructions = {
            "Short (1-2 min)": "Keep the podcast brief, around 1-2 minutes long.",
            "Medium (3-5 min)": "Aim for a moderate length, about 3-5 minutes.",
        }
        instruction = length_instructions.get(length, "Keep the podcast brief, around 1-2 minutes long.")
        modified_system_prompt += f"\n\nLENGTH: {instruction}"
    if language:
        modified_system_prompt += (
            f"\n\nOUTPUT LANGUAGE <IMPORTANT>: The podcast should be {language}."
        )

    messages = modified_system_prompt + '\nTEXT: ' + text

    def _call():
        return llm.invoke([messages])

    response = retry_with_backoff(_call, operation_name=f"generate_dialog_step_{step_idx}")

    return {
        "Step": [state.get('step', '')],
        "Finding": [text],
        "Dialog": [response.content],
        "indexed_dialogs": [{
            "index": step_idx,
            "step": state.get('step', ''),
            "finding": text,
            "dialog": response.content,
        }],
    }
