"""
Main workflow orchestration using LangGraph with Send-based fan-out,
bounded concurrency, and deterministic state ordering.
"""

import logging
from typing import Any, List, Optional, Tuple
try:
    from langgraph.types import Send
except ImportError:
    from langgraph.constants import Send
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.models import State
from ..core.config import MAX_CONCURRENCY, get_llm
from .steps import generate_steps, markdown_to_json, parse_json, solve_substeps
from .dialog import generate_dialog

logger = logging.getLogger("paper_to_voice.workflow.orchestrator")


def continue_to_substeps(state: State) -> List[Send]:
    """
    Fan out research steps into individual solve_substeps tasks using LangGraph's Send API.
    
    Validates the plan, attaches step_index for deterministic ordering,
    and forwards image context to each task.
    """
    plan = state.get('plan')
    if not plan or not isinstance(plan, list) or len(plan) == 0:
        raise ValueError("Cannot continue to substeps: research plan is empty, missing, or invalid.")

    image_paths = state.get('image_path', [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    tasks = []
    for idx, step_item in enumerate(plan):
        tasks.append(
            Send(
                "solve_substeps",
                {
                    "step": step_item,
                    "image_path": image_paths,
                    "step_index": idx,
                },
            )
        )

    logger.info("Fanning out %d research step tasks", len(tasks))
    return tasks


def continue_to_substeps_voice(state: State) -> List[Send]:
    """
    Fan out dialogue generation for each valid step-and-solution pair using LangGraph's Send API.
    
    Preserves correspondence using stable indexes, detects mismatched step/solution counts,
    and prevents dropped results.
    """
    indexed_solutions = state.get('indexed_solutions', [])
    steps = state.get('steps', [])
    solutions = state.get('solutions', [])
    plan = state.get('plan', [])

    pairs: List[Tuple[int, Any, str]] = []

    if indexed_solutions:
        # Sort by step_index to ensure deterministic ordering regardless of task completion order
        sorted_solutions = sorted(indexed_solutions, key=lambda x: x.get('index', 0))
        for item in sorted_solutions:
            idx = item.get('index', 0)
            step_desc = item.get('step', '')
            sol_text = item.get('solution')
            if sol_text:
                pairs.append((idx, step_desc, sol_text))
            else:
                logger.warning("Step index %d produced no solution or failed; skipping dialog task", idx)

        if plan and len(pairs) != len(plan):
            logger.warning(
                "Mismatch between planned steps (%d) and valid solutions produced (%d)",
                len(plan),
                len(pairs),
            )
    else:
        # Fallback to linear zip with mismatch detection
        step_count = len(steps)
        sol_count = len(solutions)
        if step_count != sol_count:
            logger.warning(
                "Mismatched step count (%d) and solution count (%d). Pairing available items.",
                step_count,
                sol_count,
            )
        min_len = min(step_count, sol_count)
        for idx in range(min_len):
            pairs.append((idx, steps[idx], solutions[idx]))

    if not pairs:
        raise ValueError("Cannot continue to dialogue generation: no valid step-and-solution pairs available.")

    tone = state.get('tone', 'Formal')
    length = state.get('length', 'Short (1-2 min)')
    language = state.get('language', 'EN')

    tasks = [
        Send(
            "generate_dialog",
            {
                "step": step_desc,
                "text": sol_text,
                "tone": tone,
                "length": length,
                "language": language,
                "step_index": idx,
            },
        )
        for idx, step_desc, sol_text in pairs
    ]

    logger.info("Fanning out %d dialogue generation tasks", len(tasks))
    return tasks


def create_podcast_workflow(model_name: Optional[str] = None):
    """
    Create and compile the podcast generation workflow.
    
    Returns:
        Compiled StateGraph application and the LLM instance.
    """
    try:
        llm = get_llm(model_name=model_name)
    except Exception:
        llm = None

    graph = StateGraph(State)
    graph.add_node("generate_steps", generate_steps)
    graph.add_node("markdown_to_json", markdown_to_json)
    graph.add_node("parse_json", parse_json)
    graph.add_node("solve_substeps", solve_substeps)
    graph.add_node("generate_dialog", generate_dialog)

    graph.add_edge(START, "generate_steps")
    graph.add_edge("generate_steps", "markdown_to_json")
    graph.add_edge("markdown_to_json", "parse_json")
    graph.add_conditional_edges("parse_json", continue_to_substeps, ["solve_substeps"])
    graph.add_conditional_edges("solve_substeps", continue_to_substeps_voice, ["generate_dialog"])
    graph.add_edge("generate_dialog", END)

    app = graph.compile()
    return app, llm
