"""Kairos Agent — LangGraph Pipeline Orchestrator.

Defines the LangGraph state machine that orchestrates the full pipeline:
  Idea Agent -> Simulation Agent -> Video Editor Agent -> Human Review -> Publish

State is checkpointed to PostgreSQL at each node. Edge conditions handle
success, retry, escalation, and rejection routing.

Agent logic is plain Python — LangGraph orchestrates but agents have no
dependency on LangGraph abstractions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from kairos.config import get_settings
from kairos.exceptions import (
    ConceptGenerationError,
    PipelineError,
    SimulationExecutionError,
    ValidationError,
    VideoAssemblyError,
)
from kairos.models.contracts import (
    PipelineState,
    PipelineStatus,
    ReviewAction,
)
from kairos.pipeline.registry import get_pipeline

logger = logging.getLogger(__name__)

# Max retries before escalation
MAX_CONCEPT_ATTEMPTS = 3
MAX_SIMULATION_ITERATIONS = 5


# =============================================================================
# LangGraph State Schema (TypedDict ensures all keys are tracked/merged)
# =============================================================================


class PipelineGraphState(TypedDict, total=False):
    """TypedDict schema for LangGraph state tracking.

    Using TypedDict ensures LangGraph properly merges partial updates
    from nodes with the full state, rather than only keeping the last
    node's output.
    """

    pipeline_run_id: str
    pipeline: str
    status: str
    concept: dict[str, Any] | None
    concept_attempts: int
    simulation_code: str
    simulation_result: dict[str, Any] | None
    simulation_stats: dict[str, Any] | None
    validation_result: dict[str, Any] | None
    simulation_iteration: int
    raw_video_path: str
    captions: dict[str, Any] | None
    music_track: dict[str, Any] | None
    final_video_path: str
    video_output: dict[str, Any] | None
    review_action: str | None
    review_feedback: str
    total_cost_usd: float
    errors: list[str]


# =============================================================================
# Node Functions
# =============================================================================


async def idea_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Idea Agent to generate a concept.

    On failure, raises immediately so the pipeline stops.
    """
    attempt = state.get("concept_attempts", 0) + 1
    logger.info("[idea_node] Generating concept (attempt %d/%d)", attempt, MAX_CONCEPT_ATTEMPTS)

    pipeline_name = state.get("pipeline", "physics")
    adapter = get_pipeline(pipeline_name)
    agent = adapter.get_idea_agent()

    pipeline_state = _dict_to_pipeline_state(state)

    concept = await agent.generate_concept(pipeline_state)
    logger.info("[idea_node] OK Concept generated: %s (category=%s)", concept.title, concept.category.value)
    return {
        "status": PipelineStatus.SIMULATION_PHASE.value,
        "concept": concept.model_dump(mode="json"),
        "concept_attempts": attempt,
        "errors": [],
    }


async def simulation_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Simulation Agent loop.

    Generates code, executes in sandbox, validates, adjusts.
    The agent's internal run_loop handles iteration retries.
    If it still fails after all iterations, the error propagates here.
    """
    logger.info("[simulation_node] Starting simulation loop")

    pipeline_name = state.get("pipeline", "physics")
    adapter = get_pipeline(pipeline_name)
    agent = adapter.get_simulation_agent()

    pipeline_state = _dict_to_pipeline_state(state)

    updated_state = await agent.run_loop(pipeline_state.concept, pipeline_state)  # type: ignore[arg-type]

    # If run_loop completed but validation never passed, fail explicitly
    if updated_state.validation_result and not updated_state.validation_result.passed:
        msg = (
            f"Simulation failed validation after {updated_state.simulation_iteration} iterations: "
            f"{updated_state.validation_result.summary}"
        )
        logger.error("[simulation_node] %s", msg)
        raise SimulationExecutionError(msg)

    if not updated_state.raw_video_path:
        msg = f"No video produced after {updated_state.simulation_iteration} iterations"
        logger.error("[simulation_node] %s", msg)
        raise SimulationExecutionError(msg)

    logger.info(
        "[simulation_node] OK Simulation passed on iteration %d -- video: %s",
        updated_state.simulation_iteration,
        updated_state.raw_video_path,
    )
    return {
        "status": PipelineStatus.EDITING_PHASE.value,
        "simulation_code": updated_state.simulation_code,
        "simulation_result": (
            updated_state.simulation_result.model_dump(mode="json")
            if updated_state.simulation_result
            else None
        ),
        "simulation_stats": (
            updated_state.simulation_stats.model_dump(mode="json")
            if updated_state.simulation_stats
            else None
        ),
        "validation_result": (
            updated_state.validation_result.model_dump(mode="json")
            if updated_state.validation_result
            else None
        ),
        "simulation_iteration": updated_state.simulation_iteration,
        "raw_video_path": updated_state.raw_video_path,
        "errors": [],
    }


async def video_editor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Video Editor Agent to assemble the final video.

    Selects music, generates captions, composes video with FFmpeg.
    """
    logger.info("[video_editor_node] Assembling video")

    pipeline_name = state.get("pipeline", "physics")
    adapter = get_pipeline(pipeline_name)
    agent = adapter.get_video_editor_agent()

    pipeline_state = _dict_to_pipeline_state(state)
    concept = pipeline_state.concept

    if concept is None:
        raise VideoAssemblyError("No concept available for video editing")

    # Select music
    from kairos.models.contracts import SimulationStats

    stats_data = state.get("simulation_stats")
    stats = SimulationStats(**stats_data) if stats_data else SimulationStats(
        duration_sec=65.0,
        peak_body_count=100,
        avg_fps=30.0,
        min_fps=28.0,
        payoff_timestamp_sec=48.0,
        total_frames=1950,
        file_size_bytes=40_000_000,
    )

    logger.info("[video_editor_node] Selecting music track...")
    music = await agent.select_music(concept, stats)
    logger.info("[video_editor_node] OK Music: %s", music.title)

    # Generate captions
    logger.info("[video_editor_node] Generating captions...")
    captions = await agent.generate_captions(concept)
    logger.info("[video_editor_node] OK Captions generated (%d segments)", len(captions.segments))

    # Compose video
    raw_video_path = state.get("raw_video_path", "")
    logger.info("[video_editor_node] Composing final video with FFmpeg...")
    video_output = await agent.compose_video(
        raw_video_path=raw_video_path,
        music=music,
        captions=captions,
        concept=concept,
    )

    logger.info("[video_editor_node] OK Video assembled: %s", video_output.final_video_path)
    return {
        "status": PipelineStatus.PENDING_REVIEW.value,
        "captions": captions.model_dump(mode="json"),
        "music_track": music.model_dump(mode="json"),
        "final_video_path": video_output.final_video_path,
        "video_output": video_output.model_dump(mode="json"),
        "errors": [],
    }


async def human_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Human review gate — waits for review decision.

    In pipeline execution, this node sets status to PENDING_REVIEW.
    The actual review happens externally (via dashboard or CLI).
    The pipeline resumes when review_action is set.

    For automated testing, the review_action can be pre-set.
    """
    logger.info("[human_review_node] Video pending review")

    # If review_action is already set (e.g., from test or resumed state), process it
    review_action = state.get("review_action")
    if review_action:
        return {"status": PipelineStatus.PENDING_REVIEW.value}

    # Otherwise, mark as pending — pipeline will be resumed after review
    return {"status": PipelineStatus.PENDING_REVIEW.value}


async def publish_node(state: dict[str, Any]) -> dict[str, Any]:
    """Enqueue the approved video for publishing.

    Marks pipeline as approved and queues for distribution.
    """
    logger.info("[publish_node] Queueing video for publishing")

    return {
        "status": PipelineStatus.APPROVED.value,
    }


# =============================================================================
# Edge Condition Functions (Routing)
# =============================================================================


def route_after_idea(state: dict[str, Any]) -> str:
    """Route after Idea Agent node.

    - If concept generated -> simulation_agent
    - If max attempts exceeded -> __end__ (fail)
    - If failed but retries remain -> idea_agent (retry)
    """
    concept = state.get("concept")
    attempts = state.get("concept_attempts", 0)

    if concept is not None:
        return "simulation_agent"

    if attempts >= MAX_CONCEPT_ATTEMPTS:
        logger.error("Max concept attempts (%d) exceeded — pipeline failed", MAX_CONCEPT_ATTEMPTS)
        return "__end__"

    logger.info("Concept generation failed, retrying (attempt %d/%d)", attempts, MAX_CONCEPT_ATTEMPTS)
    return "idea_agent"


def route_after_simulation(state: dict[str, Any]) -> str:
    """Route after Simulation Agent node.

    - If simulation succeeded (has raw_video_path) -> video_editor_agent
    - If max iterations exceeded -> idea_agent (too_complex escalation)
    - If failed but retries remain -> simulation_agent (retry)
    """
    raw_video = state.get("raw_video_path")
    validation = state.get("validation_result")
    iteration = state.get("simulation_iteration", 0)

    # Success: have a video and validation passed
    if raw_video and validation:
        validation_passed = validation.get("passed", False) if isinstance(validation, dict) else False
        if validation_passed:
            return "video_editor_agent"

    # Max iterations — escalate to idea agent (too_complex)
    max_iter = MAX_SIMULATION_ITERATIONS
    if iteration >= max_iter:
        logger.warning("Max simulation iterations (%d) exceeded — escalating to Idea Agent", max_iter)
        return "idea_agent"

    # Retry
    return "simulation_agent"


def route_after_review(state: dict[str, Any]) -> str:
    """Route after Human Review node.

    - approved -> publish_queue
    - bad_concept -> idea_agent (new concept)
    - bad_simulation -> simulation_agent (re-simulate)
    - bad_edit / request_reedit -> video_editor_agent (re-edit)
    - None (waiting for review) -> __end__ (interrupt)
    """
    review_action = state.get("review_action")

    if review_action is None:
        # Pipeline pauses here — will be resumed after human review
        return "__end__"

    if review_action == ReviewAction.APPROVED.value:
        return "publish_queue"

    if review_action == ReviewAction.BAD_CONCEPT.value:
        return "idea_agent"

    if review_action == ReviewAction.BAD_SIMULATION.value:
        return "simulation_agent"

    if review_action in (ReviewAction.BAD_EDIT.value, ReviewAction.REQUEST_REEDIT.value):
        return "video_editor_agent"

    # Unknown action — fail safe
    logger.warning("Unknown review action: %s", review_action)
    return "__end__"


# =============================================================================
# Graph Construction
# =============================================================================


def build_pipeline_graph() -> StateGraph:
    """Build the LangGraph state machine for the content pipeline.

    Graph structure:
        START -> idea_agent -> simulation_agent -> video_editor_agent
              -> human_review -> publish_queue -> END

    With conditional edges for retry, escalation, and review routing.
    """
    graph = StateGraph(PipelineGraphState)

    # Add nodes
    graph.add_node("idea_agent", idea_node)
    graph.add_node("simulation_agent", simulation_node)
    graph.add_node("video_editor_agent", video_editor_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("publish_queue", publish_node)

    # Entry edge
    graph.add_edge(START, "idea_agent")

    # Conditional edges
    graph.add_conditional_edges(
        "idea_agent",
        route_after_idea,
        {
            "simulation_agent": "simulation_agent",
            "idea_agent": "idea_agent",
            "__end__": END,
        },
    )

    graph.add_conditional_edges(
        "simulation_agent",
        route_after_simulation,
        {
            "video_editor_agent": "video_editor_agent",
            "simulation_agent": "simulation_agent",
            "idea_agent": "idea_agent",
        },
    )

    graph.add_edge("video_editor_agent", "human_review")

    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "publish_queue": "publish_queue",
            "idea_agent": "idea_agent",
            "simulation_agent": "simulation_agent",
            "video_editor_agent": "video_editor_agent",
            "__end__": END,
        },
    )

    graph.add_edge("publish_queue", END)

    return graph


def compile_pipeline(
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the pipeline graph with optional checkpointing.

    Args:
        checkpointer: LangGraph checkpointer (MemorySaver, AsyncPostgresSaver, etc.).
                     If None, uses MemorySaver for in-memory checkpointing.

    Returns:
        Compiled LangGraph state machine.
    """
    graph = build_pipeline_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


async def run_pipeline(
    pipeline_name: str = "physics",
    *,
    checkpointer: Any | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline end-to-end.

    Args:
        pipeline_name: Pipeline to run (default: 'physics').
        checkpointer: Optional checkpointer for state persistence.
        thread_id: Optional thread ID for checkpoint keying.

    Returns:
        Final pipeline state dict.
    """
    compiled = compile_pipeline(checkpointer=checkpointer)

    pipeline_run_id = str(uuid4())
    initial_state: dict[str, Any] = {
        "pipeline_run_id": pipeline_run_id,
        "pipeline": pipeline_name,
        "status": PipelineStatus.RUNNING.value,
        "concept": None,
        "concept_attempts": 0,
        "simulation_code": "",
        "simulation_result": None,
        "simulation_stats": None,
        "validation_result": None,
        "simulation_iteration": 0,
        "raw_video_path": "",
        "captions": None,
        "music_track": None,
        "final_video_path": "",
        "video_output": None,
        "review_action": None,
        "review_feedback": "",
        "total_cost_usd": 0.0,
        "errors": [],
    }

    config = {"configurable": {"thread_id": thread_id or pipeline_run_id}}

    logger.info("="  * 60)
    logger.info("Pipeline run: %s", pipeline_run_id)
    logger.info("Pipeline:     %s", pipeline_name)
    logger.info("="  * 60)

    final_state = await compiled.ainvoke(initial_state, config)

    status = final_state.get("status", "unknown")
    logger.info("="  * 60)
    logger.info("Pipeline %s finished -- status: %s", pipeline_run_id, status)
    if final_state.get("final_video_path"):
        logger.info("Output video: %s", final_state["final_video_path"])
    if final_state.get("errors"):
        for err in final_state["errors"]:
            logger.warning("Accumulated error: %s", err)
    logger.info("="  * 60)

    return final_state


async def resume_pipeline(
    thread_id: str,
    *,
    updates: dict[str, Any] | None = None,
    checkpointer: Any | None = None,
) -> dict[str, Any]:
    """Resume a checkpointed pipeline run.

    Used after human review or to retry after transient failures.

    Args:
        thread_id: The thread ID used when the pipeline was started.
        updates: Optional state updates (e.g., review_action).
        checkpointer: Must be the same checkpointer used for the original run.

    Returns:
        Updated final state.
    """
    compiled = compile_pipeline(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    if updates:
        return await compiled.ainvoke(updates, config)

    return await compiled.ainvoke(None, config)


# =============================================================================
# Helpers
# =============================================================================


def _dict_to_pipeline_state(state: dict[str, Any]) -> PipelineState:
    """Convert a LangGraph state dict to a PipelineState object.

    LangGraph uses plain dicts for state. This helper reconstructs
    the typed PipelineState for use by agent methods that expect it.
    """
    from kairos.models.contracts import (
        CaptionSet,
        ConceptBrief,
        MusicTrackMetadata,
        SimulationResult,
        SimulationStats,
        ValidationResult,
    )

    ps = PipelineState(
        pipeline_run_id=state.get("pipeline_run_id", uuid4()),
        pipeline=state.get("pipeline", "physics"),
        status=PipelineStatus(state.get("status", "running")),
        concept_attempts=state.get("concept_attempts", 0),
        simulation_code=state.get("simulation_code", ""),
        simulation_iteration=state.get("simulation_iteration", 0),
        raw_video_path=state.get("raw_video_path", ""),
        final_video_path=state.get("final_video_path", ""),
        total_cost_usd=state.get("total_cost_usd", 0.0),
        errors=state.get("errors", []),
    )

    # Reconstruct nested Pydantic objects from dicts
    if state.get("concept") and isinstance(state["concept"], dict):
        ps.concept = ConceptBrief(**state["concept"])
    elif state.get("concept") and isinstance(state["concept"], ConceptBrief):
        ps.concept = state["concept"]

    if state.get("simulation_result") and isinstance(state["simulation_result"], dict):
        ps.simulation_result = SimulationResult(**state["simulation_result"])

    if state.get("simulation_stats") and isinstance(state["simulation_stats"], dict):
        ps.simulation_stats = SimulationStats(**state["simulation_stats"])

    if state.get("validation_result") and isinstance(state["validation_result"], dict):
        ps.validation_result = ValidationResult(**state["validation_result"])

    if state.get("captions") and isinstance(state["captions"], dict):
        ps.captions = CaptionSet(**state["captions"])

    if state.get("music_track") and isinstance(state["music_track"], dict):
        ps.music_track = MusicTrackMetadata(**state["music_track"])

    return ps
