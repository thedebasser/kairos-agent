"""Kairos Agent — LangGraph Pipeline Orchestrator.

Defines the LangGraph state machine that orchestrates the full pipeline:
  Idea Agent -> Simulation Agent -> Video Editor Agent
  -> Video Review -> Audio Review -> Human Review -> Publish

State is checkpointed to PostgreSQL at each node. Edge conditions handle
success, retry, escalation, and rejection routing.

Agent logic is plain Python — LangGraph orchestrates but agents have no
dependency on LangGraph abstractions.
"""

from __future__ import annotations

import hashlib
import json
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
from kairos.schemas.contracts import (
    IdeaAgentInput,
    PipelineState,
    PipelineStatus,
    ReviewAction,
)
from kairos.orchestrator.registry import get_pipeline
from kairos.ai.llm.cache import get_cache, init_cache
from kairos.ai.tracing.tracer import get_tracer, init_tracer
from kairos.ai.llm.routing import collect_llm_calls, collect_thinking

logger = logging.getLogger(__name__)

# Max retries before escalation
MAX_CONCEPT_ATTEMPTS = 3
MAX_SIMULATION_ITERATIONS = 5


def _state_input_hash(state: dict[str, Any], keys: list[str]) -> str:
    """Compute a short deterministic hash of selected state fields.

    Used as the ``input_hash`` for step-level caching so that different
    inputs produce different cache entries (Finding 2.3).
    """
    parts = {k: state.get(k) for k in keys}
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# =============================================================================
# LangGraph State Schema (TypedDict ensures all keys are tracked/merged)
# =============================================================================


class PipelineGraphState(TypedDict, total=False):
    """TypedDict schema for LangGraph state tracking.

    Using TypedDict ensures LangGraph properly merges partial updates
    from nodes with the full state, rather than only keeping the last
    node's output.

    Field names MUST mirror ``PipelineState`` (contracts.py).  A startup
    assertion enforces parity (Finding 2.1).
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
    # Added — previously injected ad-hoc by simulation_node
    theme_name: str
    # Review phase — automated (video + audio review agents)
    video_review_result: dict[str, Any] | None
    audio_review_result: dict[str, Any] | None
    video_review_attempts: int
    audio_review_attempts: int
    # Output versioning — tracks review-triggered re-renders (§1)
    output_version: int


# ── Parity check (Finding 2.1) ─────────────────────────────────────────

def _assert_state_parity() -> None:
    """Verify PipelineGraphState and PipelineState share the same field names.

    Run at import time.  Mismatches indicate a field was added to one
    model but not the other—this will cause silent data loss.
    """
    graph_keys = set(PipelineGraphState.__annotations__)
    pydantic_keys = set(PipelineState.model_fields)
    missing_in_graph = pydantic_keys - graph_keys
    missing_in_pydantic = graph_keys - pydantic_keys
    if missing_in_graph or missing_in_pydantic:
        parts: list[str] = []
        if missing_in_graph:
            parts.append(f"  Missing in PipelineGraphState: {missing_in_graph}")
        if missing_in_pydantic:
            parts.append(f"  Missing in PipelineState: {missing_in_pydantic}")
        logger.warning(
            "State model parity violation (Finding 2.1):\n%s",
            "\n".join(parts),
        )

_assert_state_parity()


# =============================================================================
# Node Functions
# =============================================================================


async def idea_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Idea Agent to generate a concept.

    PipelineError subclasses are caught so the routing function can
    decide whether to retry or fail.  InfrastructureError propagates
    and crashes the process as intended.
    """
    # ── Step cache check ────────────────────────────────────────────────
    cache = get_cache()
    idea_hash = _state_input_hash(state, ["pipeline", "concept_attempts"])
    if cache:
        cached = cache.get_step("idea_node", idea_hash)
        if cached:
            logger.info("[idea_node] Returning cached result (no LLM call)")
            return cached

    attempt = state.get("concept_attempts", 0) + 1
    logger.info("[idea_node] Generating concept (attempt %d/%d)", attempt, MAX_CONCEPT_ATTEMPTS)

    pipeline_name = state.get("pipeline", "physics")
    adapter = get_pipeline(pipeline_name)
    agent = adapter.get_idea_agent()

    # Narrow DTO: only pass what the agent actually needs (Finding 2.2)
    idea_input = IdeaAgentInput(pipeline=pipeline_name)

    tracer = get_tracer()
    with tracer.step("idea_agent", 1, attempt=attempt) as span:
        try:
            concept = await agent.generate_concept(idea_input)
        except PipelineError as exc:
            logger.error("[idea_node] Concept generation failed: %s", exc)
            span.fail(str(exc))
            collect_llm_calls()
            return {
                "status": "error",
                "concept": None,
                "concept_attempts": attempt,
                "errors": [*state.get("errors", []), f"idea_node: {exc}"],
            }

        logger.info("[idea_node] OK Concept generated: %s (category=%s)", concept.title, concept.category.value)
        collect_thinking()
        collect_llm_calls()
        span.set_outputs({"concept_title": concept.title, "category": concept.category.value})

        result = {
            "status": PipelineStatus.SIMULATION_PHASE.value,
            "concept": concept.model_dump(mode="json"),
            "concept_attempts": attempt,
            "errors": [],
        }

        # ── Step cache store ────────────────────────────────────────────────
        if cache:
            cache.put_step("idea_node", result, idea_hash)

        return result


async def simulation_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Simulation Agent loop.

    Generates code, executes in sandbox, validates, adjusts.
    The agent's internal run_loop handles iteration retries.
    If it still fails after all iterations, the error propagates here.
    """
    # ── Step cache check ────────────────────────────────────────────────
    cache = get_cache()
    sim_hash = _state_input_hash(state, ["pipeline", "concept"])
    if cache:
        cached = cache.get_step("simulation_node", sim_hash)
        if cached:
            logger.info("[simulation_node] Returning cached result (no LLM/sandbox calls)")
            return cached

    logger.info("[simulation_node] Starting simulation loop")

    pipeline_name = state.get("pipeline", "physics")
    adapter = get_pipeline(pipeline_name)
    agent = adapter.get_simulation_agent()

    # Narrow DTO: only pass concept, not entire state (Finding 2.2)
    pipeline_state = _dict_to_pipeline_state(state)
    concept = pipeline_state.concept

    tracer = get_tracer()
    with tracer.step("simulation_agent", 2) as span:
        try:
            loop_result = await agent.run_loop(concept)  # type: ignore[arg-type]
        except Exception as exc:
            logger.error("[simulation_node] Simulation agent failed: %s", exc)
            span.fail(str(exc))
            collect_llm_calls()
            return {
                "status": PipelineStatus.SIMULATION_PHASE.value,
                "simulation_iteration": pipeline_state.simulation_iteration,
                "errors": [*state.get("errors", []), f"simulation_node: {exc}"],
            }

        # Map SimulationLoopResult → state update dict (Finding 2.2)
        # If run_loop completed but validation never passed, warn but proceed if video exists
        if loop_result.validation_result and not loop_result.validation_result.passed:
            msg = (
                f"Simulation validation incomplete after {loop_result.simulation_iteration} iterations: "
                f"{loop_result.validation_result.summary}"
            )
            if loop_result.raw_video_path:
                logger.warning("[simulation_node] %s — proceeding with rendered video", msg)
            else:
                logger.error("[simulation_node] %s", msg)
                span.fail(msg)
                return {
                    "status": "error",
                    "simulation_iteration": loop_result.simulation_iteration,
                    "errors": [*state.get("errors", []), f"simulation_node: {msg}"],
                }

        if not loop_result.raw_video_path:
            msg = f"No video produced after {loop_result.simulation_iteration} iterations"
            logger.error("[simulation_node] %s", msg)
            span.fail(msg)
            return {
                "status": "error",
                "simulation_iteration": loop_result.simulation_iteration,
                "errors": [*state.get("errors", []), f"simulation_node: {msg}"],
            }

        logger.info(
            "[simulation_node] OK Simulation passed on iteration %d -- video: %s",
            loop_result.simulation_iteration,
            loop_result.raw_video_path,
        )

        thinking_entries = collect_thinking()
        collect_llm_calls()
        span.set_outputs({
            "simulation_iteration": loop_result.simulation_iteration,
            "raw_video_path": loop_result.raw_video_path or "",
        })

        # Save the raw simulation code via file writer
        writer = tracer._writer
        if writer and loop_result.simulation_code:
            writer.write_file("steps/02_simulation_agent/simulation_code.py", loop_result.simulation_code)

        sim_result = {
            "status": PipelineStatus.EDITING_PHASE.value,
            "simulation_code": loop_result.simulation_code,
            "simulation_result": (
                loop_result.simulation_result.model_dump(mode="json")
                if loop_result.simulation_result
                else None
            ),
            "simulation_stats": (
                loop_result.simulation_stats.model_dump(mode="json")
                if loop_result.simulation_stats
                else None
            ),
            "validation_result": (
                loop_result.validation_result.model_dump(mode="json")
                if loop_result.validation_result
                else None
            ),
            "simulation_iteration": loop_result.simulation_iteration,
            "raw_video_path": loop_result.raw_video_path,
            "errors": [],
        }

        # Extract theme_name from theme_config.json if available
        if loop_result.raw_video_path:
            from pathlib import Path as _Path
            _theme_cfg_path = _Path(loop_result.raw_video_path).parent / "theme_config.json"
            if _theme_cfg_path.exists():
                try:
                    import json as _json
                    sim_result["theme_name"] = _json.loads(
                        _theme_cfg_path.read_text(encoding="utf-8")
                    ).get("theme_name", "")
                except Exception:
                    pass

        # ── Step cache store ────────────────────────────────────────────────
        if cache:
            cache.put_step("simulation_node", sim_result, sim_hash)

        # ── Learning loop: store training example + update category knowledge ──
        try:
            from kairos.ai.learning.learning_loop import (
                record_training_example,
                update_category_knowledge,
            )
            from kairos.schemas.contracts import ValidationResult as _VR

            _concept_data = state.get("concept") or {}
            _category = _concept_data.get("category", "") if isinstance(_concept_data, dict) else ""
            _val_raw = loop_result.validation_result
            _passed = _val_raw.passed if _val_raw else False

            # Collect reasoning + thinking from the thinking buffer
            _reasoning = ""
            _thinking = ""
            if thinking_entries:
                _thinking = "\n---\n".join(
                    entry.get("thinking", "") for entry in thinking_entries if entry.get("thinking")
                )
                # First thinking entry is usually the generation reasoning
                if thinking_entries:
                    _reasoning = thinking_entries[0].get("thinking", "")[:2000]

            await record_training_example(
                pipeline=pipeline_name,
                category=_category,
                concept_brief=_concept_data,
                simulation_code=loop_result.simulation_code,
                validation_passed=_passed,
                iteration_count=loop_result.simulation_iteration,
                reasoning=_reasoning,
                thinking_content=_thinking,
            )

            await update_category_knowledge(
                pipeline=pipeline_name,
                category=_category,
                iteration_count=loop_result.simulation_iteration,
                validation_result=_val_raw,
            )
        except Exception as _ll_exc:
            logger.debug("Learning loop post-simulation hook skipped: %s", _ll_exc)

        return sim_result


async def video_editor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Video Editor Agent to assemble the final video.

    Selects music, generates captions, composes video with FFmpeg.
    PipelineError subclasses are caught so the graph can route to
    retry or fail gracefully.  InfrastructureError propagates.
    """
    # ── Step cache check ────────────────────────────────────────────────
    cache = get_cache()
    editor_hash = _state_input_hash(state, ["pipeline", "concept", "raw_video_path"])
    if cache:
        cached = cache.get_step("video_editor_node", editor_hash)
        if cached:
            logger.info("[video_editor_node] Returning cached result (no LLM/FFmpeg calls)")
            return cached

    logger.info("[video_editor_node] Assembling video")

    pipeline_name = state.get("pipeline", "physics")
    adapter = get_pipeline(pipeline_name)
    agent = adapter.get_video_editor_agent()

    # Point the agent's output dir into the versioned runs/ folder
    # so the final video lands in runs/{run_id}/output/v{n}/
    tracer = get_tracer()
    output_version = state.get("output_version", 0) + 1
    agent._output_dir = tracer.get_output_dir(output_version)

    pipeline_state = _dict_to_pipeline_state(state)
    concept = pipeline_state.concept

    if concept is None:
        # No concept is a logic error, not recoverable — still use return
        return {
            "status": "error",
            "errors": [*state.get("errors", []), "video_editor_node: No concept available"],
        }

    with tracer.step("video_editor", 3) as span:
        try:
            # Select music
            from kairos.schemas.contracts import SimulationStats

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
            _t0 = time.monotonic()
            music = await agent.select_music(concept, stats)
            tracer.action(
                "music:select",
                input_summary=f"concept={concept.title[:60]}",
                output_summary=f"{music.track_id} | {music.bpm} BPM | {music.mood}",
                status="success",
                duration_ms=int((time.monotonic() - _t0) * 1000),
            )
            logger.info("[video_editor_node] OK Music: %s", music.track_id)

            # Generate captions
            logger.info("[video_editor_node] Generating captions...")
            _t0 = time.monotonic()
            theme_name = state.get("theme_name", "")
            captions = await agent.generate_captions(concept, theme_name=theme_name)
            _caption_texts = " / ".join(c.text for c in captions.captions[:3]) if captions.captions else ""
            tracer.action(
                "captions:generate",
                input_summary=f"concept={concept.title[:60]}, theme={theme_name or 'none'}",
                output_summary=f"{len(captions.captions)} captions: {_caption_texts[:120]}",
                status="success",
                duration_ms=int((time.monotonic() - _t0) * 1000),
            )
            logger.info("[video_editor_node] OK Captions generated (%d captions)", len(captions.captions))

            # Compose video
            raw_video_path = state.get("raw_video_path", "")
            logger.info("[video_editor_node] Composing final video with FFmpeg...")
            _t0 = time.monotonic()
            video_output = await agent.compose_video(
                raw_video_path=raw_video_path,
                music=music,
                captions=captions,
                concept=concept,
            )
            _final_size = 0
            try:
                import os as _os
                _final_size = _os.path.getsize(video_output.final_video_path) // 1024 // 1024
            except Exception:
                pass
            tracer.action(
                "ffmpeg:compose",
                input_summary=f"raw={raw_video_path.split('/')[-1] if raw_video_path else '?'}, music={music.track_id}",
                output_summary=f"{video_output.final_video_path.split('/')[-1]}  ({_final_size}MB)",
                status="success",
                duration_ms=int((time.monotonic() - _t0) * 1000),
            )
        except PipelineError as exc:
            logger.error("[video_editor_node] Video assembly failed: %s", exc)
            span.fail(str(exc))
            collect_llm_calls()
            return {
                "status": "error",
                "errors": [*state.get("errors", []), f"video_editor_node: {exc}"],
            }

        logger.info("[video_editor_node] OK Video assembled: %s", video_output.final_video_path)
        collect_thinking()
        collect_llm_calls()
        span.set_outputs({"final_video_path": video_output.final_video_path})

        editor_result = {
            "status": PipelineStatus.PENDING_REVIEW.value,
            "captions": captions.model_dump(mode="json"),
            "music_track": music.model_dump(mode="json"),
            "final_video_path": video_output.final_video_path,
            "video_output": video_output.model_dump(mode="json"),
            "output_version": state.get("output_version", 0) + 1,
            "errors": [],
        }

        # ── Step cache store ────────────────────────────────────────────────
        if cache:
            cache.put_step("video_editor_node", editor_result, editor_hash)

        return editor_result


async def human_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Human review gate — waits for review decision.

    In pipeline execution, this node sets status to PENDING_REVIEW.
    The actual review happens externally (via dashboard or CLI).
    The pipeline resumes when review_action is set.

    For automated testing, the review_action can be pre-set.
    """
    # If review_action is already set (e.g., from test or resumed state), process it
    review_action = state.get("review_action")
    if review_action:
        logger.info(
            "[human_review_node] Resuming with review_action=%s (output=%s)",
            review_action,
            state.get("output_id", "unknown"),
        )
    else:
        logger.info("[human_review_node] Video pending review — pipeline will pause")

    return {"status": PipelineStatus.PENDING_REVIEW.value}


async def publish_node(state: dict[str, Any]) -> dict[str, Any]:
    """Enqueue the approved video for publishing.

    Marks pipeline as approved and queues for distribution.
    """
    logger.info("[publish_node] Queueing video for publishing")

    return {
        "status": PipelineStatus.APPROVED.value,
    }


# ── Review nodes (video + audio) ────────────────────────────────────────

MAX_VIDEO_REVIEW_ATTEMPTS = 2
MAX_AUDIO_REVIEW_ATTEMPTS = 2


async def video_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Video Review Agent on the final composed video.

    Uses a VLM (default: Qwen3-VL-8B) to inspect frames for visual/physics
    quality issues.  Escalates to a heavier model (Qwen3-VL-30B-A3B) for
    uncertain clips.

    On failure: routes back to simulation_agent for a retry.
    On pass:    routes to audio_review.
    """
    attempt = state.get("video_review_attempts", 0) + 1
    logger.info("[video_review_node] Reviewing video (attempt %d/%d)", attempt, MAX_VIDEO_REVIEW_ATTEMPTS)

    final_video_path = state.get("final_video_path", "")
    if not final_video_path:
        logger.error("[video_review_node] No final_video_path in state")
        return {
            "status": "error",
            "video_review_attempts": attempt,
            "errors": [*state.get("errors", []), "video_review_node: No final_video_path available"],
        }

    pipeline_name = state.get("pipeline", "physics")
    adapter = get_pipeline(pipeline_name)
    agent = adapter.get_video_review_agent()

    # Reconstruct concept for context
    concept = None
    if state.get("concept"):
        try:
            from kairos.schemas.contracts import ConceptBrief
            concept_data = state["concept"]
            if isinstance(concept_data, dict):
                concept = ConceptBrief(**concept_data)
            elif isinstance(concept_data, ConceptBrief):
                concept = concept_data
        except Exception:
            pass

    tracer = get_tracer()
    with tracer.step("video_review", 4, attempt=attempt) as span:
        try:
            review_result = await agent.review_video(final_video_path, concept)
        except Exception as exc:
            logger.error("[video_review_node] Video review failed: %s", exc)
            span.fail(str(exc))
            collect_llm_calls()
            # On reviewer failure, pass through with warning (don't block pipeline)
            return {
                "video_review_result": None,
                "video_review_attempts": attempt,
                "errors": [*state.get("errors", []), f"video_review_node: {exc}"],
            }

        logger.info("[video_review_node] Result: %s", review_result.summary)
        collect_llm_calls()
        span.set_outputs({"passed": review_result.passed, "summary": review_result.summary[:200]})

        return {
            "video_review_result": review_result.model_dump(mode="json"),
            "video_review_attempts": attempt,
        }


async def audio_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the Audio Review Agent on the final composed video's audio.

    Uses an omni-modal LLM (default: Qwen2.5-Omni-7B) + FFmpeg ebur128
    loudness analysis to inspect the audio mix.

    On failure: routes back to video_editor_agent for a re-edit.
    On pass:    routes to human_review.
    """
    attempt = state.get("audio_review_attempts", 0) + 1
    logger.info("[audio_review_node] Reviewing audio (attempt %d/%d)", attempt, MAX_AUDIO_REVIEW_ATTEMPTS)

    final_video_path = state.get("final_video_path", "")
    if not final_video_path:
        logger.error("[audio_review_node] No final_video_path in state")
        return {
            "status": "error",
            "audio_review_attempts": attempt,
            "errors": [*state.get("errors", []), "audio_review_node: No final_video_path available"],
        }

    pipeline_name = state.get("pipeline", "physics")
    adapter = get_pipeline(pipeline_name)
    agent = adapter.get_audio_review_agent()

    # Get expected transcript from captions if available
    expected_transcript = ""
    captions_data = state.get("captions")
    if captions_data and isinstance(captions_data, dict):
        # Extract text from caption entries
        for cap in captions_data.get("captions", []):
            if isinstance(cap, dict):
                expected_transcript += cap.get("text", "") + " "
        expected_transcript = expected_transcript.strip()

    tracer = get_tracer()
    with tracer.step("audio_review", 5, attempt=attempt) as span:
        try:
            review_result = await agent.review_audio(final_video_path, expected_transcript)
        except Exception as exc:
            logger.error("[audio_review_node] Audio review failed: %s", exc)
            span.fail(str(exc))
            collect_llm_calls()
            # On reviewer failure, pass through with warning
            return {
                "audio_review_result": None,
                "audio_review_attempts": attempt,
                "errors": [*state.get("errors", []), f"audio_review_node: {exc}"],
            }

        logger.info("[audio_review_node] Result: %s", review_result.summary)
        collect_llm_calls()
        span.set_outputs({"passed": review_result.passed, "summary": review_result.summary[:200]})

        return {
            "audio_review_result": review_result.model_dump(mode="json"),
            "audio_review_attempts": attempt,
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

    - If simulation produced a video -> video_editor_agent
    - If max iterations exceeded without video -> idea_agent (too_complex escalation)
    - If failed but retries remain -> simulation_agent (retry)
    """
    raw_video = state.get("raw_video_path")

    # Success: have a rendered video (even if validation was partial)
    if raw_video:
        return "video_editor_agent"

    # Max iterations — escalate to idea agent (too_complex)
    iteration = state.get("simulation_iteration", 0)
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
        logger.info("[route_after_review] No review action yet — pausing pipeline")
        return "__end__"

    if review_action == ReviewAction.APPROVED.value:
        logger.info("[route_after_review] Approved — routing to publish_queue")
        return "publish_queue"

    if review_action == ReviewAction.BAD_CONCEPT.value:
        logger.info("[route_after_review] Bad concept — routing to idea_agent for new concept")
        return "idea_agent"

    if review_action == ReviewAction.BAD_SIMULATION.value:
        logger.info("[route_after_review] Bad simulation — routing to simulation_agent")
        return "simulation_agent"

    if review_action in (ReviewAction.BAD_EDIT.value, ReviewAction.REQUEST_REEDIT.value):
        logger.info("[route_after_review] %s — routing to video_editor_agent", review_action)
        return "video_editor_agent"

    # Unknown action — fail safe
    logger.warning("[route_after_review] Unknown review action: %s — ending pipeline", review_action)
    return "__end__"


def route_after_video_editor(state: dict[str, Any]) -> str:
    """Route after Video Editor Agent node.

    - If video assembled successfully -> video_review (automated review)
    - If error -> __end__ (pipeline failed)
    """
    status = state.get("status", "")
    if status == "error":
        logger.error("Video editor failed — pipeline ending")
        return "__end__"
    return "video_review"


def route_after_video_review(state: dict[str, Any]) -> str:
    """Route after Video Review Agent node.

    - If video passed review -> audio_review
    - If video failed and retries remain -> simulation_agent (re-simulate)
    - If max review attempts exceeded -> audio_review (proceed anyway with warning)
    - If reviewer errored (result is None) -> audio_review (don't block pipeline)
    """
    review_data = state.get("video_review_result")
    attempts = state.get("video_review_attempts", 0)

    # If reviewer errored (no result), proceed to audio review
    if review_data is None:
        logger.warning(
            "[route_after_video_review] Video review produced no result (attempt %d) "
            "— proceeding to audio review without video quality check",
            attempts,
        )
        return "audio_review"

    passed = review_data.get("passed", True) if isinstance(review_data, dict) else True

    if passed:
        logger.info("[route_after_video_review] Video passed review — proceeding to audio review")
        return "audio_review"

    # If every issue is reviewer_error (e.g. model 404, model not pulled),
    # re-simulating won't fix the reviewer — proceed to audio review.
    issues = review_data.get("issues", []) if isinstance(review_data, dict) else []
    if issues and all(
        (i.get("category") if isinstance(i, dict) else getattr(i, "category", None)) == "reviewer_error"
        for i in issues
    ):
        logger.error(
            "[route_after_video_review] Video review failed due to reviewer infrastructure "
            "error (e.g. model unavailable) after %d attempt(s) — skipping re-simulation. "
            "Issues: %s",
            attempts,
            [i.get("description", str(i)) if isinstance(i, dict) else str(i) for i in issues],
        )
        return "audio_review"

    # Failed — try re-simulation if attempts remain
    if attempts < MAX_VIDEO_REVIEW_ATTEMPTS:
        logger.info("Video review failed (attempt %d/%d) — routing back to simulation_agent",
                     attempts, MAX_VIDEO_REVIEW_ATTEMPTS)
        return "simulation_agent"

    # Max attempts — proceed to audio review with accumulated warnings
    logger.warning("Video review failed after %d attempts — proceeding to audio review", attempts)
    return "audio_review"


def route_after_audio_review(state: dict[str, Any]) -> str:
    """Route after Audio Review Agent node.

    - If audio passed review -> human_review
    - If audio failed and retries remain -> video_editor_agent (re-edit)
    - If max review attempts exceeded -> human_review (let human decide)
    - If reviewer errored (result is None) -> human_review (don't block pipeline)
    """
    review_data = state.get("audio_review_result")
    attempts = state.get("audio_review_attempts", 0)

    # If reviewer errored (no result), proceed to human review
    if review_data is None:
        logger.warning(
            "[route_after_audio_review] Audio review produced no result (attempt %d) "
            "— proceeding to human review without audio quality check",
            attempts,
        )
        return "human_review"

    passed = review_data.get("passed", True) if isinstance(review_data, dict) else True

    if passed:
        return "human_review"

    # Failed — try re-edit if attempts remain
    if attempts < MAX_AUDIO_REVIEW_ATTEMPTS:
        logger.info("Audio review failed (attempt %d/%d) — routing back to video_editor_agent",
                     attempts, MAX_AUDIO_REVIEW_ATTEMPTS)
        return "video_editor_agent"

    # Max attempts — send to human review
    logger.warning("Audio review failed after %d attempts — proceeding to human review", attempts)
    return "human_review"


# =============================================================================
# Graph Construction
# =============================================================================


def build_pipeline_graph() -> StateGraph:
    """Build the LangGraph state machine for the content pipeline.

    Graph structure:
        START -> idea_agent -> simulation_agent -> video_editor_agent
              -> video_review -> audio_review -> human_review
              -> publish_queue -> END

    With conditional edges for retry, escalation, and review routing.
    """
    graph = StateGraph(PipelineGraphState)

    # Add nodes
    graph.add_node("idea_agent", idea_node)
    graph.add_node("simulation_agent", simulation_node)
    graph.add_node("video_editor_agent", video_editor_node)
    graph.add_node("video_review", video_review_node)
    graph.add_node("audio_review", audio_review_node)
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

    graph.add_conditional_edges(
        "video_editor_agent",
        route_after_video_editor,
        {
            "video_review": "video_review",
            "__end__": END,
        },
    )

    graph.add_conditional_edges(
        "video_review",
        route_after_video_review,
        {
            "audio_review": "audio_review",
            "simulation_agent": "simulation_agent",
        },
    )

    graph.add_conditional_edges(
        "audio_review",
        route_after_audio_review,
        {
            "human_review": "human_review",
            "video_editor_agent": "video_editor_agent",
        },
    )

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
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline end-to-end.

    Args:
        pipeline_name: Pipeline to run (default: 'physics').
        checkpointer: Optional checkpointer for state persistence.
        thread_id: Optional thread ID for checkpoint keying.
        run_id: Optional run ID. If set, reuses an existing run's cache
                so completed steps are skipped (zero LLM/sandbox cost).

    Returns:
        Final pipeline state dict.
    """
    compiled = compile_pipeline(checkpointer=checkpointer)

    pipeline_run_id = run_id or str(uuid4())

    # Initialise RunTracer with sinks for this run
    from kairos.ai.tracing.sinks.langfuse_sink import LangfuseSink
    from kairos.ai.tracing.sinks.db_sink import DatabaseSink

    tracer = init_tracer()
    tracer.add_sink(LangfuseSink())
    tracer.add_sink(DatabaseSink())
    tracer.init_run(pipeline_run_id, pipeline_name)

    # Initialise response cache — automatically reuses LLM/sandbox outputs
    run_cache = init_cache(pipeline_run_id)

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
        "video_review_result": None,
        "audio_review_result": None,
        "video_review_attempts": 0,
        "audio_review_attempts": 0,
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

    # Finalise run tracing
    concept_data = final_state.get("concept")
    concept_title = None
    if isinstance(concept_data, dict):
        concept_title = concept_data.get("title")
    tracer.complete_run(
        status,
        errors=final_state.get("errors", []),
        final_video_path=final_state.get("final_video_path"),
        concept_title=concept_title,
    )

    # --- P3.29: Post-run cost alert check ---
    try:
        from kairos.ai.tracing.sinks.langfuse_sink import AlertManager

        total_cost = final_state.get("total_cost_usd", 0.0)
        alert_mgr = AlertManager()
        alert_mgr.check_run_cost(pipeline_run_id, total_cost)
    except Exception as exc:
        logger.debug("Cost alert check skipped: %s", exc)

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
    from kairos.schemas.contracts import (
        CaptionSet,
        ConceptBrief,
        MusicTrackMetadata,
        SimulationResult,
        SimulationStats,
        ValidationResult,
        VideoReviewResult,
        AudioReviewResult,
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
        video_review_attempts=state.get("video_review_attempts", 0),
        audio_review_attempts=state.get("audio_review_attempts", 0),
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

    if state.get("video_review_result") and isinstance(state["video_review_result"], dict):
        ps.video_review_result = VideoReviewResult(**state["video_review_result"])

    if state.get("audio_review_result") and isinstance(state["audio_review_result"], dict):
        ps.audio_review_result = AudioReviewResult(**state["audio_review_result"])

    return ps
