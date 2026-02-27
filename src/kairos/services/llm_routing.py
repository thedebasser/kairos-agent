"""Kairos Agent — LLM Routing Service.

Unified LLM access via LiteLLM + Instructor for structured output.
Includes quality-based fallback (local → cloud) with learning loop.
Langfuse tracing is wired in for every LLM call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

import asyncio
import functools

import instructor
import litellm
import yaml
from pydantic import BaseModel

from kairos.services.monitoring import trace_llm_call

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _format_messages(messages: list[dict[str, str]]) -> str:
    """Format messages for debug logging (truncate long content)."""
    lines = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if len(content) > 500:
            content = content[:500] + f"... ({len(content)} chars total)"
        lines.append(f"  [{role}] {content}")
    return "\n".join(lines)


# Suppress LiteLLM debug noise
litellm.suppress_debug_info = True
litellm.set_verbose = False

# ---------------------------------------------------------------------------
# Populate litellm.model_alias_map from litellm_config.yaml so that alias
# names (e.g. "concept-developer") resolve to real provider models
# (e.g. "claude-sonnet-4-6") without needing a running LiteLLM proxy server.
# ---------------------------------------------------------------------------

def _load_model_alias_map() -> dict[str, str]:
    """Build alias → real-model map from litellm_config.yaml."""
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "litellm_config.yaml"
    if not config_path.exists():
        logger.warning("litellm_config.yaml not found at %s — no aliases loaded", config_path)
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        alias_map: dict[str, str] = {}
        for entry in cfg.get("model_list", []):
            alias = entry.get("model_name")
            real_model = (entry.get("litellm_params") or {}).get("model")
            if alias and real_model and alias != real_model:
                alias_map[alias] = real_model
        return alias_map
    except Exception:
        logger.exception("Failed to load litellm_config.yaml")
        return {}


_model_alias_map = _load_model_alias_map()
litellm.model_alias_map = _model_alias_map
logger.info("Loaded %d LiteLLM model aliases", len(_model_alias_map))


def _resolve_model(model: str) -> str:
    """Resolve an alias like 'concept-developer' to the real provider model.

    Falls back to the original string if no alias is found.
    """
    return _model_alias_map.get(model, model)


def get_instructor_client() -> instructor.Instructor:
    """Get an Instructor client wrapping LiteLLM for structured output."""
    return instructor.from_litellm(litellm.completion)


async def call_llm(
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    max_retries: int = 2,
) -> T:
    """Call an LLM via LiteLLM + Instructor with structured output.

    Runs the synchronous Instructor client in a thread executor to avoid
    blocking the async event loop.

    Args:
        model: LiteLLM model name (e.g., 'concept-developer').
        messages: Chat messages.
        response_model: Pydantic model for structured output parsing.
        max_retries: Number of retries on malformed output.

    Returns:
        Parsed and validated Pydantic model instance.
    """
    client = get_instructor_client()
    loop = asyncio.get_running_loop()

    # Resolve alias (e.g. "concept-developer" → "claude-sonnet-4-6")
    resolved_model = _resolve_model(model)

    # Log what we're about to send
    prompt_preview = messages[-1]["content"][:200] if messages else "(empty)"
    logger.info("[LLM] Calling %s (resolved: %s) | response_model=%s", model, resolved_model, response_model.__name__)
    logger.debug("[LLM] Full prompt (%d messages):\n%s", len(messages), _format_messages(messages))

    start = time.monotonic()
    status = "success"
    error_msg: str | None = None
    result: T | None = None

    try:
        result = await loop.run_in_executor(
            None,
            functools.partial(
                client.chat.completions.create,
                model=resolved_model,
                messages=messages,
                response_model=response_model,
                max_retries=max_retries,
            ),
        )  # type: ignore[assignment]
        return result  # type: ignore[return-value]
    except Exception as exc:
        status = "error"
        error_msg = str(exc)
        raise
    finally:
        latency = int((time.monotonic() - start) * 1000)
        if result is not None:
            result_str = result.model_dump_json(indent=2) if hasattr(result, 'model_dump_json') else str(result)
            logger.info("[LLM] %s responded in %dms | preview: %.300s", model, latency, result_str)
            logger.debug("[LLM] Full response:\n%s", result_str)
        else:
            logger.error("[LLM] %s FAILED after %dms | error: %s", model, latency, error_msg)
        trace_llm_call(
            trace_name=f"call_llm:{model}",
            model=model,
            input_messages=messages,
            output=result,
            latency_ms=latency,
            status=status,
            error=error_msg,
        )


async def call_with_quality_fallback(
    primary_model: str,
    fallback_model: str,
    messages: list[dict[str, str]],
    validator: Callable[[Any], bool],
    response_model: type[T],
    *,
    pipeline_run_id: UUID | None = None,
) -> T:
    """Try primary (local), validate output, fall back to cloud if invalid.

    Records successful cloud fallbacks as RAG learnings for local model improvement.
    When primary_model == fallback_model (i.e. local is disabled and both
    resolve to the cloud alias), skip the redundant first attempt and go
    straight to cloud — but still record training data.

    Args:
        primary_model: First model to try (typically local).
        fallback_model: Fallback model (typically cloud).
        messages: Chat messages.
        validator: Function that returns True if output is acceptable.
        response_model: Pydantic model for structured output parsing.
        pipeline_run_id: For tracing and learning loop.

    Returns:
        Validated response from either primary or fallback model.
    """
    client = get_instructor_client()

    loop = asyncio.get_running_loop()

    # Resolve aliases to real provider model names
    resolved_primary = _resolve_model(primary_model)
    resolved_fallback = _resolve_model(fallback_model)

    # Fast path: when local is disabled (primary == fallback), skip the
    # redundant first attempt and go straight to the single cloud call.
    skip_primary = resolved_primary == resolved_fallback

    if skip_primary:
        logger.info("[LLM] quality_fallback: local disabled, going direct to %s (resolved: %s)", fallback_model, resolved_fallback)
    else:
        logger.info("[LLM] quality_fallback: trying %s (resolved: %s), fallback %s", primary_model, resolved_primary, fallback_model)
    logger.debug("[LLM] Full prompt (%d messages):\n%s", len(messages), _format_messages(messages))

    # Try primary model (unless we're skipping it)
    if not skip_primary:
        try:
            start = time.monotonic()
            result = await loop.run_in_executor(
                None,
                functools.partial(
                    client.chat.completions.create,
                    model=resolved_primary,
                    messages=messages,
                    response_model=response_model,
                    max_retries=1,
                ),
            )  # type: ignore[assignment]
            latency = int((time.monotonic() - start) * 1000)
            logger.debug("%s responded in %dms", primary_model, latency)

            if validator(result):
                trace_llm_call(
                    trace_name=f"quality_fallback:{primary_model}",
                    model=primary_model,
                    input_messages=messages,
                    output=result,
                    latency_ms=latency,
                    status="success",
                )
                return result
            logger.warning("Quality check failed on %s, falling back to %s", primary_model, fallback_model)
            trace_llm_call(
                trace_name=f"quality_fallback:{primary_model}",
                model=primary_model,
                input_messages=messages,
                output=result,
                latency_ms=latency,
                status="quality_failed",
            )
        except Exception:
            logger.warning(
                "%s failed, falling back to %s",
                primary_model,
                fallback_model,
                exc_info=True,
            )
            trace_llm_call(
                trace_name=f"quality_fallback:{primary_model}",
                model=primary_model,
                input_messages=messages,
                output=None,
                status="error",
                error=f"Exception, falling back to {fallback_model}",
            )

    # Cloud fallback (or direct cloud when local is disabled)
    start = time.monotonic()
    cloud_result: T = await loop.run_in_executor(
        None,
        functools.partial(
            client.chat.completions.create,
            model=resolved_fallback,
            messages=messages,
            response_model=response_model,
            max_retries=2,
        ),
    )  # type: ignore[assignment]
    latency = int((time.monotonic() - start) * 1000)
    label = "Cloud direct" if skip_primary else "Cloud fallback"
    result_str = cloud_result.model_dump_json(indent=2) if hasattr(cloud_result, 'model_dump_json') else str(cloud_result)
    logger.info("[LLM] %s %s responded in %dms | preview: %.300s", label, fallback_model, latency, result_str)
    logger.debug("[LLM] Full response:\n%s", result_str)
    trace_llm_call(
        trace_name=f"quality_fallback:{fallback_model}",
        model=fallback_model,
        input_messages=messages,
        output=cloud_result,
        latency_ms=latency,
        status="success",
    )

    # Learning loop: always record cloud output for future local model improvement.
    # When local is disabled (skip_primary), we still capture data so it can be
    # used for fine-tuning once the user has a GPU machine.
    from kairos.services.llm_config import always_store_training_data as _always_store

    should_record = validator(cloud_result) and (not skip_primary or _always_store())
    if should_record:
        await _record_cloud_learning(
            pipeline_run_id=pipeline_run_id,
            primary_model=primary_model,
            fallback_model=fallback_model,
            messages=messages,
            successful_output=cloud_result,
        )

    return cloud_result


async def _record_cloud_learning(
    *,
    pipeline_run_id: UUID | None,
    primary_model: str,
    fallback_model: str,
    messages: list[dict[str, str]],
    successful_output: BaseModel,
) -> None:
    """Record a cloud fallback success for the RAG learning loop.

    Stores the input/output pair in:
    1. agent_runs table with status='escalated'
    2. Knowledge directory under cloud_learnings/ (for future ChromaDB ingestion)
    """
    logger.info(
        "Learning loop: %s failed, %s succeeded (run=%s). "
        "Recording for future local model improvement.",
        primary_model,
        fallback_model,
        pipeline_run_id,
    )

    # 1. Record in agent_runs table with status='escalated'
    try:
        from kairos.db.session import async_session_factory
        from kairos.db.operations import log_agent_run

        async with async_session_factory() as session:
            await log_agent_run(
                session,
                pipeline_run_id=pipeline_run_id,
                agent_name="llm_routing",
                step_name="quality_fallback",
                model_used=fallback_model,
                input_summary={"messages": messages, "primary_model": primary_model},
                output_summary=successful_output.model_dump() if hasattr(successful_output, "model_dump") else str(successful_output),
                status="escalated",
            )
            await session.commit()
    except Exception as e:
        logger.warning("Failed to record escalation in agent_runs: %s", e)

    # 2. Write learning to cloud_learnings directory for future RAG ingestion
    try:
        import json
        from pathlib import Path
        from datetime import datetime

        learnings_dir = Path(__file__).resolve().parent.parent.parent.parent / "knowledge" / "cloud_learnings"
        learnings_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{primary_model.replace('/', '_')}_{timestamp}.json"
        learning = {
            "timestamp": datetime.now().isoformat(),
            "primary_model": primary_model,
            "fallback_model": fallback_model,
            "pipeline_run_id": str(pipeline_run_id) if pipeline_run_id else None,
            "messages": messages,
            "successful_output": successful_output.model_dump() if hasattr(successful_output, "model_dump") else str(successful_output),
        }
        (learnings_dir / filename).write_text(json.dumps(learning, indent=2, default=str))
        logger.info("Saved cloud learning to %s", learnings_dir / filename)
    except Exception as e:
        logger.warning("Failed to save cloud learning to file: %s", e)
