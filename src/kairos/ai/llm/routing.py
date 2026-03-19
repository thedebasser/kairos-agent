"""Kairos Agent — LLM Routing Service.

Unified LLM access via LiteLLM + Instructor for structured output.
Includes quality-based fallback (local → cloud) with learning loop.
Langfuse tracing is wired in for every LLM call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

import asyncio
import functools

import instructor
import litellm
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from kairos.ai.llm.capabilities import get_capabilities, ModelType
from kairos.ai.tracing.sinks.langfuse_sink import trace_llm_call

# Ensure .env is loaded (API keys etc.)
load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Thinking content accumulator (Finding 1.3)
# ---------------------------------------------------------------------------
# Graph nodes can call ``collect_thinking()`` after a step to retrieve all
# thinking texts captured during that step's LLM calls.

_thinking_buffer: list[dict[str, str]] = []


def collect_thinking() -> list[dict[str, str]]:
    """Return and clear all thinking entries accumulated since last collection.

    Each entry is ``{"model": ..., "thinking": ...}``.
    """
    global _thinking_buffer  # noqa: PLW0603
    entries = list(_thinking_buffer)
    _thinking_buffer.clear()
    return entries


def _accumulate_thinking(model: str, thinking: str | None) -> None:
    """Stash thinking content for later collection by graph nodes."""
    if thinking:
        _thinking_buffer.append({"model": model, "thinking": thinking})


# ---------------------------------------------------------------------------
# LLM Call Record Buffer (§3 — receipt box pattern)
# ---------------------------------------------------------------------------
# Every LLM call drops a structured record into this buffer.  Graph nodes
# call ``collect_llm_calls()`` at step end to grab all records and pass
# them to ``save_step(llm_calls=...)``.

_llm_call_buffer: list[dict[str, Any]] = []


def collect_llm_calls() -> list[dict[str, Any]]:
    """Return and clear all LLM call records accumulated since last collection.

    Each entry follows the §3 schema from the logging spec.
    """
    global _llm_call_buffer  # noqa: PLW0603
    entries = list(_llm_call_buffer)
    _llm_call_buffer.clear()
    return entries


def _record_llm_call(
    *,
    model_alias: str,
    model_resolved: str,
    call_pattern: str,
    routing_outcome: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: int,
    status: str,
    error: str | None = None,
    thinking_summary: str | None = None,
    model_type: str = "local",
    provider: str = "",
    raw_prompt: Any | None = None,
    raw_response: Any | None = None,
    raw_thinking: str | None = None,
) -> None:
    """Accumulate a structured LLM call record for later collection.

    Args:
        raw_prompt: Full prompt messages (written to llm_calls/ prompt file).
        raw_response: Full response content (written to llm_calls/ response file).
        raw_thinking: Full thinking/reasoning text (written to llm_calls/ thinking file).
        thinking_summary: Truncated preview for the step JSON artifact.
    """
    from uuid import uuid4

    record: dict[str, Any] = {
        "call_id": str(uuid4()),
        "model_alias": model_alias,
        "model_resolved": model_resolved,
        "model_type": model_type,
        "provider": provider,
        "call_pattern": call_pattern,
        "routing_outcome": routing_outcome,
        "tokens": {
            "prompt": tokens_in,
            "completion": tokens_out,
            "thinking": None,  # Populated when thinking token counts available
            "total": tokens_in + tokens_out,
        },
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "status": status,
        "error": error,
        "thinking_summary": (
            thinking_summary[:200] if thinking_summary else None
        ),
    }
    # Attach raw data for file persistence (popped by _persist_llm_call_files)
    if raw_prompt is not None:
        record["_raw_prompt"] = raw_prompt
    if raw_response is not None:
        record["_raw_response"] = raw_response
    if raw_thinking is not None:
        record["_raw_thinking"] = raw_thinking
    _llm_call_buffer.append(record)


# ---------------------------------------------------------------------------
# Extended Thinking helpers (delegated to ModelCapabilities)
# ---------------------------------------------------------------------------

def _supports_thinking(resolved_model: str) -> bool:
    """Return True if the resolved model supports thinking/reasoning tokens."""
    return get_capabilities(resolved_model).supports_thinking


def _get_thinking_param(resolved_model: str) -> dict | None:
    """Return the provider-specific thinking kwarg, or None."""
    return get_capabilities(resolved_model).get_thinking_param()


def _extract_thinking(response: Any, resolved_model: str = "") -> str | None:
    """Extract thinking/reasoning content via the model capabilities layer.

    Falls back to Anthropic-style extraction if no resolved_model is given
    (backward compat).
    """
    caps = get_capabilities(resolved_model) if resolved_model else None
    if caps:
        return caps.extract_thinking(response)
    # Legacy fallback
    msg = response.choices[0].message
    if getattr(msg, "reasoning_content", None):
        return msg.reasoning_content
    blocks = getattr(msg, "thinking_blocks", None)
    if blocks:
        parts = [b.thinking for b in blocks if hasattr(b, "thinking")]
        if parts:
            return "\n".join(parts)
    return None


def _log_thinking(model: str, thinking: str | None, label: str = "") -> None:
    """Log extended thinking content if present."""
    if not thinking:
        return
    tag = f"[THINKING:{label}] " if label else "[THINKING] "
    # Show a truncated preview at INFO, full content at DEBUG
    preview = thinking[:1000]
    if len(thinking) > 1000:
        preview += f"\n... ({len(thinking)} chars total)"
    logger.info("%s%s thinking (%d chars):\n%s", tag, model, len(thinking), preview)
    logger.debug("%sFull thinking:\n%s", tag, thinking)


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


# ---------------------------------------------------------------------------
# Context window budget estimation (Finding 3.4)
# ---------------------------------------------------------------------------

def _check_context_budget(
    resolved_model: str,
    messages: list[dict[str, str]],
    max_output_tokens: int,
) -> dict[str, int]:
    """Estimate input token count and warn if near context window limit.

    Returns a dict with ``input_tokens``, ``context_window``,
    ``available_for_output``, and ``utilisation_pct``.
    Does *not* raise on overflow — just logs a WARNING so the call can
    still proceed (the model will do its own truncation or error).
    """
    try:
        from litellm import get_max_tokens, token_counter
        input_tokens = token_counter(model=resolved_model, messages=messages)
        context_window = get_max_tokens(resolved_model) or 200_000
    except Exception:
        # token_counter may fail for unknown models — don't block the call
        return {"input_tokens": 0, "context_window": 0, "available_for_output": 0, "utilisation_pct": 0}

    available = context_window - input_tokens
    utilisation_pct = int(input_tokens / context_window * 100) if context_window else 0

    budget = {
        "input_tokens": input_tokens,
        "context_window": context_window,
        "available_for_output": available,
        "utilisation_pct": utilisation_pct,
    }

    if available < max_output_tokens:
        logger.warning(
            "[LLM] Context budget overflow: input=%d + max_output=%d = %d > context_window=%d (%d%% used). "
            "Response may be truncated.",
            input_tokens, max_output_tokens, input_tokens + max_output_tokens,
            context_window, utilisation_pct,
        )
    elif utilisation_pct > 80:
        logger.info(
            "[LLM] Context budget high: input=%d tokens (%d%% of %d window)",
            input_tokens, utilisation_pct, context_window,
        )
    else:
        logger.debug(
            "[LLM] Context budget OK: input=%d tokens (%d%% of %d window)",
            input_tokens, utilisation_pct, context_window,
        )

    return budget


# ---------------------------------------------------------------------------
# Token Usage / Cost Extraction  (Finding 1.1, 6.1)
# ---------------------------------------------------------------------------

def _extract_usage(response: Any, resolved_model: str) -> tuple[int, int, float]:
    """Extract token counts and compute cost from a LiteLLM response.

    Uses the model capabilities layer for pricing.  Cloud models get
    real costs; local models report $0.00.

    Returns (tokens_in, tokens_out, cost_usd).
    """
    tokens_in = 0
    tokens_out = 0
    cost_usd = 0.0

    usage = getattr(response, "usage", None)
    if usage:
        tokens_in = getattr(usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(usage, "completion_tokens", 0) or 0

    caps = get_capabilities(resolved_model)

    if caps.model_type == ModelType.CLOUD:
        # Try litellm.completion_cost() first — it knows cloud provider pricing.
        try:
            cost_usd = litellm.completion_cost(completion_response=response) or 0.0
        except Exception:
            # Fall back to capabilities pricing
            cost_usd = caps.compute_cost(tokens_in, tokens_out, resolved_model)
    else:
        # Local models are always free
        cost_usd = 0.0

    return tokens_in, tokens_out, cost_usd


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


def _load_model_api_bases() -> dict[str, str]:
    """Build alias → api_base map from litellm_config.yaml."""
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "litellm_config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bases: dict[str, str] = {}
        for entry in cfg.get("model_list", []):
            alias = entry.get("model_name")
            api_base = (entry.get("litellm_params") or {}).get("api_base")
            if alias and api_base:
                bases[alias] = api_base
        return bases
    except Exception:
        return {}


_model_alias_map = _load_model_alias_map()
_model_api_bases = _load_model_api_bases()
litellm.model_alias_map = _model_alias_map
logger.info("Loaded %d LiteLLM model aliases", len(_model_alias_map))


def _resolve_model(model: str) -> str:
    """Resolve an alias like 'concept-developer' to the real provider model.

    Falls back to the original string if no alias is found.
    """
    return _model_alias_map.get(model, model)


# ---------------------------------------------------------------------------
# Direct Ollama API call (bypasses litellm)
# ---------------------------------------------------------------------------
# litellm silently drops ``content`` for Ollama thinking models (qwen3-vl,
# qwen3.5) when the response includes a ``reasoning`` field.  For review
# agents where we want BOTH the thinking (reasoning) AND the response
# (JSON command), we call Ollama's OpenAI-compat endpoint directly.
# ---------------------------------------------------------------------------

@dataclass
class OllamaDirectResponse:
    """Structured result from a direct Ollama API call.

    Separates *thinking* (the model's reasoning) from *content* (the
    actionable response / JSON command).  Both are always logged.
    """
    content: str
    thinking: str | None
    tokens_in: int
    tokens_out: int
    model: str


def call_ollama_direct(
    model_alias: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 2048,
    timeout: int = 120,
    api_base: str | None = None,
    json_mode: bool = False,
) -> OllamaDirectResponse:
    """Call an Ollama model directly, preserving both thinking and content.

    Bypasses litellm which silently drops ``content`` for thinking models.
    Uses Ollama's ``/v1/chat/completions`` (OpenAI-compat) endpoint so
    message format (including ``image_url`` content parts) is identical
    to what litellm would send.

    When *json_mode* is True the function:
      1. Prepends ``/nothink`` to the system prompt so the model puts
         real content into the ``content`` field instead of just ``{}``.
      2. Adds ``response_format: {"type": "json_object"}`` to force
         structurally valid JSON output from Ollama.

    The model's reasoning is still captured via the ``reasoning`` field
    in the Ollama response even with ``/nothink``.

    Args:
        model_alias: LiteLLM alias (e.g. 'video-reviewer-default') OR
                     raw Ollama model (e.g. 'ollama/qwen3-vl:8b').
        messages: Chat messages in OpenAI format.
        max_tokens: Maximum completion tokens.
        timeout: Request timeout in seconds.
        api_base: Override Ollama base URL; defaults to the value from
                  litellm_config.yaml, then ``http://localhost:11434``.
        json_mode: When True, enforce JSON output via ``response_format``
                   and ``/nothink``.  Use for review agents that need
                   structured JSON responses.

    Returns:
        ``OllamaDirectResponse`` with separated thinking and content.
    """
    import copy
    import requests as _requests

    # Resolve alias → real Ollama model name
    resolved = _resolve_model(model_alias)
    # Strip 'ollama/' prefix that litellm uses
    ollama_model = resolved.removeprefix("ollama/")

    # Resolve API base
    if api_base is None:
        api_base = _model_api_bases.get(model_alias, "http://localhost:11434")
    url = f"{api_base.rstrip('/')}/v1/chat/completions"

    # Deep copy messages so we don't mutate the caller's list
    msgs = copy.deepcopy(messages)

    # ── JSON mode: prepend /nothink to system prompt ──────────────
    # Ollama thinking models (qwen3-vl, qwen3.5) dump everything into
    # the reasoning field and return empty/minimal content.  /nothink
    # tells the model to put the real answer into content while Ollama
    # still populates the reasoning field for our logs.
    if json_mode:
        for m in msgs:
            if m.get("role") == "system":
                existing = m.get("content", "")
                if isinstance(existing, str) and not existing.startswith("/nothink"):
                    m["content"] = "/nothink\n" + existing
                break

    payload: dict[str, Any] = {
        "model": ollama_model,
        "messages": msgs,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    logger.info(
        "[ollama_direct] Calling %s (json_mode=%s, timeout=%ds, max_tokens=%d)",
        ollama_model, json_mode, timeout, max_tokens,
    )

    resp = _requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    thinking = msg.get("reasoning") or msg.get("reasoning_content") or None

    usage = data.get("usage", {})
    tokens_in = usage.get("prompt_tokens", 0) or 0
    tokens_out = usage.get("completion_tokens", 0) or 0

    logger.info(
        "[ollama_direct] %s responded: %d content chars, %s thinking, %d/%d tokens",
        ollama_model,
        len(content),
        f"{len(thinking)} chars" if thinking else "none",
        tokens_in,
        tokens_out,
    )

    return OllamaDirectResponse(
        content=content,
        thinking=thinking,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        model=ollama_model,
    )


def get_instructor_client(*, mode: instructor.Mode | None = None) -> instructor.Instructor:
    """Get an Instructor client wrapping LiteLLM for structured output.

    Args:
        mode: Override the Instructor parsing mode.  Defaults to TOOLS.
              Use ``instructor.Mode.ANTHROPIC_REASONING_TOOLS`` when extended
              thinking is enabled on Anthropic models.
    """
    kwargs: dict[str, Any] = {}
    if mode is not None:
        kwargs["mode"] = mode
    return instructor.from_litellm(litellm.completion, **kwargs)


async def call_llm_code(
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    max_tokens: int = 16384,
    cache_step: str | None = None,
) -> T:
    """Call an LLM for code-heavy responses, bypassing Instructor tool_use.

    Instructor's TOOLS mode forces the model to wrap output in a function call,
    which catastrophically slows down large code responses (300s+ timeout).
    This function calls the LLM directly and manually parses the response
    into the Pydantic model.

    Args:
        model: LiteLLM model name.
        messages: Chat messages.
        response_model: Pydantic model (must have a ``code`` field).
        max_tokens: Maximum output tokens.
        cache_step: If set, cache the response under this step name.
                    On retry the cached response is returned without an LLM call.

    Returns:
        Parsed Pydantic model instance.
    """
    import json as _json
    import re as _re

    # ── Cache check ─────────────────────────────────────────────────
    from kairos.ai.llm.cache import get_cache
    cache = get_cache()
    if cache and cache_step:
        cached = cache.get_llm(cache_step, model, messages)
        if cached:
            resp_data = cached.get("response", cached)
            fields = {k: v for k, v in resp_data.items() if k in response_model.model_fields}
            return response_model(**fields)

    loop = asyncio.get_running_loop()
    timeout_sec = 300

    resolved_model = _resolve_model(model)
    logger.info("[LLM] Calling %s (resolved: %s) | code-mode | response_model=%s", model, resolved_model, response_model.__name__)
    logger.debug("[LLM] Full prompt (%d messages):\n%s", len(messages), _format_messages(messages))

    start = time.monotonic()
    status = "success"
    error_msg: str | None = None
    result: T | None = None
    thinking_text: str | None = None
    tokens_in = tokens_out = 0
    cost_usd = 0.0

    # Resolve capabilities for this model
    caps = get_capabilities(resolved_model)
    thinking_param = caps.get_thinking_param()
    if thinking_param:
        logger.info("[LLM] Extended thinking enabled for %s (budget=%d tokens)", model, thinking_param["budget_tokens"])

    # Pre-flight context window budget check (Finding 3.4)
    _check_context_budget(resolved_model, messages, max_tokens)

    try:
        call_kwargs: dict[str, Any] = dict(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=timeout_sec,
        )
        if thinking_param:
            call_kwargs["thinking"] = thinking_param

        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                functools.partial(litellm.completion, **call_kwargs),
            ),
            timeout=timeout_sec + 30,
        )

        # Extract token usage and cost (Finding 1.1)
        tokens_in, tokens_out, cost_usd = _extract_usage(response, resolved_model)

        # Extract and log thinking content via capabilities
        thinking_text = caps.extract_thinking(response)
        _log_thinking(model, thinking_text, label="code-mode")
        _accumulate_thinking(model, thinking_text)

        content = response.choices[0].message.content or ""

        # Strip think tags from content if present (Qwen3 models)
        if thinking_text and hasattr(caps, 'strip_think_tags'):
            content = caps.strip_think_tags(content)

        # Try to extract structured fields from the response
        code = ""
        reasoning = ""
        parsed: dict[str, Any] | None = None

        # Strategy 1: try to parse as JSON (if model returned JSON)
        try:
            parsed = _json.loads(content)
            if isinstance(parsed, dict):
                code = parsed.get("code", "")
                reasoning = parsed.get("reasoning", "")
        except (ValueError, _json.JSONDecodeError):
            parsed = None

        # Strategy 2: extract code block from markdown
        if not code:
            code_match = _re.search(r"```python\s*\n(.*?)```", content, _re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                # Finding 3.1: Explicit failure instead of treating entire response as code.
                # Strategy 3 previously assigned the full (potentially natural-language)
                # response to `code`, causing opaque sandbox syntax errors downstream.
                from kairos.exceptions import ConceptGenerationError
                raise ConceptGenerationError(
                    f"LLM response did not contain an extractable code block. "
                    f"Response preview ({len(content)} chars): {content[:500]}"
                )

        # Build the response model
        fields: dict[str, Any] = {"code": code}
        # Add reasoning if the model supports it
        if hasattr(response_model, "model_fields") and "reasoning" in response_model.model_fields:
            fields["reasoning"] = reasoning or "Generated via code-mode (no structured reasoning)"
        # Add changes_made if the model supports it (for AdjustedSimulationCode)
        if hasattr(response_model, "model_fields") and "changes_made" in response_model.model_fields:
            changes: list[str] = []
            if parsed is not None:
                changes = parsed.get("changes_made", [])
            if not changes:
                changes = ["Code adjusted via code-mode"]
            fields["changes_made"] = changes

        result = response_model(**fields)

        # ── Cache store ─────────────────────────────────────────────
        if cache and cache_step and result is not None:
            cache.put_llm(cache_step, model, messages, result)

        return result

    except asyncio.TimeoutError:
        status = "error"
        error_msg = f"LLM call timed out after {timeout_sec}s"
        logger.error("[LLM] %s timed out after %ds", model, timeout_sec)
        raise TimeoutError(error_msg)
    except Exception as exc:
        status = "error"
        error_msg = str(exc)
        raise
    finally:
        latency = int((time.monotonic() - start) * 1000)
        try:
            if result is not None:
                result_str = result.model_dump_json(indent=2) if hasattr(result, 'model_dump_json') else str(result)
                logger.info("[LLM] %s responded in %dms (code-mode) | tokens=%d/%d cost=$%.4f | preview: %.300s",
                            model, latency, tokens_in, tokens_out, cost_usd, result_str[:300])
                logger.debug("[LLM] Full response:\n%s", result_str)
            else:
                logger.error("[LLM] %s FAILED after %dms (code-mode) | error: %s", model, latency, error_msg)
            trace_llm_call(
                trace_name=f"call_llm_code:{model}",
                model=model,
                input_messages=messages,
                output=result,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency,
                status=status,
                error=error_msg,
                thinking=thinking_text,
            )
            # Buffer call record for step artifact (§3)
            _record_llm_call(
                model_alias=model,
                model_resolved=resolved_model,
                call_pattern="direct",
                routing_outcome="cloud" if caps.model_type == ModelType.CLOUD else "local",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency,
                status=status,
                error=error_msg,
                thinking_summary=thinking_text,
                model_type=caps.model_type,
                provider=caps.family_name,
                raw_prompt=messages,
                raw_response=result.model_dump(mode="json") if result and hasattr(result, "model_dump") else None,
                raw_thinking=thinking_text,
            )
        except Exception as _trace_exc:
            logger.debug("Post-LLM tracing failed (non-fatal): %s", _trace_exc)


async def call_llm(
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    *,
    max_retries: int = 2,
    max_tokens: int = 16384,
    cache_step: str | None = None,
) -> T:
    """Call an LLM via LiteLLM + Instructor with structured output.

    Runs the synchronous Instructor client in a thread executor to avoid
    blocking the async event loop.

    Args:
        model: LiteLLM model name (e.g., 'concept-developer').
        messages: Chat messages.
        response_model: Pydantic model for structured output parsing.
        max_retries: Number of retries on malformed output.
        max_tokens: Maximum output tokens (default 16384 — needed for
                    Anthropic which defaults to 4096 if not set).
        cache_step: If set, cache the response under this step name.
                    On retry the cached response is returned without an LLM call.

    Returns:
        Parsed and validated Pydantic model instance.
    """
    # ── Cache check ─────────────────────────────────────────────────
    from kairos.ai.llm.cache import get_cache
    cache = get_cache()
    if cache and cache_step:
        cached = cache.get_llm(cache_step, model, messages)
        if cached:
            resp_data = cached.get("response", cached)
            fields = {k: v for k, v in resp_data.items() if k in response_model.model_fields}
            return response_model(**fields)

    loop = asyncio.get_running_loop()

    # Timeout: local Ollama models may take 2-3 min on first load;
    # cloud simulation code-gen also needs 5 min for long responses.
    is_local = model.startswith("ollama/") or "/" in _resolve_model(model)
    timeout_sec = 300  # 5 min for both local and cloud

    # Resolve alias (e.g. "concept-developer" → "claude-sonnet-4-6")
    resolved_model = _resolve_model(model)

    # Look up model capabilities
    caps = get_capabilities(resolved_model)
    thinking_param = caps.get_thinking_param()

    # Use the Instructor mode appropriate for this model family
    client_mode = caps.get_instructor_mode(thinking_enabled=bool(thinking_param))
    client = get_instructor_client(mode=client_mode)

    # Log what we're about to send
    prompt_preview = messages[-1]["content"][:200] if messages else "(empty)"
    logger.info("[LLM] Calling %s (resolved: %s) | response_model=%s", model, resolved_model, response_model.__name__)
    logger.debug("[LLM] Full prompt (%d messages):\n%s", len(messages), _format_messages(messages))

    start = time.monotonic()
    status = "success"
    error_msg: str | None = None
    result: T | None = None
    thinking_text: str | None = None
    tokens_in = tokens_out = 0
    cost_usd = 0.0

    if thinking_param:
        logger.info("[LLM] Extended thinking enabled for %s (budget=%d tokens)", model, thinking_param["budget_tokens"])

    # Pre-flight context window budget check (Finding 3.4)
    _check_context_budget(resolved_model, messages, max_tokens)

    try:
        # Extra call params (e.g. think=False for Ollama thinking models)
        extra_params = caps.get_extra_call_params(structured_output=True)

        call_kwargs: dict[str, Any] = dict(
            model=resolved_model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            timeout=timeout_sec,
            max_tokens=max_tokens,
            **extra_params,
        )
        if thinking_param:
            call_kwargs["thinking"] = thinking_param

        result, completion = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                functools.partial(
                    client.chat.completions.create_with_completion,
                    **call_kwargs,
                ),
            ),
            timeout=timeout_sec + 30,  # Grace period beyond LiteLLM timeout
        )  # type: ignore[assignment]

        # Extract token usage and cost (Finding 1.1)
        tokens_in, tokens_out, cost_usd = _extract_usage(completion, resolved_model)

        # Extract and log thinking content from raw completion
        thinking_text = caps.extract_thinking(completion)
        _log_thinking(model, thinking_text)
        _accumulate_thinking(model, thinking_text)

        # ── Cache store ─────────────────────────────────────────────
        if cache and cache_step and result is not None:
            cache.put_llm(cache_step, model, messages, result)

        return result  # type: ignore[return-value]
    except asyncio.TimeoutError:
        status = "error"
        error_msg = f"LLM call timed out after {timeout_sec}s"
        logger.error("[LLM] %s timed out after %ds", model, timeout_sec)
        raise TimeoutError(error_msg)
    except Exception as exc:
        status = "error"
        error_msg = str(exc)
        raise
    finally:
        latency = int((time.monotonic() - start) * 1000)
        try:
            if result is not None:
                result_str = result.model_dump_json(indent=2) if hasattr(result, 'model_dump_json') else str(result)
                logger.info("[LLM] %s responded in %dms | tokens=%d/%d cost=$%.4f | preview: %.300s",
                            model, latency, tokens_in, tokens_out, cost_usd, result_str)
                logger.debug("[LLM] Full response:\n%s", result_str)
            else:
                logger.error("[LLM] %s FAILED after %dms | error: %s", model, latency, error_msg)
            trace_llm_call(
                trace_name=f"call_llm:{model}",
                model=model,
                input_messages=messages,
                output=result,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency,
                status=status,
                error=error_msg,
                thinking=thinking_text,
            )
            # Buffer call record for step artifact (§3)
            _record_llm_call(
                model_alias=model,
                model_resolved=resolved_model,
                call_pattern="direct",
                routing_outcome="cloud" if caps.model_type == ModelType.CLOUD else "local",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency,
                status=status,
                error=error_msg,
                thinking_summary=thinking_text,
                model_type=caps.model_type,
                provider=caps.family_name,
                raw_prompt=messages,
                raw_response=result.model_dump(mode="json") if result and hasattr(result, "model_dump") else None,
                raw_thinking=thinking_text,
            )
        except Exception as _trace_exc:
            logger.debug("Post-LLM tracing failed (non-fatal): %s", _trace_exc)


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
    loop = asyncio.get_running_loop()

    # Resolve aliases to real provider model names
    resolved_primary = _resolve_model(primary_model)
    resolved_fallback = _resolve_model(fallback_model)

    # Look up capabilities for both models
    primary_caps = get_capabilities(resolved_primary)
    fallback_caps = get_capabilities(resolved_fallback)

    # Use the correct Instructor mode for the primary (local) model
    # (e.g. Ollama models need JSON mode, not TOOLS)
    primary_client_mode = primary_caps.get_instructor_mode(thinking_enabled=False)
    client = get_instructor_client(mode=primary_client_mode) if primary_client_mode else get_instructor_client()

    # Extra call params for the primary model (e.g. think=False for Ollama thinking models)
    primary_extra_params = primary_caps.get_extra_call_params(structured_output=True)

    # Build thinking for the cloud/fallback model via capabilities
    thinking_param = fallback_caps.get_thinking_param()

    # Use the Instructor mode appropriate for the fallback model
    cloud_client_mode = fallback_caps.get_instructor_mode(thinking_enabled=bool(thinking_param))
    cloud_client = get_instructor_client(mode=cloud_client_mode) if cloud_client_mode else client

    # Fast path: when local is disabled (primary == fallback), skip the
    # redundant first attempt and go straight to the single cloud call.
    skip_primary = resolved_primary == resolved_fallback

    if skip_primary:
        logger.info("[LLM] quality_fallback: local disabled, going direct to %s (resolved: %s)", fallback_model, resolved_fallback)
    else:
        logger.info("[LLM] quality_fallback: trying %s (resolved: %s), fallback %s", primary_model, resolved_primary, fallback_model)
    logger.debug("[LLM] Full prompt (%d messages):\n%s", len(messages), _format_messages(messages))

    # Try primary model (unless we're skipping it)
    # Local models may be slow on first inference (model loading)
    local_timeout_sec = 300
    if not skip_primary:
        try:
            start = time.monotonic()
            local_call_kwargs: dict[str, Any] = dict(
                model=resolved_primary,
                messages=messages,
                response_model=response_model,
                max_retries=1,
                timeout=local_timeout_sec,
                max_tokens=16384,
                **primary_extra_params,
            )
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(
                        client.chat.completions.create,
                        **local_call_kwargs,
                    ),
                ),
                timeout=local_timeout_sec + 30,
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
                _record_llm_call(
                    model_alias=primary_model,
                    model_resolved=resolved_primary,
                    call_pattern="quality_fallback",
                    routing_outcome="local",
                    tokens_in=0, tokens_out=0, cost_usd=0.0,
                    latency_ms=latency,
                    status="success",
                    model_type=primary_caps.model_type,
                    provider=primary_caps.family_name,
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
            _record_llm_call(
                model_alias=primary_model,
                model_resolved=resolved_primary,
                call_pattern="quality_fallback",
                routing_outcome="local",
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                latency_ms=latency,
                status="quality_failed",
                model_type=primary_caps.model_type,
                provider=primary_caps.family_name,
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
            _record_llm_call(
                model_alias=primary_model,
                model_resolved=resolved_primary,
                call_pattern="quality_fallback",
                routing_outcome="local",
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                latency_ms=0,
                status="error",
                error=f"Exception, falling back to {fallback_model}",
                model_type=primary_caps.model_type,
                provider=primary_caps.family_name,
            )

    # Cloud fallback (or direct cloud when local is disabled)
    if thinking_param:
        logger.info("[LLM] Extended thinking enabled for %s fallback (budget=%d tokens)", fallback_model, thinking_param["budget_tokens"])

    # Pre-flight context window budget check (Finding 3.4)
    _check_context_budget(resolved_fallback, messages, 16384)

    cloud_timeout_sec = 300  # Match local timeout (Finding 4.3)
    cloud_kwargs: dict[str, Any] = dict(
        model=resolved_fallback,
        messages=messages,
        response_model=response_model,
        max_retries=2,
        max_tokens=16384,
    )
    if thinking_param:
        cloud_kwargs["thinking"] = thinking_param

    start = time.monotonic()
    cloud_result, cloud_completion = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            functools.partial(
                cloud_client.chat.completions.create_with_completion,
                **cloud_kwargs,
            ),
        ),
        timeout=cloud_timeout_sec + 30,  # Grace period matching local path
    )  # type: ignore[assignment]

    # Extract and log thinking from cloud response via capabilities
    cloud_thinking = fallback_caps.extract_thinking(cloud_completion)
    _log_thinking(fallback_model, cloud_thinking, label="quality_fallback")
    _accumulate_thinking(fallback_model, cloud_thinking)

    latency = int((time.monotonic() - start) * 1000)
    # Extract token usage and cost
    fb_tokens_in, fb_tokens_out, fb_cost = _extract_usage(cloud_completion, resolved_fallback)
    label = "Cloud direct" if skip_primary else "Cloud fallback"
    result_str = cloud_result.model_dump_json(indent=2) if hasattr(cloud_result, 'model_dump_json') else str(cloud_result)
    logger.info("[LLM] %s %s responded in %dms | preview: %.300s", label, fallback_model, latency, result_str)
    logger.debug("[LLM] Full response:\n%s", result_str)
    trace_llm_call(
        trace_name=f"quality_fallback:{fallback_model}",
        model=fallback_model,
        input_messages=messages,
        output=cloud_result,
        tokens_in=fb_tokens_in,
        tokens_out=fb_tokens_out,
        cost_usd=fb_cost,
        latency_ms=latency,
        status="success",
        thinking=cloud_thinking,
    )
    _record_llm_call(
        model_alias=fallback_model,
        model_resolved=resolved_fallback,
        call_pattern="quality_fallback",
        routing_outcome="cloud" if skip_primary else "local_then_cloud",
        tokens_in=fb_tokens_in,
        tokens_out=fb_tokens_out,
        cost_usd=fb_cost,
        latency_ms=latency,
        status="success",
        thinking_summary=cloud_thinking,
        model_type=fallback_caps.model_type,
        provider=fallback_caps.family_name,
        raw_prompt=messages,
        raw_response=cloud_result.model_dump(mode="json") if hasattr(cloud_result, "model_dump") else None,
        raw_thinking=cloud_thinking,
    )

    # Learning loop: always record cloud output for future local model improvement.
    # When local is disabled (skip_primary), we still capture data so it can be
    # used for fine-tuning once the user has a GPU machine.
    from kairos.ai.llm.config import always_store_training_data as _always_store

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
        from datetime import datetime, timezone

        learnings_dir = Path(__file__).resolve().parent.parent.parent.parent / "knowledge" / "cloud_learnings"
        learnings_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{primary_model.replace('/', '_')}_{timestamp}.json"
        learning = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
