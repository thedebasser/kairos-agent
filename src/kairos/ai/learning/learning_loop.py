"""Kairos Agent — Learning Loop Service.

Implements the capture → store → retrieve → inject cycle described in the
AI Architecture Review (§1–§6).  Every public function is safe to call even
when the database is unavailable — failures are logged and swallowed so the
pipeline never crashes due to the learning system.

Key design decisions:

* **Enable/disable flag** — ``Settings.learning_loop_enabled`` controls
  whether data is *recorded*.  Even when enabled, examples are NOT injected
  into prompts until the ``verified`` flag on the row is ``True``.
* **Verified gate** — ``get_few_shot_examples`` only returns rows where
  ``verified=True``.  An operator must explicitly promote examples.
* **Category knowledge** — accumulated per-category JSONB on
  ``category_stats.knowledge``.  Updated after every successful run
  (if the learning loop is enabled).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from kairos.config import get_settings
from kairos.schemas.contracts import (
    CategoryKnowledge,
    FailedCheck,
    PastFix,
    ValidationCheck,
    ValidationFeedback,
    ValidationResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Capture — store training example after a successful run
# ============================================================================


async def record_training_example(
    *,
    pipeline: str,
    category: str,
    concept_brief: dict[str, Any],
    simulation_code: str,
    validation_passed: bool,
    iteration_count: int = 1,
    reasoning: str = "",
    thinking_content: str = "",
    simulation_id: UUID | None = None,
) -> None:
    """Persist a training example after a successful simulation run.

    Respects ``learning_loop_enabled``.  The row is created with
    ``human_approved=True`` but ``verified=False`` — it will NOT be
    used in prompts until an operator sets verified=True.
    """
    settings = get_settings()
    if not settings.learning_loop_enabled:
        logger.debug("Learning loop disabled — skipping training example storage")
        return

    try:
        from kairos.db.session import async_session_factory
        from kairos.db.operations import create_training_example

        async with async_session_factory() as session:
            await create_training_example(
                session,
                simulation_id=simulation_id or uuid4(),
                pipeline=pipeline,
                category=category,
                concept_brief=concept_brief,
                simulation_code=simulation_code,
                validation_passed=validation_passed,
                human_approved=True,
                verified=False,  # <-- gate: not used until operator promotes
                iteration_count=iteration_count,
                reasoning=reasoning,
                thinking_content=thinking_content,
            )
            await session.commit()
        logger.info(
            "Stored training example (pipeline=%s, category=%s, iters=%d, verified=False)",
            pipeline,
            category,
            iteration_count,
        )
    except Exception as exc:
        logger.warning("Failed to store training example: %s", exc)


# ============================================================================
# Retrieve — get few-shot examples for prompt injection
# ============================================================================


async def get_few_shot_examples(
    pipeline: str,
    category: str | None = None,
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Return verified training examples formatted for prompt injection.

    Each dict has keys: ``title``, ``code``, ``reasoning``, ``iteration_count``.
    Returns an empty list when the DB is unreachable or no verified examples exist.
    """
    try:
        from kairos.db.session import async_session_factory
        from kairos.db.operations import get_verified_training_examples

        async with async_session_factory() as session:
            rows = await get_verified_training_examples(
                session,
                pipeline=pipeline,
                category=category,
                limit=limit,
            )
        examples: list[dict[str, Any]] = []
        for row in rows:
            title = ""
            if isinstance(row.concept_brief, dict):
                title = row.concept_brief.get("title", "")
            examples.append({
                "title": title,
                "category": row.category or "",
                "code": row.simulation_code,
                "reasoning": row.reasoning or "",
                "thinking": row.thinking_content or "",
                "iteration_count": row.iteration_count,
            })
        return examples
    except Exception as exc:
        logger.debug("Could not load few-shot examples: %s", exc)
        return []


def format_few_shot_prompt(examples: list[dict[str, Any]]) -> str:
    """Render few-shot examples as a prompt section.

    Returns an empty string when there are no examples.
    """
    if not examples:
        return ""
    lines = [
        "### Working Examples From Previous Successful Runs",
        "Use these as reference for structure, parameter ranges, and output format.\n",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(f"=== Example {i}: {ex.get('title', 'Untitled')} ===")
        if ex.get("reasoning"):
            lines.append(f"Design reasoning: {ex['reasoning']}")
        if ex.get("thinking"):
            # Truncate long thinking to keep prompt within budget
            thinking = ex["thinking"]
            if len(thinking) > 1500:
                thinking = thinking[:1500] + "…"
            lines.append(f"LLM thinking: {thinking}")
        lines.append(f"```python\n{ex['code']}\n```\n")
    return "\n".join(lines)


# ============================================================================
# Category knowledge — retrieve & update
# ============================================================================


async def get_category_knowledge_for_prompt(
    pipeline: str,
    category: str,
) -> str:
    """Load category knowledge and render as prompt text.

    Returns empty string when no knowledge exists or DB is unavailable.
    """
    try:
        from kairos.db.session import async_session_factory
        from kairos.db.operations import get_category_knowledge

        async with async_session_factory() as session:
            raw = await get_category_knowledge(session, pipeline, category)
        if not raw:
            return ""
        ck = CategoryKnowledge(**raw)
        return ck.to_prompt_text()
    except Exception as exc:
        logger.debug("Could not load category knowledge: %s", exc)
        return ""


async def update_category_knowledge(
    *,
    pipeline: str,
    category: str,
    iteration_count: int,
    validation_result: ValidationResult | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Update accumulated category knowledge after a run.

    Merges new data with whatever knowledge already exists in the DB.
    Respects ``learning_loop_enabled``.
    """
    settings = get_settings()
    if not settings.learning_loop_enabled:
        return

    try:
        from kairos.db.session import async_session_factory
        from kairos.db.operations import (
            get_category_knowledge,
            upsert_category_knowledge,
        )

        async with async_session_factory() as session:
            existing_raw = await get_category_knowledge(session, pipeline, category)
            ck = CategoryKnowledge(**(existing_raw or {}))
            ck.category = category

            # Merge iteration count
            n = ck.total_examples
            if n > 0 and ck.avg_iterations_to_pass:
                ck.avg_iterations_to_pass = (
                    (ck.avg_iterations_to_pass * n + iteration_count) / (n + 1)
                )
            else:
                ck.avg_iterations_to_pass = float(iteration_count)
            ck.total_examples = n + 1

            # Merge failed check names from this run
            if validation_result:
                for check in validation_result.failed_checks:
                    if check.name not in ck.common_failure_modes:
                        ck.common_failure_modes.append(check.name)
                # Keep only the 10 most recent failure modes
                ck.common_failure_modes = ck.common_failure_modes[-10:]

            # Merge parameter ranges from config
            if config:
                for key, val in config.items():
                    if isinstance(val, (int, float)):
                        existing = ck.parameter_ranges.get(key)
                        if existing and len(existing) == 2:
                            ck.parameter_ranges[key] = [
                                min(existing[0], float(val)),
                                max(existing[1], float(val)),
                            ]
                        else:
                            ck.parameter_ranges[key] = [float(val), float(val)]

            await upsert_category_knowledge(
                session,
                pipeline=pipeline,
                category=category,
                knowledge=ck.model_dump(),
            )
            await session.commit()

        logger.info(
            "Updated category knowledge (pipeline=%s, category=%s, total=%d)",
            pipeline,
            category,
            ck.total_examples,
        )
    except Exception as exc:
        logger.warning("Failed to update category knowledge: %s", exc)


# ============================================================================
# Validation → structured feedback (AI review §2)
# ============================================================================


def build_validation_feedback(
    validation_result: ValidationResult,
    iteration: int,
    max_iterations: int = 5,
    iteration_history: list[str] | None = None,
) -> ValidationFeedback:
    """Convert a ValidationResult into structured, actionable LLM feedback.

    Adds quantitative deltas, urgency signals, and iteration history.
    """
    # Determine urgency based on iteration number
    if iteration <= 1:
        urgency = "minor_tweak"
    elif iteration <= 3:
        urgency = "significant_change"
    else:
        urgency = "fundamental_rethink"

    failed: list[FailedCheck] = []
    suggested: dict[str, str] = {}

    for check in validation_result.failed_checks:
        actual_val = _to_float(check.value)
        threshold_val = _to_float(check.threshold)

        fc = FailedCheck(
            check_name=check.name,
            actual=actual_val,
            suggested_fix=_suggest_fix(check),
        )

        # Duration-specific enrichment
        if check.name == "duration" and actual_val is not None:
            target_min = 62.0
            target_max = 68.0
            if actual_val < target_min:
                delta = actual_val - target_min
                fc = fc.model_copy(update={
                    "target_min": target_min,
                    "target_max": target_max,
                    "delta": delta,
                    "suggested_fix": (
                        f"Increase SIMULATION_TIME by ~{abs(delta):.0f} seconds "
                        "or reduce physics speed."
                    ),
                })
                suggested["SIMULATION_TIME"] = f"increase by {abs(delta):.0f}s"
            elif actual_val > target_max:
                delta = actual_val - target_max
                fc = fc.model_copy(update={
                    "target_min": target_min,
                    "target_max": target_max,
                    "delta": delta,
                    "suggested_fix": (
                        f"Decrease SIMULATION_TIME by ~{abs(delta):.0f} seconds."
                    ),
                })
                suggested["SIMULATION_TIME"] = f"decrease by {abs(delta):.0f}s"

        # Resolution-specific enrichment
        elif check.name == "resolution":
            fc = fc.model_copy(update={
                "suggested_fix": "Ensure canvas is 1080×1920 (9:16 portrait).",
            })

        # FPS-specific enrichment
        elif check.name == "fps" and actual_val is not None and threshold_val is not None:
            delta = actual_val - threshold_val
            fc = fc.model_copy(update={
                "target_min": threshold_val,
                "delta": delta,
                "suggested_fix": (
                    "Reduce body count or simplify collision geometry."
                ),
            })

        # Completion ratio enrichment
        elif check.name == "completion_ratio" and actual_val is not None:
            fc = fc.model_copy(update={
                "target_min": 0.75,
                "delta": actual_val - 0.75,
                "suggested_fix": (
                    "Simulation chain/cascade did not complete. "
                    "Adjust trigger timing or spacing."
                ),
            })

        failed.append(fc)

    # Build iteration history summary
    history_summary = ""
    if iteration_history:
        history_summary = "\n".join(
            f"- Iteration {i + 1}: {h}" for i, h in enumerate(iteration_history)
        )

    return ValidationFeedback(
        failed_checks=failed,
        suggested_parameter_changes=suggested,
        similar_past_fixes=[],  # populated from DB in future enhancement
        iteration=iteration,
        max_iterations=max_iterations,
        urgency=urgency,
        iteration_history_summary=history_summary,
    )


def _to_float(val: Any) -> float | None:
    """Coerce a value to float or None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _suggest_fix(check: ValidationCheck) -> str:
    """Generate a human-readable fix suggestion from a failed check."""
    suggestions = {
        "valid_mp4": "Ensure FFmpeg pipe writes a valid MP4. Check for subprocess errors.",
        "duration": "Adjust SIMULATION_TIME to hit 62–68s target.",
        "resolution": "Ensure canvas is 1080×1920 (9:16 portrait).",
        "fps": "Reduce body count or simplify collision geometry to maintain ≥28 FPS.",
        "audio_present": "Audio is added by the video editor — this check can be skipped.",
        "completion_ratio": "Cascade/chain not completing — check trigger timing and spacing.",
    }
    return suggestions.get(check.name, check.message)


# ============================================================================
# Code-validation rules as prompt text (AI review §4)
# ============================================================================


def get_validation_rules_prompt() -> str:
    """Format static code_validation.py checks as prompt instructions.

    This is the 'dual representation' — the same rules exist as both
    validation checks AND LLM prompt instructions, so the LLM avoids
    mistakes instead of making them and getting caught.
    """
    return """### CRITICAL RULES (from past failure analysis)

These rules are enforced by automated validation.  Violations
will be caught and your code will be rejected.  Follow them exactly.

**Scene Setup:**
- ALWAYS configure gravity to (0, 0, -9.81) in Blender scene
- ALWAYS use Blender's rigid body physics for all dynamic objects
- Origin is world centre; Z is up

**Rendering:**
- ALWAYS use headless Blender rendering (no GUI)
- ALWAYS set render output format to FFMPEG / MP4
- ALWAYS configure output resolution to 1080x1920 (vertical short-form)

**Physics:**
- ALWAYS bake rigid body simulation before rendering
- ALWAYS set substeps_per_frame ≥ 10 for stable physics
- ALWAYS set solver_iterations ≥ 20 for accurate collisions

**Video Output:**
- ALWAYS output to the designated run output directory
- ALWAYS print `PAYOFF_TIMESTAMP=<seconds>` and `PEAK_BODY_COUNT=<count>` to stdout
"""
