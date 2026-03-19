"""Kairos Agent -- Prompt Lineage Tracking.

Records the provenance of every rendered prompt so that each
``NNN_request.json`` in the run artifacts links back to its source
template, version, and content hash (D9).

Usage::

    from kairos.ai.tracing.lineage import compute_lineage

    lineage = compute_lineage(
        template_name="system/concept_developer.txt",
        raw_template=template_body,
        pipeline="physics",
        step="concept",
    )
    # lineage == {
    #   "template_name": "system/concept_developer.txt",
    #   "template_version": 3,
    #   "template_hash": "a1b2c3...",
    #   "pipeline": "physics",
    #   "step": "concept",
    # }
"""

from __future__ import annotations

import hashlib
from typing import Any

from kairos.ai.prompts.registry import PromptRegistry


def compute_template_hash(raw_template: str) -> str:
    """SHA-256 of the raw template text (first 12 hex chars)."""
    return hashlib.sha256(raw_template.encode("utf-8")).hexdigest()[:12]


def compute_lineage(
    *,
    template_name: str,
    raw_template: str,
    pipeline: str = "",
    step: str = "",
    version: int | None = None,
) -> dict[str, Any]:
    """Build a lineage metadata dict for a rendered prompt.

    Args:
        template_name: Relative path of the template (e.g. ``system/concept_developer.txt``).
        raw_template: The raw template body text (before variable substitution).
        pipeline: Pipeline name (physics, domino, ...).
        step: Step name (concept, simulation, ...).
        version: Override the version number.  If *None*, defaults to 0.

    Returns:
        A dict suitable for inclusion in ``NNN_request.json``.
    """
    return {
        "template_name": template_name,
        "template_version": version if version is not None else 0,
        "template_hash": compute_template_hash(raw_template),
        "pipeline": pipeline,
        "step": step,
    }


def compute_lineage_from_registry(
    registry: PromptRegistry,
    template_name: str,
    *,
    pipeline: str = "",
    step: str = "",
) -> dict[str, Any]:
    """Build lineage by loading a template from a ``PromptRegistry``.

    Fetches the version from front-matter and hashes the raw body.
    """
    meta, body = registry._load(template_name)
    return {
        "template_name": template_name,
        "template_version": meta.version,
        "template_hash": compute_template_hash(body),
        "pipeline": pipeline,
        "step": step,
    }
