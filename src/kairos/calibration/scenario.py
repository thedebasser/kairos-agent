"""Kairos Agent — Scenario ↔ DominoCourseConfig Mapping.

Bridges the calibration system's ScenarioDescriptor with the existing
domino pipeline's DominoCourseConfig.  Enables the calibration system to
produce configs that feed directly into the existing Blender scripts.
"""

from __future__ import annotations

import logging
from typing import Any

from kairos.calibration.models import (
    BASELINE_PHYSICS,
    CorrectionFactors,
    PathDescriptor,
    PathType,
    ScenarioDescriptor,
    SizeProfile,
    SizeProfileDescriptor,
    SurfaceDescriptor,
    SurfaceType,
)

logger = logging.getLogger(__name__)

# Map PathType → DominoArchetype string values.
# The existing Blender script uses archetype names, not generic path types.
_PATH_TO_ARCHETYPE: dict[PathType, str] = {
    PathType.STRAIGHT: "s_curve",       # s_curve with amplitude=0 is a straight line
    PathType.ARC: "s_curve",            # single arc via low cycles + amplitude
    PathType.S_CURVE: "s_curve",
    PathType.SPIRAL: "spiral",
    PathType.CASCADE: "cascade",
    PathType.BRANCHING: "branching",
    PathType.WORD_SPELL: "word_spell",
}


def scenario_to_blender_config(
    scenario: ScenarioDescriptor,
    corrections: CorrectionFactors | None = None,
) -> dict[str, Any]:
    """Convert a ScenarioDescriptor + corrections into a Blender config dict.

    This produces the same flat dict format that generate_domino_course.py
    expects, making the calibration system compatible with the existing
    Blender scripts without modification.
    """
    if corrections is None:
        corrections = CorrectionFactors()

    # Start with baseline physics, apply correction factors
    physics = corrections.apply_to_baseline()

    archetype = _PATH_TO_ARCHETYPE.get(scenario.path.type, "s_curve")

    config: dict[str, Any] = {
        "seed": 42,
        "archetype": archetype,
        "domino_count": scenario.domino_count,
        # Path params
        "path_amplitude": scenario.path.amplitude or (0.0 if scenario.path.type == PathType.STRAIGHT else 1.0),
        "path_cycles": scenario.path.cycles or 2.0,
        "spiral_turns": scenario.path.spiral_turns or 2.0,
        "branch_count": scenario.path.branch_count or 3,
        # Domino geometry
        "domino_width": 0.08,
        "domino_height": scenario.size_profile.start_height,
        "domino_depth": 0.06,
        # Physics (from corrections applied to baseline)
        "spacing_ratio": physics["spacing_ratio"],
        "domino_mass": physics["domino_mass"],
        "domino_friction": physics["domino_friction"],
        "domino_bounce": physics["domino_bounce"],
        "ground_friction": physics["ground_friction"],
        "trigger_impulse": physics["trigger_impulse"],
        "trigger_tilt_degrees": physics["trigger_tilt_degrees"],
        "substeps_per_frame": int(physics["substeps_per_frame"]),
        "solver_iterations": int(physics["solver_iterations"]),
        # Timing
        "trigger_frame": 30,
        "duration_sec": 65,
        "fps": 30,
        # Visual (defaults for calibration — not the focus)
        "palette": "rainbow",
        "camera_style": "tracking",
        "lighting_preset": "studio",
        "finale_type": "none",
    }

    return config


def blender_config_to_scenario(config: dict[str, Any]) -> ScenarioDescriptor:
    """Extract a ScenarioDescriptor from an existing Blender config dict.

    Useful for converting existing pipeline configs into the calibration
    system's format for ChromaDB lookup.
    """
    archetype = config.get("archetype", "s_curve")

    # Map archetype back to PathType
    archetype_to_path: dict[str, PathType] = {
        "spiral": PathType.SPIRAL,
        "s_curve": PathType.S_CURVE,
        "branching": PathType.BRANCHING,
        "cascade": PathType.CASCADE,
        "word_spell": PathType.WORD_SPELL,
    }
    path_type = archetype_to_path.get(archetype, PathType.S_CURVE)

    # Detect straight line: s_curve with zero amplitude
    amplitude = config.get("path_amplitude", 1.0)
    if archetype == "s_curve" and amplitude == 0.0:
        path_type = PathType.STRAIGHT

    path = PathDescriptor(
        type=path_type,
        amplitude=amplitude if path_type in (PathType.S_CURVE, PathType.ARC) else None,
        cycles=config.get("path_cycles") if path_type in (PathType.S_CURVE, PathType.ARC) else None,
        spiral_turns=config.get("spiral_turns") if path_type == PathType.SPIRAL else None,
        branch_count=config.get("branch_count") if path_type == PathType.BRANCHING else None,
    )

    return ScenarioDescriptor(
        path=path,
        surface=SurfaceDescriptor(type=SurfaceType.FLAT),
        size_profile=SizeProfileDescriptor(
            type=SizeProfile.UNIFORM,
            start_height=config.get("domino_height", 0.4),
            end_height=config.get("domino_height", 0.4),
        ),
        domino_count=config.get("domino_count", 300),
    )


def compute_dimensional_overlap(
    query: ScenarioDescriptor,
    stored_meta: dict[str, Any],
) -> tuple[float, list[str]]:
    """Compute how many structural dimensions match between a query and stored entry.

    Returns (overlap_score, matching_dimension_names).
    Each matching dimension contributes a weighted score.
    """
    score = 0.0
    matches: list[str] = []

    weights = {
        "path_type": 0.30,
        "surface_type": 0.20,
        "size_profile": 0.15,
        "domino_count_similar": 0.10,
        "domino_height_similar": 0.10,
        "path_amplitude_similar": 0.08,
        "spiral_turns_similar": 0.07,
    }

    # Path type match
    if query.path.type.value == stored_meta.get("path_type"):
        score += weights["path_type"]
        matches.append("path_type")

    # Surface type match
    if query.surface.type.value == stored_meta.get("surface_type"):
        score += weights["surface_type"]
        matches.append("surface_type")

    # Size profile match
    if query.size_profile.type.value == stored_meta.get("size_profile"):
        score += weights["size_profile"]
        matches.append("size_profile")

    # Domino count similarity (within ±30%)
    stored_count = stored_meta.get("domino_count", 0)
    if stored_count > 0:
        ratio = min(query.domino_count, stored_count) / max(query.domino_count, stored_count)
        if ratio >= 0.7:
            score += weights["domino_count_similar"]
            matches.append("domino_count_similar")

    # Domino height similarity (within ±20%)
    stored_height = stored_meta.get("domino_height", 0)
    if stored_height > 0:
        ratio = min(query.size_profile.start_height, stored_height) / max(
            query.size_profile.start_height, stored_height
        )
        if ratio >= 0.8:
            score += weights["domino_height_similar"]
            matches.append("domino_height_similar")

    # Path amplitude similarity (if both have it)
    if query.path.amplitude is not None and "path_amplitude" in stored_meta:
        stored_amp = stored_meta["path_amplitude"]
        if stored_amp > 0:
            ratio = min(query.path.amplitude, stored_amp) / max(query.path.amplitude, stored_amp)
            if ratio >= 0.7:
                score += weights["path_amplitude_similar"]
                matches.append("path_amplitude_similar")

    # Spiral turns similarity
    if query.path.spiral_turns is not None and "spiral_turns" in stored_meta:
        stored_turns = stored_meta["spiral_turns"]
        if stored_turns > 0:
            ratio = min(query.path.spiral_turns, stored_turns) / max(
                query.path.spiral_turns, stored_turns
            )
            if ratio >= 0.7:
                score += weights["spiral_turns_similar"]
                matches.append("spiral_turns_similar")

    return round(score, 4), matches


def composite_corrections(
    matches: list[dict[str, Any]],
    query: ScenarioDescriptor,
) -> CorrectionFactors:
    """Combine correction factors from multiple partial matches.

    Uses weighted blending based on combined_score × confidence.
    Starts from the baseline (all 1.0) and blends in corrections
    from matching dimensions.
    """
    if not matches:
        return CorrectionFactors()

    # Accumulate weighted corrections
    weighted_sums: dict[str, float] = {}
    weight_totals: dict[str, float] = {}

    for match in matches:
        corrections = match.get("corrections")
        if corrections is None:
            continue

        if isinstance(corrections, dict):
            corrections = CorrectionFactors(**corrections)

        weight = match.get("combined_score", 0.5) * match.get("confidence", 0.5)
        dims = match.get("matching_dimensions", [])

        for field_name in CorrectionFactors.model_fields:
            if field_name == "notes":
                continue
            val = getattr(corrections, field_name)
            if val == 1.0:
                continue  # Skip default values — they carry no information

            # Only apply corrections from matching dimensions
            should_apply = (
                "path_type" in dims
                or "surface_type" in dims
                or field_name in ("substeps_per_frame", "solver_iterations")
            )
            if not should_apply:
                continue

            if field_name not in weighted_sums:
                weighted_sums[field_name] = 0.0
                weight_totals[field_name] = 0.0

            weighted_sums[field_name] += val * weight
            weight_totals[field_name] += weight

    # Compute weighted averages, falling back to 1.0 (baseline)
    result: dict[str, Any] = {}
    for field_name in CorrectionFactors.model_fields:
        if field_name == "notes":
            continue
        if field_name in weighted_sums and weight_totals[field_name] > 0:
            result[field_name] = round(weighted_sums[field_name] / weight_totals[field_name], 6)
        else:
            result[field_name] = 1.0

    return CorrectionFactors(**result)
