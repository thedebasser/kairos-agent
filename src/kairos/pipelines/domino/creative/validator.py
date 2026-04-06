"""Per-step validator — validates each creative agent's output.

Runs deterministic structural checks after every agent step.
Returns a StepValidationResult that the pipeline uses for
failure attribution and cascade re-run decisions.
"""

from __future__ import annotations

import logging
import math

from kairos.pipelines.domino.creative.models import (
    AgentRole,
    ConnectorOutput,
    ObjectRole,
    PathOutput,
    SceneManifest,
    SegmentType,
    StepValidationResult,
)

logger = logging.getLogger(__name__)

# ─── Thresholds ──────────────────────────────────────────────────────

MAX_GRADIENT_DEGREES = 15.0
SCENE_HALF_EXTENT = 5.0  # 10×10 m area ⇒ ±5m from origin
MIN_FUNCTIONAL_OBJECTS = 2
MAX_OBJECTS = 30
MIN_SEGMENTS = 2


def _check(name: str, passed: bool, message: str = "") -> dict:
    return {"name": name, "passed": passed, "message": message}


# ─── Scene Manifest Validation ───────────────────────────────────────


def validate_scene(manifest: SceneManifest) -> StepValidationResult:
    """Validate Set Designer output."""
    checks: list[dict] = []

    # 1. Functional object count
    functional = [o for o in manifest.objects if o.role == ObjectRole.FUNCTIONAL]
    checks.append(
        _check(
            "min_functional_objects",
            len(functional) >= MIN_FUNCTIONAL_OBJECTS,
            f"Need ≥{MIN_FUNCTIONAL_OBJECTS} functional objects, got {len(functional)}",
        )
    )

    # 2. Total object count sanity
    checks.append(
        _check(
            "max_objects",
            len(manifest.objects) <= MAX_OBJECTS,
            f"Too many objects: {len(manifest.objects)} (max {MAX_OBJECTS})",
        )
    )

    # 3. Functional objects must have surface_name
    missing_surface = [
        o.asset_id for o in functional if not o.surface_name
    ]
    checks.append(
        _check(
            "functional_surfaces",
            len(missing_surface) == 0,
            f"Functional objects without surface_name: {missing_surface}",
        )
    )

    # 4. Out-of-bounds check
    oob = [
        o.asset_id
        for o in manifest.objects
        if (
            abs(o.position[0]) > SCENE_HALF_EXTENT
            or abs(o.position[1]) > SCENE_HALF_EXTENT
        )
    ]
    checks.append(
        _check(
            "in_bounds",
            len(oob) == 0,
            f"Objects outside 10×10m area: {oob}",
        )
    )

    # 5. Overlapping positions (simple centre-distance check)
    overlap_pairs: list[str] = []
    objs = manifest.objects
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            dx = objs[i].position[0] - objs[j].position[0]
            dy = objs[i].position[1] - objs[j].position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.1:  # within 10 cm
                overlap_pairs.append(f"{objs[i].asset_id}<->{objs[j].asset_id}")
    checks.append(
        _check(
            "no_overlaps",
            len(overlap_pairs) == 0,
            f"Overlapping objects: {overlap_pairs}",
        )
    )

    # 6. Different elevations among functional objects
    heights = {o.position[2] for o in functional}
    checks.append(
        _check(
            "varied_heights",
            len(heights) >= 2,
            f"Functional objects should be at ≥2 different heights, got {len(heights)}",
        )
    )

    passed = all(c["passed"] for c in checks)
    errors = [c["message"] for c in checks if not c["passed"]]
    error_summary = "; ".join(errors) if errors else ""

    result = StepValidationResult(
        agent=AgentRole.SET_DESIGNER,
        passed=passed,
        checks=checks,
        error_summary=error_summary,
    )
    logger.info("[validator] Scene: %s — %s", "PASS" if passed else "FAIL", error_summary or "OK")
    return result


# ─── Path Output Validation ──────────────────────────────────────────


def validate_path(path_output: PathOutput, manifest: SceneManifest) -> StepValidationResult:
    """Validate Path Setter output against the scene manifest."""
    checks: list[dict] = []

    # 1. At least N segments
    checks.append(
        _check(
            "min_segments",
            len(path_output.segments) >= MIN_SEGMENTS,
            f"Need ≥{MIN_SEGMENTS} segments, got {len(path_output.segments)}",
        )
    )

    # 2. At least one height transition flagged
    transitions = [s for s in path_output.segments if s.needs_connector]
    checks.append(
        _check(
            "has_transitions",
            len(transitions) >= 1,
            f"Path should flag ≥1 height transition, got {len(transitions)}",
        )
    )

    # 3. Gradient limit
    steep = [
        s.id for s in path_output.segments
        if s.gradient > MAX_GRADIENT_DEGREES
    ]
    checks.append(
        _check(
            "gradient_limit",
            len(steep) == 0,
            f"Segments with gradient > {MAX_GRADIENT_DEGREES}°: {steep}",
        )
    )

    # 4. Transition segments must have from/to height set
    bad_heights = [
        s.id for s in transitions
        if s.from_height == 0.0 and s.to_height == 0.0
    ]
    checks.append(
        _check(
            "transition_heights",
            len(bad_heights) == 0,
            f"Transition segments with zero height delta: {bad_heights}",
        )
    )

    # 5. Segments should have waypoints (or be transitions waiting for connector)
    empty_waypoints = [
        s.id for s in path_output.segments
        if not s.needs_connector and len(s.waypoints) == 0
    ]
    checks.append(
        _check(
            "segment_waypoints",
            len(empty_waypoints) == 0,
            f"Non-transition segments missing waypoints: {empty_waypoints}",
        )
    )

    # 6. Domino count carried forward
    checks.append(
        _check(
            "domino_count",
            path_output.domino_count > 0,
            "Domino count should be positive",
        )
    )

    passed = all(c["passed"] for c in checks)
    errors = [c["message"] for c in checks if not c["passed"]]
    error_summary = "; ".join(errors) if errors else ""

    result = StepValidationResult(
        agent=AgentRole.PATH_SETTER,
        passed=passed,
        checks=checks,
        error_summary=error_summary,
    )
    logger.info("[validator] Path: %s — %s", "PASS" if passed else "FAIL", error_summary or "OK")
    return result


# ─── Connector Output Validation ─────────────────────────────────────


def validate_connectors(
    connector_output: ConnectorOutput,
    path_output: PathOutput,
) -> StepValidationResult:
    """Validate Connector Agent output against the path specification."""
    checks: list[dict] = []

    # 1. All transition segments have a matching connector
    transition_ids = {s.id for s in path_output.segments if s.needs_connector}
    resolved_ids = {c.for_segment for c in connector_output.connectors}
    missing = transition_ids - resolved_ids
    checks.append(
        _check(
            "all_transitions_resolved",
            len(missing) == 0,
            f"Unresolved transitions: {missing}",
        )
    )

    # 2. Connectors have generated waypoints
    empty_connectors = [
        c.id for c in connector_output.connectors
        if len(c.generated_waypoints) == 0
    ]
    checks.append(
        _check(
            "connector_waypoints",
            len(empty_connectors) == 0,
            f"Connectors with no waypoints: {empty_connectors}",
        )
    )

    # 3. Complete path has waypoints
    checks.append(
        _check(
            "complete_path_nonempty",
            len(connector_output.complete_path_waypoints) > 0,
            "Complete merged path has no waypoints",
        )
    )

    # 4. Segment types length matches segments
    expected_segments = len(path_output.segments)
    actual_types = len(connector_output.segment_types)
    checks.append(
        _check(
            "segment_types_count",
            actual_types == expected_segments,
            f"segment_types length ({actual_types}) ≠ path segments ({expected_segments})",
        )
    )

    passed = all(c["passed"] for c in checks)
    errors = [c["message"] for c in checks if not c["passed"]]
    error_summary = "; ".join(errors) if errors else ""

    result = StepValidationResult(
        agent=AgentRole.CONNECTOR,
        passed=passed,
        checks=checks,
        error_summary=error_summary,
    )
    logger.info("[validator] Connectors: %s — %s", "PASS" if passed else "FAIL", error_summary or "OK")
    return result
