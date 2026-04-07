"""Connector validator for marble run track pieces.

Validates that a sequence of track pieces connects properly:
- Diameter matching at each junction
- Direction alignment (no sharp reversals)
- Gap tolerance between exit and entry ports
- Momentum continuity through the complete track
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# Maximum allowed gap between connected ports (metres)
MAX_PORT_GAP_M = 0.02

# Maximum angle deviation between port directions (degrees)
MAX_DIRECTION_DEVIATION_DEG = 45.0

# Diameter tolerance for port matching (metres)
DIAMETER_TOLERANCE_M = 0.01


@dataclass(frozen=True)
class ConnectionCheck:
    """Result of validating a single connection between two pieces."""
    from_piece_index: int
    to_piece_index: int
    passed: bool
    gap_m: float
    direction_deviation_deg: float
    diameter_matched: bool
    message: str = ""


@dataclass
class TrackValidationResult:
    """Complete validation result for a marble run track."""
    passed: bool
    connection_checks: list[ConnectionCheck] = field(default_factory=list)
    momentum_checks: list[dict[str, Any]] = field(default_factory=list)
    total_pieces: int = 0
    total_length: float = 0.0
    total_height_drop: float = 0.0
    issues: list[str] = field(default_factory=list)


def _vec_length(v: list[float]) -> float:
    return math.sqrt(sum(c * c for c in v))


def _vec_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _vec_dot(a: list[float], b: list[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _angle_between_deg(a: list[float], b: list[float]) -> float:
    """Angle between two direction vectors in degrees."""
    la = _vec_length(a)
    lb = _vec_length(b)
    if la < 1e-9 or lb < 1e-9:
        return 0.0
    cos_angle = _vec_dot(a, b) / (la * lb)
    # Clamp for numerical stability
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def validate_connection(
    exit_port: dict[str, Any],
    entry_port: dict[str, Any],
    from_index: int,
    to_index: int,
) -> ConnectionCheck:
    """Validate the connection between two adjacent track pieces.

    Args:
        exit_port: Exit port descriptor from the upstream piece.
        entry_port: Entry port descriptor from the downstream piece.
        from_index: Index of the upstream piece.
        to_index: Index of the downstream piece.

    Returns:
        ConnectionCheck with pass/fail and diagnostics.
    """
    gap = _vec_distance(exit_port["position"], entry_port["position"])
    angle_dev = _angle_between_deg(exit_port["direction"], entry_port["direction"])

    exit_diam = exit_port.get("diameter", 0.05)
    entry_diam = entry_port.get("diameter", 0.05)
    diam_matched = abs(exit_diam - entry_diam) <= DIAMETER_TOLERANCE_M

    issues: list[str] = []
    if gap > MAX_PORT_GAP_M:
        issues.append(f"Gap {gap:.4f}m exceeds max {MAX_PORT_GAP_M}m")
    if angle_dev > MAX_DIRECTION_DEVIATION_DEG:
        issues.append(
            f"Direction deviation {angle_dev:.1f}° exceeds max {MAX_DIRECTION_DEVIATION_DEG}°"
        )
    if not diam_matched:
        issues.append(
            f"Diameter mismatch: exit={exit_diam:.4f}m, entry={entry_diam:.4f}m"
        )

    passed = len(issues) == 0
    message = "; ".join(issues) if issues else "OK"

    return ConnectionCheck(
        from_piece_index=from_index,
        to_piece_index=to_index,
        passed=passed,
        gap_m=round(gap, 6),
        direction_deviation_deg=round(angle_dev, 2),
        diameter_matched=diam_matched,
        message=message,
    )


def validate_track(
    pieces: list[dict[str, Any]],
    *,
    initial_speed: float = 0.0,
    min_exit_speed: float = 0.3,
) -> TrackValidationResult:
    """Validate a complete marble run track.

    Checks:
    1. All adjacent pieces connect (gap, direction, diameter)
    2. Momentum is sufficient through the chain

    Args:
        pieces: List of track piece build-param dicts (each must have
            entry_port, exit_port, height_drop, and length/arc_length).
        initial_speed: Speed at the start of the first piece.
        min_exit_speed: Minimum speed at each junction.

    Returns:
        TrackValidationResult with all checks.
    """
    if not pieces:
        return TrackValidationResult(
            passed=False,
            issues=["No track pieces provided"],
        )

    from kairos.skills.marble_run.momentum_calculator import validate_momentum_chain

    conn_checks: list[ConnectionCheck] = []
    issues: list[str] = []
    total_length = 0.0
    total_height_drop = 0.0

    # Connection checks
    for i in range(len(pieces) - 1):
        exit_port = pieces[i].get("exit_port")
        entry_port = pieces[i + 1].get("entry_port")

        if exit_port is None:
            issues.append(f"Piece {i} missing exit_port")
            continue
        if entry_port is None:
            issues.append(f"Piece {i + 1} missing entry_port")
            continue

        check = validate_connection(exit_port, entry_port, i, i + 1)
        conn_checks.append(check)
        if not check.passed:
            issues.append(f"Connection {i}->{i + 1}: {check.message}")

    # Build momentum segments
    segments: list[dict] = []
    for piece in pieces:
        h = piece.get("height_drop", 0.0)
        length = piece.get("length") or piece.get("arc_length") or piece.get("loop_circumference", 0.1)
        segments.append({"height_drop": h, "length": length})
        total_length += length
        total_height_drop += h

    momentum_checks = validate_momentum_chain(
        segments,
        initial_speed=initial_speed,
        min_exit_speed=min_exit_speed,
    )

    for i, mc in enumerate(momentum_checks):
        if not mc["passed"]:
            issues.append(
                f"Piece {i}: marble stalls (exit speed {mc['exit_speed']:.3f} m/s "
                f"< min {min_exit_speed} m/s)"
            )

    all_conn_passed = all(c.passed for c in conn_checks)
    all_momentum_passed = all(m["passed"] for m in momentum_checks)

    return TrackValidationResult(
        passed=all_conn_passed and all_momentum_passed and not issues,
        connection_checks=conn_checks,
        momentum_checks=momentum_checks,
        total_pieces=len(pieces),
        total_length=round(total_length, 4),
        total_height_drop=round(total_height_drop, 4),
        issues=issues,
    )
