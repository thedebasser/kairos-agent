"""Momentum calculator for marble run track validation.

Computes whether a marble has sufficient velocity at each track piece
junction to continue through the next segment.  Uses conservation of
energy with empirical friction losses.
"""

from __future__ import annotations

import math


# Gravitational acceleration (m/s²)
_G = 9.81

# Empirical friction loss coefficient per metre of track
# (accounts for rolling friction + wall contact).
_DEFAULT_FRICTION_LOSS_PER_M = 0.15


def compute_momentum_required(
    height_drop: float,
    segment_length: float,
    *,
    min_exit_speed: float = 0.3,
    friction_loss_per_m: float = _DEFAULT_FRICTION_LOSS_PER_M,
) -> float:
    """Compute minimum entry speed for a marble to traverse a segment.

    Uses energy conservation:
        ½mv² + mgh = ½mv_exit² + friction_loss
        v_entry = sqrt(v_exit² - 2gh + 2·f·L)

    where h is positive when dropping (marble gains energy) and
    negative when climbing (marble loses energy).

    Args:
        height_drop: Height change in metres (positive = downhill).
        segment_length: Track length in metres.
        min_exit_speed: Required exit speed in m/s.
        friction_loss_per_m: Energy loss coefficient per metre.

    Returns:
        Required entry speed in m/s. Returns 0.0 if gravity provides
        sufficient energy (marble can start from rest).
    """
    # Energy balance: v² = v_exit² - 2g·h_drop + 2·friction·length
    # h_drop positive = downhill = gain energy = negative term
    v_sq = (
        min_exit_speed ** 2
        - 2.0 * _G * height_drop
        + 2.0 * friction_loss_per_m * segment_length
    )
    if v_sq <= 0.0:
        return 0.0
    return math.sqrt(v_sq)


def compute_exit_speed(
    entry_speed: float,
    height_drop: float,
    segment_length: float,
    *,
    friction_loss_per_m: float = _DEFAULT_FRICTION_LOSS_PER_M,
) -> float:
    """Compute marble exit speed after traversing a segment.

    Args:
        entry_speed: Marble speed at segment entry in m/s.
        height_drop: Height change (positive = downhill).
        segment_length: Track length in metres.
        friction_loss_per_m: Friction loss coefficient per metre.

    Returns:
        Exit speed in m/s. Returns 0.0 if the marble stalls.
    """
    v_sq = (
        entry_speed ** 2
        + 2.0 * _G * height_drop
        - 2.0 * friction_loss_per_m * segment_length
    )
    if v_sq <= 0.0:
        return 0.0
    return math.sqrt(v_sq)


def validate_momentum_chain(
    segments: list[dict],
    *,
    initial_speed: float = 0.0,
    min_exit_speed: float = 0.3,
    friction_loss_per_m: float = _DEFAULT_FRICTION_LOSS_PER_M,
) -> list[dict]:
    """Validate momentum through a sequence of track segments.

    Each segment dict must have:
        - "height_drop": float (positive = downhill)
        - "length": float (track length in metres)

    Args:
        segments: List of segment dicts with height_drop and length.
        initial_speed: Speed at the start of the first segment.
        min_exit_speed: Minimum speed to consider the marble still moving.
        friction_loss_per_m: Friction loss coefficient.

    Returns:
        List of result dicts with entry_speed, exit_speed and passed flag
        for each segment.
    """
    results: list[dict] = []
    speed = initial_speed

    for seg in segments:
        h = seg["height_drop"]
        length = seg["length"]

        exit_speed = compute_exit_speed(
            speed, h, length,
            friction_loss_per_m=friction_loss_per_m,
        )

        results.append({
            "entry_speed": round(speed, 4),
            "exit_speed": round(exit_speed, 4),
            "height_drop": h,
            "length": length,
            "passed": exit_speed >= min_exit_speed,
        })

        speed = exit_speed

    return results
