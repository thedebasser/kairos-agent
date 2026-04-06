"""Domino size gradient — vary domino dimensions along a path.

Optional effect: dominos can grow or shrink along the chain for
visual interest (e.g. tiny-to-giant or giant-to-tiny).
"""

from __future__ import annotations


def compute_size_gradient(
    index: int,
    total_count: int,
    *,
    start_scale: float = 1.0,
    end_scale: float = 1.0,
    easing: str = "linear",
) -> float:
    """Compute the scale factor for a domino at *index* within *total_count*.

    Args:
        index: Zero-based index of the domino in the chain.
        total_count: Total number of dominos.
        start_scale: Scale at the first domino.
        end_scale: Scale at the last domino.
        easing: Interpolation curve — ``"linear"`` or ``"ease_in_out"``.

    Returns:
        Scale multiplier (1.0 = default size).
    """
    if total_count <= 1:
        return start_scale

    t = index / (total_count - 1)

    if easing == "ease_in_out":
        # Smoothstep: 3t² - 2t³
        t = t * t * (3.0 - 2.0 * t)

    return start_scale + (end_scale - start_scale) * t
