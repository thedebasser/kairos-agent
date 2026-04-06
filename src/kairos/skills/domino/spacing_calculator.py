"""Domino spacing calculator with calibration-aware corrections.

Implements the spacing formula from the design doc:

    base_spacing = domino_height × 0.35
    if gradient > 0: base_spacing *= (1.0 - 0.01 × |gradient|)   # tighter uphill
    if gradient < 0: base_spacing *= (1.0 + 0.005 × |gradient|)  # wider downhill
    if curvature > 0: base_spacing *= (1.0 - 0.1 × curvature)    # tighter on curves
    if calibration.spacing_correction: base_spacing *= correction  # ChromaDB correction
"""

from __future__ import annotations


def compute_spacing(
    domino_height: float = 0.08,
    *,
    gradient_deg: float = 0.0,
    curvature: float = 0.0,
    calibration_correction: float | None = None,
    base_ratio: float = 0.35,
    min_spacing_ratio: float = 0.15,
    max_spacing_ratio: float = 0.60,
) -> float:
    """Compute the centre-to-centre spacing for adjacent dominos.

    Args:
        domino_height: Height of a single domino (metres).
        gradient_deg: Path gradient at the current waypoint (degrees).
            Positive = uphill, negative = downhill.
        curvature: Normalised curvature at the current waypoint [0–1].
            0 = straight, 1 = tightest expected curve.
        calibration_correction: Optional multiplier from ChromaDB
            calibration data.  ``None`` = no correction.
        base_ratio: Baseline spacing as a fraction of domino height.
        min_spacing_ratio: Floor clamp (fraction of domino height).
        max_spacing_ratio: Ceiling clamp (fraction of domino height).

    Returns:
        Spacing in metres.
    """
    spacing = domino_height * base_ratio

    # Gradient adjustment
    if gradient_deg > 0:
        spacing *= 1.0 - 0.01 * abs(gradient_deg)
    elif gradient_deg < 0:
        spacing *= 1.0 + 0.005 * abs(gradient_deg)

    # Curvature adjustment
    if curvature > 0:
        spacing *= 1.0 - 0.1 * curvature

    # Calibration correction from ChromaDB
    if calibration_correction is not None:
        spacing *= calibration_correction

    # Clamp to safe bounds
    min_spacing = domino_height * min_spacing_ratio
    max_spacing = domino_height * max_spacing_ratio
    return max(min_spacing, min(spacing, max_spacing))
