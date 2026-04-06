"""Initial-push trigger for starting a domino chain.

Configures the first domino to receive an initial impulse that
topples it into the second domino.
"""

from __future__ import annotations

from typing import Any


def setup_trigger(
    first_domino_index: int = 0,
    *,
    force_magnitude: float = 10.0,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    method: str = "velocity",
) -> dict[str, Any]:
    """Return trigger config for the domino chain start.

    Two methods:
    - ``"velocity"``: Set initial linear velocity on the first domino.
    - ``"tilt"``: Pre-tilt the first domino a few degrees toward the
      second so gravity initiates the cascade.

    Args:
        first_domino_index: Index of the first domino in the placement list.
        force_magnitude: Impulse magnitude (N) or tilt degrees depending
            on *method*.
        direction: Unit vector for the push direction.
        method: ``"velocity"`` or ``"tilt"``.

    Returns:
        Trigger params dict for the Blender mesh builder.
    """
    return {
        "first_domino_index": first_domino_index,
        "method": method,
        "force_magnitude": force_magnitude,
        "direction": list(direction),
    }
