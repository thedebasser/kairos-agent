"""Simulation configuration schemas.

The LLM generates JSON matching these schemas instead of writing raw Python.
A fixed template per category consumes the JSON and runs the simulation.
"""

from kairos.pipelines.physics.configs.base import BaseSimulationConfig
from kairos.pipelines.physics.configs.ball_pit import BallPitConfig
from kairos.pipelines.physics.configs.domino_chain import DominoChainConfig
from kairos.pipelines.physics.configs.destruction import DestructionConfig
from kairos.pipelines.physics.configs.marble_funnel import MarbleFunnelConfig

# Map category name → config class for dynamic dispatch
CONFIG_REGISTRY: dict[str, type[BaseSimulationConfig]] = {
    "ball_pit": BallPitConfig,
    "domino_chain": DominoChainConfig,
    "destruction": DestructionConfig,
    "marble_funnel": MarbleFunnelConfig,
}

__all__ = [
    "BaseSimulationConfig",
    "BallPitConfig",
    "DominoChainConfig",
    "DestructionConfig",
    "MarbleFunnelConfig",
    "CONFIG_REGISTRY",
]
