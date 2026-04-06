"""Primitive contract schema for the Kairos skill library.

Every skill primitive follows a standard contract:
- Declared inputs with types and ranges
- Declared outputs
- Test criteria for validation
- Category and content-type compatibility tags

Path primitives are pure math (no bpy dependency).
Connector, placement, and environment primitives run inside Blender.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContentType(str, Enum):
    """Content types supported by the pipeline."""
    DOMINO = "domino"
    MARBLE_RUN = "marble_run"
    ALL = "all"


class PrimitiveCategory(str, Enum):
    """Skill library primitive categories."""
    PATH = "path"
    CONNECTOR = "connector"
    SURFACE = "surface"
    ENVIRONMENT = "environment"
    PLACEMENT = "placement"


@dataclass(frozen=True)
class Vector3:
    """Lightweight 3D vector for path waypoints."""
    x: float
    y: float
    z: float

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, idx: int) -> float:
        return (self.x, self.y, self.z)[idx]

    def distance_to(self, other: Vector3) -> float:
        import math
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2,
        )

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def length(self) -> float:
        import math
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> Vector3:
        L = self.length()
        if L < 1e-9:
            return Vector3(0, 0, 0)
        return Vector3(self.x / L, self.y / L, self.z / L)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float]) -> Vector3:
        return cls(t[0], t[1], t[2])


@dataclass
class PathResult:
    """Output of a path primitive — ordered waypoints with metadata."""
    waypoints: list[Vector3]
    total_length: float
    segment_type: str  # e.g. "straight_line", "arc", "spiral"
    gradients: list[float] = field(default_factory=list)  # gradient at each waypoint (degrees)

    @property
    def point_count(self) -> int:
        return len(self.waypoints)


@dataclass
class ConnectorResult:
    """Output of a connector primitive — waypoints + build instructions."""
    waypoints: list[Vector3]
    connector_type: str  # e.g. "ramp", "spiral_ramp", "staircase"
    params: dict[str, Any] = field(default_factory=dict)
    total_length: float = 0.0
    max_gradient: float = 0.0  # degrees
    footprint: tuple[float, float] = (0.0, 0.0)  # (width, depth) bounding box on XY


@dataclass
class EnvironmentConfig:
    """Configuration for scene environment setup."""
    ground_texture: str
    lighting_preset: str
    environment_type: str  # "indoor" or "outdoor"
    hdri: str | None = None  # Only for outdoor
    ground_material_params: dict[str, Any] = field(default_factory=dict)
    lighting_params: dict[str, Any] = field(default_factory=dict)
