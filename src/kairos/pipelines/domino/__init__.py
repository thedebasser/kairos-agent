"""Domino pipeline — Blender-based domino run shorts.

Pipeline 3 using Blender rigid body physics for procedural domino runs.
Dominoes placed along curves (spirals, S-curves, branching paths) on a
ground plane. Physics simulation handles the toppling cascade.
"""

from kairos.pipelines.adapters.domino_adapter import DominoPipelineAdapter

__all__ = ["DominoPipelineAdapter"]
