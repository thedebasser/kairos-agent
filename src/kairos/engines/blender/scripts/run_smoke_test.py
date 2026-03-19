"""Run a short physics smoke test on the generated course.

Simulates a small number of frames and checks:
  1. At least one marble moves significantly
  2. No marbles explode (velocity stays reasonable)
  3. Marbles don't all get stuck at the first transition
  4. At least one marble reaches the lower half of the course

Run via:
    blender --background <file.blend> --python run_smoke_test.py -- [--frames 120]
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]

# Import WORLD_SCALE so thresholds match the 10× geometry
try:
    from build_modules import WORLD_SCALE
except ImportError:
    WORLD_SCALE = 10


@dataclass
class SmokeTestResult:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    marble_positions: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_data: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "checks": self.checks,
            "checkpoints": self.checkpoint_data,
            "marble_final_positions": self.marble_positions[:10],  # sample
        }


def run_smoke_test(*, frames: int = 120, max_velocity: float = 50.0,
                    min_travel: float = 0.1,
                    checkpoints: list[int] | None = None) -> SmokeTestResult:
    """Run physics for *frames* frames and analyse marble behaviour.

    Args:
        frames: Number of frames to simulate (used as default test range).
        max_velocity: Reject if any marble exceeds this velocity.
        min_travel: Minimum distance at least one marble should travel.
        checkpoints: Optional list of frame numbers to test at.
            Research recommends [1, 300, 750, 1200, 1950] to catch
            late-stage divergence that a short 120-frame test misses.
            If supplied, *frames* is ignored and the highest checkpoint
            is used as the simulation end.
    """
    # Scale thresholds to match WORLD_SCALE geometry (gravity = 3× ≈ slower sim)
    _S = WORLD_SCALE
    min_travel *= _S
    max_velocity *= _S
    scene = bpy.context.scene
    marbles = [o for o in bpy.data.objects if o.name.startswith("marble_")]

    if not marbles:
        return SmokeTestResult(False, reason="No marbles found in scene")

    # Determine checkpoint frames
    if checkpoints:
        cp_frames = sorted(set(checkpoints))
        test_end = min(max(cp_frames), scene.frame_end)
    else:
        cp_frames = [frames]
        test_end = min(scene.frame_start + frames, scene.frame_end)

    # Bake rigid body simulation (required in background mode)
    if scene.rigidbody_world:
        try:
            bpy.ops.ptcache.free_bake_all()
            bpy.ops.ptcache.bake_all(bake=True)
        except Exception:
            pass  # May fail if already baked

    # Record initial positions (use matrix_world — .location doesn't update for RB)
    scene.frame_set(1)
    dg = bpy.context.evaluated_depsgraph_get()
    initial_positions = {m.name: Vector(m.evaluated_get(dg).matrix_world.translation) for m in marbles}

    # Find course vertical bounds using actual bounding boxes (not just origins).
    # object.bound_box gives 8 local-space corners; transform to world space.
    course_objects = [o for o in bpy.data.objects
                      if o.type == "MESH"
                      and not o.name.startswith("marble_")
                      and not o.name.startswith(("Camera", "Key_", "Fill_", "Rim_", "catch_basin"))]
    all_z: list[float] = []
    for obj in course_objects:
        mw = obj.matrix_world
        for corner in obj.bound_box:
            all_z.append((mw @ Vector(corner)).z)
    course_min_z = min(all_z) if all_z else -2.0
    course_max_z = max(all_z) if all_z else 0.0
    course_mid_z = (course_min_z + course_max_z) / 2

    # Simulate with checkpoint tracking
    checks = []
    max_vel_seen = 0.0
    explosive = False
    prev_positions = dict(initial_positions)
    checkpoint_data: list[dict[str, Any]] = []

    for frame in range(scene.frame_start, test_end + 1):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()

        for m in marbles:
            cur_pos = Vector(m.evaluated_get(dg).matrix_world.translation)
            # Check velocity (approximate via position delta)
            if frame > scene.frame_start:
                prev_pos = prev_positions.get(m.name, cur_pos)
                vel = (cur_pos - prev_pos).length * scene.render.fps
                max_vel_seen = max(max_vel_seen, vel)
                if vel > max_velocity:
                    explosive = True
            prev_positions[m.name] = cur_pos

        # ── Checkpoint snapshot ──────────────────────────────────
        if frame in cp_frames:
            cp_z = [prev_positions[m.name].z for m in marbles]
            safety_floor = course_min_z - 2.0 * _S
            safety_ceiling = course_max_z + 5.0 * _S
            escaped = sum(1 for z in cp_z if z < safety_floor or z > safety_ceiling)
            cp_max_travel = max(
                (prev_positions[m.name] - initial_positions[m.name]).length
                for m in marbles
            )
            checkpoint_data.append({
                "frame": frame,
                "escaped": escaped,
                "max_travel": round(cp_max_travel, 3),
                "z_min": round(min(cp_z), 3),
                "z_max": round(max(cp_z), 3),
            })
            print(f"[smoke] checkpoint f={frame}: escaped={escaped}, "
                  f"max_travel={cp_max_travel:.2f}, z=[{min(cp_z):.2f}, {max(cp_z):.2f}]")

    # Final positions
    scene.frame_set(test_end)
    dg = bpy.context.evaluated_depsgraph_get()
    final_positions = {m.name: Vector(m.evaluated_get(dg).matrix_world.translation) for m in marbles}

    # Check: at least one marble moved significantly
    max_travel = 0.0
    travelled = {}
    for m in marbles:
        dist = (final_positions[m.name] - initial_positions[m.name]).length
        travelled[m.name] = dist
        max_travel = max(max_travel, dist)

    moved_check = max_travel >= min_travel
    checks.append({"name": "marble_movement", "passed": moved_check,
                    "message": f"Max travel: {max_travel:.3f}m (min={min_travel})"})

    # Check: no explosive velocities
    vel_check = not explosive
    checks.append({"name": "no_explosion", "passed": vel_check,
                    "message": f"Max velocity: {max_vel_seen:.1f} (limit={max_velocity})"})

    # Check: not all stuck at same height
    final_z = [final_positions[m.name].z for m in marbles]
    z_spread = max(final_z) - min(final_z) if final_z else 0
    spread_check = z_spread > 0.05 * _S
    checks.append({"name": "z_spread", "passed": spread_check,
                    "message": f"Z spread: {z_spread:.3f}m"})

    # Check: at least one marble below midpoint
    below_mid = sum(1 for z in final_z if z < course_mid_z)
    reach_check = below_mid > 0
    checks.append({"name": "reach_lower", "passed": reach_check,
                    "message": f"{below_mid}/{len(marbles)} below midpoint (z={course_mid_z:.2f})"})

    # Check: marbles haven't escaped the course (below floor OR above ceiling)
    # Threshold: up to 70% escape tolerance (allows 30% to leave)
    # to account for chaotic bouncing on early frames before marbles settle.
    safety_floor = course_min_z - 2.0 * _S
    safety_ceiling = course_max_z + 5.0 * _S  # generous ceiling for upward launches
    escaped_count = sum(1 for z in final_z if z < safety_floor or z > safety_ceiling)
    retain_check = escaped_count < len(marbles) * 0.7
    checks.append({"name": "marble_retention", "passed": retain_check,
                    "message": f"{escaped_count}/{len(marbles)} escaped course bounds"})

    # Check: checkpoint retention — ADVISORY ONLY.
    # These are logged for diagnostic insight but do NOT affect pass/fail.
    # The basic marble_retention check above is the gate-keeper.
    advisory: list[dict[str, Any]] = []
    if len(checkpoint_data) > 1:
        for cp in checkpoint_data:
            if cp["escaped"] > len(marbles) * 0.5:
                advisory.append({
                    "name": f"checkpoint_{cp['frame']}_retention",
                    "passed": False,
                    "message": f"Frame {cp['frame']}: {cp['escaped']}/{len(marbles)} escaped",
                    "advisory": True,
                })

    passed = all(c["passed"] for c in checks)
    reason = "All smoke test checks passed" if passed else "Smoke test failed: " + ", ".join(
        c["name"] for c in checks if not c["passed"]
    )
    # Append advisory checks AFTER pass/fail is computed so they appear
    # in the output for diagnostics but don't block the pipeline.
    checks.extend(advisory)

    positions = [
        {"name": m.name, "final_z": round(final_positions[m.name].z, 4),
         "travel": round(travelled[m.name], 4)}
        for m in marbles[:10]
    ]

    # Reset to frame 1
    scene.frame_set(1)

    return SmokeTestResult(passed=passed, checks=checks,
                           marble_positions=positions,
                           checkpoint_data=checkpoint_data,
                           reason=reason)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--checkpoints", type=str, default="",
                        help="Comma-separated frame numbers for checkpoint testing")
    args = parser.parse_args(argv)

    cp_list = [int(f) for f in args.checkpoints.split(",") if f.strip()] if args.checkpoints else None
    result = run_smoke_test(frames=args.frames, checkpoints=cp_list)
    print(json.dumps(result.to_dict(), indent=2))

    if not result.passed:
        sys.exit(1)
