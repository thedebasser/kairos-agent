"""Smoke test for domino course physics.

Bakes a short segment (default 300 frames) and checks that dominoes
actually topple in a cascading chain.

Run via:
    blender --background <file.blend> --python smoke_test_domino.py [-- --frames 300]
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]


def _bake_partial(frame_count: int) -> None:
    """Bake rigid body simulation for a limited number of frames.

    Uses frame-stepping (scene.frame_set) which is the most reliable
    method in Blender 5.x background mode. The rigid body world
    evaluates automatically at each frame step.
    """
    scene = bpy.context.scene
    rbw = scene.rigidbody_world
    if rbw is None:
        return

    end_frame = min(scene.frame_start + frame_count, scene.frame_end)

    # First try the proper bake API (Blender 5.x temp_override)
    original_end = rbw.point_cache.frame_end
    rbw.point_cache.frame_end = end_frame
    try:
        override = bpy.context.copy()
        override["point_cache"] = rbw.point_cache
        with bpy.context.temp_override(**override):
            bpy.ops.ptcache.free_bake_all()
            bpy.ops.ptcache.bake_all(bake=True)
        print(f"[smoke] Baked via ptcache API up to frame {end_frame}")
    except Exception as exc:
        # Fallback: step frame-by-frame to evaluate physics
        print(f"[smoke] ptcache bake failed ({exc}), using frame-stepping...")
        scene.frame_set(scene.frame_start)
        for f in range(scene.frame_start, end_frame + 1):
            scene.frame_set(f)
        print(f"[smoke] Frame-stepped to frame {end_frame}")
    finally:
        rbw.point_cache.frame_end = original_end


def _count_fallen(dominos: list[bpy.types.Object], threshold_deg: float = 30.0) -> int:
    """Count how many dominos have tilted beyond threshold from upright.

    In Blender 5.x, after a ptcache bake, the evaluated rigid-body
    transforms live on the depsgraph copy, accessible via
    ``obj.evaluated_get(depsgraph).matrix_world``.  The base
    ``obj.rotation_euler`` stays at the rest value, so we MUST read
    from matrix_world instead.
    """
    threshold_rad = math.radians(threshold_deg)
    dg = bpy.context.evaluated_depsgraph_get()
    fallen = 0
    for d in dominos:
        d_eval = d.evaluated_get(dg)
        mw = d_eval.matrix_world

        # Extract the local Z axis from the world matrix.
        # An upright domino has local-Z ≈ world-Z = (0, 0, 1).
        local_z = mw.to_3x3() @ Vector((0, 0, 1))
        # Angle between local-Z and world-Z
        tilt = local_z.angle(Vector((0, 0, 1)), 0.0)
        if tilt > threshold_rad:
            fallen += 1
    return fallen


def _get_per_domino_data(dominos: list[bpy.types.Object]) -> list[dict[str, Any]]:
    """Collect per-domino physics data for calibration diagnostics.

    Returns a list of dicts with position, tilt, and velocity info
    for each domino at the current frame.
    """
    dg = bpy.context.evaluated_depsgraph_get()
    data: list[dict[str, Any]] = []
    for d in dominos:
        d_eval = d.evaluated_get(dg)
        mw = d_eval.matrix_world
        local_z = mw.to_3x3() @ Vector((0, 0, 1))
        tilt_rad = local_z.angle(Vector((0, 0, 1)), 0.0)
        pos = mw.translation
        data.append({
            "name": d.name,
            "position": [round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)],
            "tilt_degrees": round(math.degrees(tilt_rad), 2),
            "fallen": tilt_rad > math.radians(30.0),
        })
    return data


def _detect_failure_modes(
    dominos: list[bpy.types.Object],
    per_domino: list[dict[str, Any]],
    completion_ratio: float,
) -> list[dict[str, str]]:
    """Detect calibration-relevant failure modes from simulation state.

    Returns a list of {mode, detail} dicts. Modes match the FailureMode
    enum in kairos.calibration.models.
    """
    failures: list[dict[str, str]] = []
    dg = bpy.context.evaluated_depsgraph_get()

    # Chain break detection: find gaps in the topple sequence
    fallen_flags = [d["fallen"] for d in per_domino]
    if fallen_flags:
        # Find first non-fallen after a fallen sequence
        saw_fallen = False
        for i, f in enumerate(fallen_flags):
            if f:
                saw_fallen = True
            elif saw_fallen:
                failures.append({
                    "mode": "chain_break",
                    "detail": f"Chain break at domino {i} ({per_domino[i]['name']})",
                })
                break

    # Incomplete propagation
    if 0.0 < completion_ratio < 0.90:
        failures.append({
            "mode": "incomplete_propagation",
            "detail": f"Only {completion_ratio:.0%} of dominoes fell",
        })

    # Trigger miss: first domino didn't fall
    if fallen_flags and not fallen_flags[0]:
        failures.append({
            "mode": "trigger_miss",
            "detail": "First domino did not fall — trigger may have missed",
        })

    # Physics anomalies: floating or explosion
    for d_data in per_domino:
        z = d_data["position"][2]
        if z > 20:
            failures.append({
                "mode": "explosion",
                "detail": f"{d_data['name']} at z={z} — likely explosion",
            })
            break
        if z < -5:
            failures.append({
                "mode": "floating",
                "detail": f"{d_data['name']} at z={z} — fell through ground",
            })
            break

    return failures


def run_smoke_test(frame_count: int = 0, calibration_mode: bool = False) -> dict[str, Any]:
    """Run a partial physics bake and check domino toppling.

    If *frame_count* is 0 (default), it is auto-computed from the
    number of dominos so the chain has enough time to propagate fully.
    """
    scene = bpy.context.scene
    checks = []

    # Gather domino objects
    dominos = sorted(
        [o for o in bpy.data.objects if o.name.startswith("Domino_")],
        key=lambda o: o.name,
    )
    total = len(dominos)

    if total == 0:
        return {
            "passed": False,
            "reason": "No domino objects found",
            "checks": [{"name": "dominos_exist", "passed": False,
                         "message": "No Domino_ objects in scene"}],
        }

    # Auto-compute frame count if not provided.
    # Chain propagates at ~5 dominos/sec at 30 fps → ~6 frames per domino.
    # Add generous 50% margin + trigger warm-up.
    if frame_count <= 0:
        fps = scene.render.fps or 30
        frame_count = max(300, int(total * 9) + 60)  # ~9 frames/domino + 60 for trigger
        frame_count = min(frame_count, scene.frame_end)
    print(f"Smoke test: {total} dominos, baking {frame_count} frames")
    # Record initial positions (XY for explosion detection, Z for reference)
    scene.frame_set(1)
    initial_pos = {d.name: (d.location.x, d.location.y, d.location.z) for d in dominos}
    initial_z = {k: v[2] for k, v in initial_pos.items()}  # legacy ref

    # Bake partial
    print(f"Baking {frame_count} frames for smoke test...")
    _bake_partial(frame_count)

    # Check at the end of baked range
    check_frame = min(scene.frame_start + frame_count, scene.frame_end)
    scene.frame_set(check_frame)

    fallen = _count_fallen(dominos)
    ratio = fallen / total if total > 0 else 0

    # Check 1: At least some dominos fell (trigger worked)
    trigger_worked = fallen > 0
    checks.append({
        "name": "trigger_works",
        "passed": trigger_worked,
        "message": f"{fallen}/{total} fell by frame {check_frame}" if trigger_worked
                   else f"No dominos fell by frame {check_frame} — trigger may have missed",
    })

    # Check 2: Chain propagation (at least 90% fell in partial bake)
    chain_ok = ratio >= 0.90
    checks.append({
        "name": "chain_propagation",
        "passed": chain_ok,
        "message": f"Chain propagation: {ratio:.0%} fell"
                   + ("" if chain_ok else " (need >=90%)"),
    })

    # Check 3: Physics stability — no dominos flew off or launched laterally.
    # An explosion can scatter dominoes in XY without going high in Z, so we
    # check both vertical flyaway (z out of bounds) AND lateral launch
    # (XY displacement > 4× domino height from starting position).
    flew_away = 0
    launched = 0
    dg = bpy.context.evaluated_depsgraph_get()
    for d in dominos:
        d_eval = d.evaluated_get(dg)
        pos = d_eval.matrix_world.translation
        # Vertical flyaway
        if pos.z > 20 or pos.z < -5:
            flew_away += 1
            continue
        # Lateral explosion: displacement > 4× domino height from start
        # A legitimately toppled domino moves at most ~h/2 in XY.
        ix, iy, _ = initial_pos[d.name]
        xy_disp = math.sqrt((pos.x - ix) ** 2 + (pos.y - iy) ** 2)
        domino_h = max(d.dimensions) if max(d.dimensions) > 0 else 4.0
        if xy_disp > domino_h * 4.0:
            launched += 1
    anomaly_count = flew_away + launched
    stability_ok = anomaly_count <= max(1, int(total * 0.05))
    checks.append({
        "name": "physics_stability",
        "passed": stability_ok,
        "message": (
            f"{flew_away} flew off vertically, {launched} launched laterally — likely explosion"
            if not stability_ok
            else f"Physics stable — no flyaways or lateral launches"
        ),
    })

    all_passed = all(c["passed"] for c in checks)

    result: dict[str, Any] = {
        "passed": all_passed,
        "reason": "All smoke checks passed" if all_passed else "Some checks failed",
        "checks": checks,
        "fallen_count": fallen,
        "total_count": total,
        "completion_ratio": round(ratio, 3),
        "physics_anomalies": anomaly_count,
    }

    # Extended calibration diagnostics
    if calibration_mode:
        per_domino = _get_per_domino_data(dominos)
        failure_modes = _detect_failure_modes(dominos, per_domino, ratio)
        result["calibration"] = {
            "per_domino": per_domino,
            "failure_modes": failure_modes,
            "physics_anomalies": anomaly_count,
        }

    return result


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    frame_count = 0  # 0 = auto-compute based on domino count
    calibration_mode = False
    for i, arg in enumerate(argv):
        if arg == "--frames" and i + 1 < len(argv):
            frame_count = int(argv[i + 1])
        if arg == "--calibration-mode":
            calibration_mode = True

    result = run_smoke_test(frame_count, calibration_mode=calibration_mode)
    print(json.dumps(result, indent=2))

    if not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
