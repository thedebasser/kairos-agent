"""Physics-aware tracking camera for domino chains.

After the rigid-body simulation is fully baked, this script:
  1. Scans every frame to detect each domino's tip frame via depsgraph
  2. Computes the wave-front centroid + tangent per frame
  3. Derives camera position (side-on, behind, elevated)
  4. Gaussian-smooths all position/rotation channels
  5. Adds a pull-back to a wide overview shot when the chain completes
  6. Bulk-inserts keyframes using the low-level F-curve API

Run via:
    blender --background <baked_file.blend> --python physics_camera.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector, Euler, Quaternion  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILT_STANDING_DEG = 15.0     # >15° = tipped (avoids rigid body settling jitter)
TILT_FALLEN_DEG = 80.0       # >80° = fully fallen
WAVE_WINDOW_FRAMES = 5       # ±frames around current to define wave front
SMOOTH_SIGMA = 10.0          # Gaussian smoothing sigma (frames) — lower = tracks curves tighter
KEY_INTERVAL = 5             # insert a keyframe every N frames
PULLBACK_FRAMES = 90         # 3 seconds at 30fps for the pullback transition
HOLD_FRAMES = 30             # hold on the wide shot at the end


# ---------------------------------------------------------------------------
# Pass 1: Scan baked physics — find each domino's tip frame
# ---------------------------------------------------------------------------

def _get_tilt_degrees(obj_eval: bpy.types.Object) -> float:
    """Return tilt angle in degrees from upright for evaluated object."""
    mat = obj_eval.matrix_world
    local_z = Vector((mat[0][2], mat[1][2], mat[2][2])).normalized()
    return math.degrees(local_z.angle(Vector((0, 0, 1))))


def scan_tip_frames(
    dominos: list[bpy.types.Object],
    frame_start: int,
    frame_end: int,
) -> dict[str, int]:
    """Scan baked physics to find the first frame each domino tips past threshold.

    Returns {domino_name: tip_frame}. Dominos that never tip are omitted.
    """
    scene = bpy.context.scene
    tip_frames: dict[str, int] = {}
    remaining = set(obj.name for obj in dominos)

    for frame in range(frame_start, frame_end + 1):
        if not remaining:
            break
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()

        for obj in dominos:
            if obj.name not in remaining:
                continue
            obj_eval = obj.evaluated_get(depsgraph)
            tilt = _get_tilt_degrees(obj_eval)
            if tilt > TILT_STANDING_DEG:
                tip_frames[obj.name] = frame
                remaining.discard(obj.name)

    print(f"[physics_camera] Scanned {frame_end - frame_start + 1} frames, "
          f"{len(tip_frames)}/{len(dominos)} dominos tipped")
    return tip_frames


# ---------------------------------------------------------------------------
# Pass 2: Compute wave-front position + tangent per frame
# ---------------------------------------------------------------------------

def compute_wave_front_per_frame(
    dominos: list[bpy.types.Object],
    tip_frames: dict[str, int],
    frame_start: int,
    frame_end: int,
) -> list[dict[str, Any]]:
    """For each frame, compute wave-front centroid and travel tangent.

    The wave front at frame F is the average position of dominos whose
    tip_frame falls within [F - WAVE_WINDOW, F + WAVE_WINDOW].
    The tangent is the displacement between the previous and current centroid.

    Returns a list of dicts with keys: frame, pos (Vector), tangent (Vector).
    """
    scene = bpy.context.scene

    # Build lookup: domino name -> rest position (XY from initial placement)
    domino_positions: dict[str, Vector] = {}
    scene.frame_set(frame_start)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in dominos:
        obj_eval = obj.evaluated_get(depsgraph)
        domino_positions[obj.name] = obj_eval.matrix_world.translation.copy()

    # Pre-sort dominos by tip frame for efficient window queries
    tipped_list = sorted(tip_frames.items(), key=lambda x: x[1])
    if not tipped_list:
        return []

    first_tip = tipped_list[0][1]
    last_tip = tipped_list[-1][1]

    # Compute per-frame wave front centroid using the actual baked positions
    results: list[dict[str, Any]] = []
    prev_pos: Vector | None = None

    for frame in range(frame_start, frame_end + 1):
        # Find dominos tipping within the window
        window_start = frame - WAVE_WINDOW_FRAMES
        window_end = frame + WAVE_WINDOW_FRAMES

        wave_dominos = [
            name for name, tf in tipped_list
            if window_start <= tf <= window_end
        ]

        if not wave_dominos:
            # Before first tip or after last — use nearest
            if frame < first_tip:
                # Use the first domino that will tip
                wave_dominos = [tipped_list[0][0]]
            else:
                # After the last tip — use the last few
                wave_dominos = [tipped_list[-1][0]]

        # Get actual baked positions for these dominos at this frame
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()

        centroid = Vector((0, 0, 0))
        count = 0
        for name in wave_dominos:
            obj = bpy.data.objects.get(name)
            if obj:
                obj_eval = obj.evaluated_get(depsgraph)
                centroid += obj_eval.matrix_world.translation
                count += 1

        if count > 0:
            centroid /= count
        else:
            centroid = Vector((0, 0, 0))

        # Tangent = direction of travel (smoothed by using centroid diff)
        if prev_pos is not None:
            tangent = (centroid - prev_pos).copy()
            tangent.z = 0  # keep horizontal
            if tangent.length > 0.001:
                tangent.normalize()
            else:
                # Use previous tangent if stationary
                tangent = results[-1]["tangent"] if results else Vector((0, 1, 0))
        else:
            tangent = Vector((0, 1, 0))  # default forward

        results.append({
            "frame": frame,
            "pos": centroid.copy(),
            "tangent": tangent.copy(),
        })
        prev_pos = centroid.copy()

    # Second pass: smooth tangent using exponential weighted average
    smoothed_tangents: list[Vector] = []
    smooth_tan = results[0]["tangent"].copy()
    alpha = 0.10  # moderate alpha — tracks curve direction changes responsively
    for r in results:
        smooth_tan = smooth_tan.lerp(r["tangent"], alpha)
        if smooth_tan.length > 0.001:
            smooth_tan.normalize()
        smoothed_tangents.append(smooth_tan.copy())

    for i, r in enumerate(results):
        r["tangent"] = smoothed_tangents[i]

    return results


# ---------------------------------------------------------------------------
# Camera position computation
# ---------------------------------------------------------------------------

def compute_camera_positions(
    wave_data: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute raw camera position + look-at per frame from wave-front data.

    Camera is placed behind and to the side of the wave front, elevated above.
    Uses curvature-adaptive distance on tight curves.
    """
    S = config.get("world_scale", 10)

    cam_distance = config.get("cam_distance", 3.5) * S    # close to dominos
    cam_height = config.get("cam_height", 4.5) * S       # low, intimate height
    look_ahead_dist = config.get("look_ahead", 1.5) * S   # look slightly ahead of wave
    alpha_deg = config.get("cam_elevation_deg", 35)  # shallower angle
    beta_deg = config.get("cam_azimuth_deg", 20)     # tight side offset — stays aligned with dominos on curves

    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)

    cam_frames: list[dict[str, Any]] = []

    for i, wd in enumerate(wave_data):
        pos = wd["pos"]
        tangent = wd["tangent"]

        # Compute curvature (change in tangent direction)
        curvature_mult = 1.0
        if i >= 2:
            prev_tan = wave_data[i - 2]["tangent"]
            dot = max(-1.0, min(1.0, tangent.dot(prev_tan)))
            curvature = math.acos(dot) / max(1.0, 2.0)
            # On curves, pull in slightly closer instead of going wide
            curvature_mult = max(0.85, 1.0 - curvature * 0.3)

        D = cam_distance * curvature_mult

        # Right vector (perpendicular to tangent in horizontal plane)
        right = tangent.cross(Vector((0, 0, 1)))
        if right.length > 0.001:
            right.normalize()
        else:
            right = Vector((1, 0, 0))

        # Camera offset: behind and to the side, elevated
        horizontal_dir = (-tangent * math.cos(beta) + right * math.sin(beta))
        horizontal_dir.normalize()

        cam_pos = Vector((
            pos.x + horizontal_dir.x * math.cos(alpha) * D,
            pos.y + horizontal_dir.y * math.cos(alpha) * D,
            cam_height + (math.sin(alpha) * D * 0.3),  # subtle height variation
        ))

        # Look-at target: slightly ahead of wave front, at ground level
        look_target = Vector((
            pos.x + tangent.x * look_ahead_dist,
            pos.y + tangent.y * look_ahead_dist,
            pos.z * 0.5,  # look at half-height of dominos
        ))

        cam_frames.append({
            "frame": wd["frame"],
            "cam_pos": cam_pos,
            "look_target": look_target,
        })

    return cam_frames


# ---------------------------------------------------------------------------
# Pull-back to wide overview shot
# ---------------------------------------------------------------------------

def compute_pullback(
    cam_frames: list[dict[str, Any]],
    dominos: list[bpy.types.Object],
    tip_frames: dict[str, int],
    frame_end: int,
    fov_deg: float = 39.6,
) -> list[dict[str, Any]]:
    """Append a pull-back transition from tracking to wide overview.

    Starts when the last domino tips, eases out to a bounding-box view.
    """
    if not tip_frames or not cam_frames:
        return cam_frames

    last_tip_frame = max(tip_frames.values())

    # Compute overview position from bounding box of all dominos
    scene = bpy.context.scene
    scene.frame_set(frame_end)
    depsgraph = bpy.context.evaluated_depsgraph_get()

    all_positions: list[Vector] = []
    for obj in dominos:
        obj_eval = obj.evaluated_get(depsgraph)
        all_positions.append(obj_eval.matrix_world.translation.copy())

    if not all_positions:
        return cam_frames

    min_co = Vector((
        min(p.x for p in all_positions),
        min(p.y for p in all_positions),
        min(p.z for p in all_positions),
    ))
    max_co = Vector((
        max(p.x for p in all_positions),
        max(p.y for p in all_positions),
        max(p.z for p in all_positions),
    ))
    center = (min_co + max_co) / 2
    diagonal = (max_co - min_co).length
    margin = 1.4
    dist = (diagonal * margin / 2) / math.tan(math.radians(fov_deg / 2))

    # Overview camera position: slightly off-axis for visual interest
    overview_dir = Vector((0.3, -0.5, 0.7)).normalized()
    overview_pos = center + overview_dir * dist
    overview_target = center

    # Find transition start frame (when last domino tips)
    pullback_start = last_tip_frame
    pullback_end = min(pullback_start + PULLBACK_FRAMES, frame_end - HOLD_FRAMES)

    if pullback_start >= frame_end - HOLD_FRAMES:
        return cam_frames

    # Find the last tracking frame data
    last_tracking = cam_frames[-1] if cam_frames else None
    if last_tracking is None:
        return cam_frames

    # Cubic ease-in-out
    def ease_in_out(t: float) -> float:
        if t < 0.5:
            return 4 * t * t * t
        return 1 - (-2 * t + 2) ** 3 / 2

    # Replace/extend frames from pullback_start onward
    # First, trim cam_frames to before pullback
    trimmed = [cf for cf in cam_frames if cf["frame"] < pullback_start]
    tracking_pos = last_tracking["cam_pos"]
    tracking_target = last_tracking["look_target"]

    # Use the nearest tracking frame to pullback_start for the start position
    for cf in reversed(cam_frames):
        if cf["frame"] <= pullback_start:
            tracking_pos = cf["cam_pos"]
            tracking_target = cf["look_target"]
            break

    # Transition frames
    for frame in range(pullback_start, pullback_end + 1, KEY_INTERVAL):
        t = (frame - pullback_start) / max(1, pullback_end - pullback_start)
        t = max(0.0, min(1.0, t))
        eased = ease_in_out(t)

        cam_pos = tracking_pos.lerp(overview_pos, eased)
        look_target = tracking_target.lerp(overview_target, eased)

        trimmed.append({
            "frame": frame,
            "cam_pos": cam_pos.copy(),
            "look_target": look_target.copy(),
        })

    # Hold on the wide shot
    hold_start = pullback_end
    for frame in range(hold_start, frame_end + 1, KEY_INTERVAL):
        trimmed.append({
            "frame": frame,
            "cam_pos": overview_pos.copy(),
            "look_target": overview_target.copy(),
        })

    return trimmed


# ---------------------------------------------------------------------------
# Gaussian smoothing
# ---------------------------------------------------------------------------

def _gaussian_kernel(sigma: float) -> list[float]:
    """Create a 1D Gaussian kernel."""
    size = int(6 * sigma) | 1  # ensure odd
    half = size // 2
    kernel = [math.exp(-((i - half) ** 2) / (2 * sigma ** 2)) for i in range(size)]
    total = sum(kernel)
    return [k / total for k in kernel]


def _smooth_channel(values: list[float], sigma: float) -> list[float]:
    """Apply Gaussian smoothing to a 1D channel."""
    if sigma <= 0 or len(values) < 3:
        return values
    kernel = _gaussian_kernel(sigma)
    half = len(kernel) // 2
    n = len(values)
    result = []
    for i in range(n):
        acc = 0.0
        w_sum = 0.0
        for j, k in enumerate(kernel):
            idx = i + j - half
            if 0 <= idx < n:
                acc += values[idx] * k
                w_sum += k
        result.append(acc / w_sum if w_sum > 0 else values[i])
    return result


def smooth_camera_data(
    cam_frames: list[dict[str, Any]],
    sigma: float = SMOOTH_SIGMA,
) -> list[dict[str, Any]]:
    """Gaussian-smooth camera position and look-target channels."""
    if len(cam_frames) < 3:
        return cam_frames

    # Extract channels
    pos_x = [cf["cam_pos"].x for cf in cam_frames]
    pos_y = [cf["cam_pos"].y for cf in cam_frames]
    pos_z = [cf["cam_pos"].z for cf in cam_frames]
    tgt_x = [cf["look_target"].x for cf in cam_frames]
    tgt_y = [cf["look_target"].y for cf in cam_frames]
    tgt_z = [cf["look_target"].z for cf in cam_frames]

    # Smooth
    pos_x = _smooth_channel(pos_x, sigma)
    pos_y = _smooth_channel(pos_y, sigma)
    pos_z = _smooth_channel(pos_z, sigma)
    tgt_x = _smooth_channel(tgt_x, sigma)
    tgt_y = _smooth_channel(tgt_y, sigma)
    tgt_z = _smooth_channel(tgt_z, sigma)

    # Reassemble
    for i, cf in enumerate(cam_frames):
        cf["cam_pos"] = Vector((pos_x[i], pos_y[i], pos_z[i]))
        cf["look_target"] = Vector((tgt_x[i], tgt_y[i], tgt_z[i]))

    return cam_frames


# ---------------------------------------------------------------------------
# Rotation computation with quaternion SLERP smoothing
# ---------------------------------------------------------------------------

def compute_rotations(
    cam_frames: list[dict[str, Any]],
    slerp_factor: float = 0.15,
) -> list[dict[str, Any]]:
    """Compute look-at rotation per frame with quaternion SLERP smoothing."""
    prev_quat: Quaternion | None = None

    for cf in cam_frames:
        direction = cf["look_target"] - cf["cam_pos"]
        if direction.length < 0.001:
            direction = Vector((0, 0, -1))

        # Blender camera looks down -Z with Y up
        quat = direction.to_track_quat("-Z", "Y")

        if prev_quat is not None:
            # SLERP for smooth transitions
            quat = prev_quat.slerp(quat, slerp_factor)

        cf["rotation"] = quat.to_euler("XYZ")
        cf["quat"] = quat
        prev_quat = quat

    return cam_frames


# ---------------------------------------------------------------------------
# Keyframe insertion using Blender 5.x compatible F-curve API
# ---------------------------------------------------------------------------

def _get_or_create_fcurves(
    cam_obj: bpy.types.Object,
) -> dict[str, Any]:
    """Get or create F-curves for the camera's location and rotation.

    Handles Blender 5.x layered action API.
    Returns dict mapping data_path+index to fcurve objects.
    """
    # Remove existing animation data to start fresh
    if cam_obj.animation_data:
        cam_obj.animation_data_clear()

    # Insert a dummy keyframe to bootstrap the action structure
    cam_obj.keyframe_insert(data_path="location", index=0, frame=1)
    cam_obj.keyframe_insert(data_path="location", index=1, frame=1)
    cam_obj.keyframe_insert(data_path="location", index=2, frame=1)
    cam_obj.keyframe_insert(data_path="rotation_euler", index=0, frame=1)
    cam_obj.keyframe_insert(data_path="rotation_euler", index=1, frame=1)
    cam_obj.keyframe_insert(data_path="rotation_euler", index=2, frame=1)

    action = cam_obj.animation_data.action

    # Access fcurves via Blender 5.x layered API
    fcurves = []
    if hasattr(action, "layers") and action.layers:
        try:
            strip = action.layers[0].strips[0]
            if strip.channelbags:
                fcurves = list(strip.channelbags[0].fcurves)
        except (IndexError, AttributeError):
            pass

    # Legacy fallback
    if not fcurves and hasattr(action, "fcurves"):
        fcurves = list(action.fcurves)

    # Build lookup: (data_path, index) -> fcurve
    fcu_map: dict[tuple[str, int], Any] = {}
    for fc in fcurves:
        fcu_map[(fc.data_path, fc.array_index)] = fc

    return fcu_map


def insert_camera_keyframes(
    cam_obj: bpy.types.Object,
    cam_frames: list[dict[str, Any]],
) -> None:
    """Bulk-insert smoothed camera keyframes using the F-curve API."""
    if not cam_frames:
        return

    fcu_map = _get_or_create_fcurves(cam_obj)

    # Channels to keyframe
    channels = [
        ("location", 0),     # X
        ("location", 1),     # Y
        ("location", 2),     # Z
        ("rotation_euler", 0),  # X rotation
        ("rotation_euler", 1),  # Y rotation
        ("rotation_euler", 2),  # Z rotation
    ]

    def _get_value(cf: dict, dp: str, idx: int) -> float:
        if dp == "location":
            return [cf["cam_pos"].x, cf["cam_pos"].y, cf["cam_pos"].z][idx]
        else:
            return [cf["rotation"].x, cf["rotation"].y, cf["rotation"].z][idx]

    for dp, idx in channels:
        fc = fcu_map.get((dp, idx))
        if fc is None:
            print(f"[physics_camera] WARNING: No F-curve for {dp}[{idx}]")
            continue

        # Clear existing keyframes (from the bootstrap)
        while len(fc.keyframe_points) > 0:
            fc.keyframe_points.remove(fc.keyframe_points[0])

        # Bulk insert
        fc.keyframe_points.add(count=len(cam_frames))
        for j, cf in enumerate(cam_frames):
            val = _get_value(cf, dp, idx)
            fc.keyframe_points[j].co = (float(cf["frame"]), val)
            fc.keyframe_points[j].interpolation = "BEZIER"
            fc.keyframe_points[j].handle_left_type = "AUTO_CLAMPED"
            fc.keyframe_points[j].handle_right_type = "AUTO_CLAMPED"

        # Recalculate handles
        fc.update()

        # Enable continuous acceleration auto-smoothing if available
        try:
            fc.auto_smoothing = "CONT_ACCEL"
        except (AttributeError, TypeError):
            pass

    print(f"[physics_camera] Inserted {len(cam_frames)} keyframes across "
          f"{len(channels)} channels")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_physics_camera(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Main entry: compute and apply physics-aware camera to the baked scene.

    Call this AFTER rigid body bake is complete.
    """
    if config is None:
        config = {}

    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end

    # Find camera
    cam_obj = scene.camera
    if cam_obj is None:
        print("[physics_camera] No camera in scene, skipping")
        return {"status": "no_camera"}

    # Check if camera style is tracking
    camera_style = "tracking"  # default; could read from config
    if config.get("camera_style") and config["camera_style"] != "tracking":
        print(f"[physics_camera] Camera style is '{config['camera_style']}', "
              "skipping physics-based tracking")
        return {"status": "skipped", "camera_style": config["camera_style"]}

    # Collect domino objects (named "Domino_NNNN")
    dominos = sorted(
        [obj for obj in bpy.data.objects if obj.name.startswith("Domino_")],
        key=lambda o: o.name,
    )
    if len(dominos) < 4:
        print(f"[physics_camera] Only {len(dominos)} dominos found, need ≥4")
        return {"status": "too_few_dominos", "count": len(dominos)}

    print(f"[physics_camera] Starting physics-aware camera for {len(dominos)} dominos, "
          f"frames {frame_start}-{frame_end}")

    # Pass 1: Scan tip frames
    tip_frames = scan_tip_frames(dominos, frame_start, frame_end)

    if len(tip_frames) < 4:
        print("[physics_camera] Too few dominos tipped, keeping existing camera")
        return {"status": "insufficient_tips", "tipped": len(tip_frames)}

    # Pass 2: Wave-front computation (sampled at KEY_INTERVAL)
    wave_data = compute_wave_front_per_frame(
        dominos, tip_frames, frame_start, frame_end,
    )

    # Subsample to key interval for efficiency
    wave_sampled = [wd for wd in wave_data if wd["frame"] % KEY_INTERVAL == 0
                    or wd["frame"] == frame_start or wd["frame"] == frame_end]

    # Compute camera positions
    cam_frames = compute_camera_positions(wave_sampled, config)

    # Add pull-back to wide shot
    cam_data = cam_obj.data
    fov_deg = math.degrees(2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens)))
    cam_frames = compute_pullback(cam_frames, dominos, tip_frames, frame_end, fov_deg)

    # Gaussian smooth all channels
    cam_frames = smooth_camera_data(cam_frames, sigma=SMOOTH_SIGMA)

    # Compute rotations with SLERP
    cam_frames = compute_rotations(cam_frames, slerp_factor=0.12)  # responsive rotation tracking

    # Configure camera lens for tracking shot
    cam_data.lens = 40  # tighter for following shot

    # Insert keyframes
    insert_camera_keyframes(cam_obj, cam_frames)

    first_tip = min(tip_frames.values()) if tip_frames else frame_start
    last_tip = max(tip_frames.values()) if tip_frames else frame_end

    result = {
        "status": "ok",
        "dominos_scanned": len(dominos),
        "dominos_tipped": len(tip_frames),
        "first_tip_frame": first_tip,
        "last_tip_frame": last_tip,
        "keyframes_inserted": len(cam_frames),
        "pullback_start": last_tip,
        "smooth_sigma": SMOOTH_SIGMA,
    }
    print(f"[physics_camera] Done: {len(tip_frames)}/{len(dominos)} tipped, "
          f"first={first_tip} last={last_tip}, {len(cam_frames)} keyframes")
    return result


# ---------------------------------------------------------------------------
# CLI entry point (for standalone testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser(description="Apply physics-aware camera")
    parser.add_argument("--config", default=None, help="Path to config JSON")
    args = parser.parse_args(argv)

    config = {}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    result = apply_physics_camera(config)
    print(json.dumps(result, indent=2))

    # Save the .blend with updated camera
    bpy.ops.wm.save_mainfile()
    print("[physics_camera] Saved .blend with physics-aware camera")
