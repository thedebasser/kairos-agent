"""Generate a domino run course in Blender.

Places dominoes along a procedural curve (spiral, S-curve, branching,
cascade) on a ground plane, sets up rigid body physics, camera, and
lighting for a 9:16 vertical short.

Run via:
    blender --background --python generate_domino_course.py -- --config <path.json>
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector, Matrix, Euler  # type: ignore[import-untyped]

# Ensure blend/scripts is on the path so we can import siblings
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORLD_SCALE: int = 10  # Bullet needs objects ~1 m+ for reliable collision


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
PALETTES: dict[str, list[str]] = {
    "rainbow": ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"],
    "pastel": ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E8BAFF"],
    "neon": ["#FF1493", "#00FF7F", "#FFD700", "#00BFFF", "#FF4500", "#7FFF00"],
    "ocean": ["#006994", "#40A4DF", "#8ED1FC", "#B8E6FF", "#004E6A"],
    "sunset": ["#FF6B6B", "#FFA07A", "#FFD700", "#FF4500", "#DC143C", "#FF1493"],
    "monochrome": ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333"],
    "earth": ["#8B4513", "#D2691E", "#DEB887", "#F5DEB3", "#556B2F"],
}


def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    """Convert '#RRGGBB' to (r, g, b) in 0-1 range."""
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def _clear_scene() -> None:
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def _setup_rigid_body_world(config: dict[str, Any]) -> None:
    """Create / configure the rigid-body world."""
    scene = bpy.context.scene
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    rbw = scene.rigidbody_world
    rbw.substeps_per_frame = config.get("substeps_per_frame", 20)
    rbw.solver_iterations = config.get("solver_iterations", 20)
    fps = config.get("fps", 30)
    duration_sec = config.get("duration_sec", 65)
    duration_frames = fps * duration_sec
    scene.frame_start = 1
    scene.frame_end = duration_frames
    scene.render.fps = fps
    rbw.point_cache.frame_start = 1
    rbw.point_cache.frame_end = duration_frames
    scene.gravity = (0, 0, -9.81)


# ---------------------------------------------------------------------------
# Path generation — all paths on the XY ground plane (Z=0)
# ---------------------------------------------------------------------------

def _generate_spiral_path(
    config: dict[str, Any],
) -> list[tuple[float, float]]:
    """Generate a spiral path of (x, y) positions."""
    count = config.get("domino_count", 150)
    turns = config.get("spiral_turns", 2.0)
    # Scale radius with count — more dominos need a bigger spiral
    base_radius = 3.0 * WORLD_SCALE
    max_radius = base_radius * max(1.0, count / 150.0)
    min_radius = 0.5 * WORLD_SCALE
    positions = []
    for i in range(count):
        t = i / max(count - 1, 1)
        angle = t * turns * 2 * math.pi
        radius = min_radius + t * (max_radius - min_radius)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions.append((x, y))
    return positions


def _generate_s_curve_path(
    config: dict[str, Any],
) -> list[tuple[float, float]]:
    """Generate an S-curve path of (x, y) positions."""
    count = config.get("domino_count", 150)
    # Keep amplitude gentle enough that the minimum radius of curvature
    # R_min = L² / (A * (2πC)²) stays well above the domino spacing.
    amplitude = config.get("path_amplitude", 1.0) * WORLD_SCALE
    cycles = config.get("path_cycles", 1.0)
    # Scale the path length with domino count so longer chains
    # get a proportionally longer path (baseline: 60 BU for 50 dominos).
    base_extent = 6.0 * WORLD_SCALE          # 60 BU for ~50 dominos
    y_extent = base_extent * max(1.0, count / 50.0)
    positions = []
    for i in range(count):
        t = i / max(count - 1, 1)
        y = -y_extent / 2 + t * y_extent
        x = amplitude * math.sin(2 * math.pi * cycles * t)
        positions.append((x, y))
    return positions


def _generate_branching_path(
    config: dict[str, Any],
) -> list[tuple[float, float]]:
    """Generate a trunk path that fans into branches.

    Layout: straight trunk → fan-out into N branches.
    All dominoes returned as a single ordered list (trunk first,
    then each branch sequentially). The first domino of each branch
    is close enough to the trunk tip that the toppling cascade propagates.
    """
    total_count = config.get("domino_count", 150)
    branch_count = config.get("branch_count", 3)
    trunk_fraction = 0.4  # 40% of dominoes in the trunk
    trunk_count = max(10, int(total_count * trunk_fraction))
    branch_dominos = (total_count - trunk_count) // branch_count

    # Trunk runs along +Y
    trunk_len = 3.0 * WORLD_SCALE
    positions: list[tuple[float, float]] = []
    for i in range(trunk_count):
        t = i / max(trunk_count - 1, 1)
        positions.append((0.0, t * trunk_len))

    # Fan out from trunk tip
    trunk_tip_y = trunk_len
    fan_angle_spread = math.pi * 0.6  # 108° total spread
    branch_len = 2.5 * WORLD_SCALE
    for b in range(branch_count):
        angle = -fan_angle_spread / 2 + b * fan_angle_spread / max(branch_count - 1, 1)
        dx = math.sin(angle)
        dy = math.cos(angle)
        for i in range(branch_dominos):
            t = (i + 1) / branch_dominos
            x = t * branch_len * dx
            y = trunk_tip_y + t * branch_len * dy
            positions.append((x, y))

    return positions


def _smooth_path_corners(
    positions: list[tuple[float, float]],
    min_radius: float | None = None,
    iterations: int = 3,
) -> list[tuple[float, float]]:
    """Smooth sharp corners in a path using Chaikin's corner-cutting algorithm.

    After *iterations* rounds of subdivision the sharpest corners are replaced
    with smooth arcs.  If *min_radius* is provided, only corners whose
    inscribed circle would be smaller than *min_radius* are subdivided (all
    corners when None).
    """
    if len(positions) < 3:
        return positions

    for _ in range(iterations):
        smoothed: list[tuple[float, float]] = [positions[0]]
        for i in range(1, len(positions) - 1):
            p0 = positions[i - 1]
            p1 = positions[i]
            p2 = positions[i + 1]

            # Check if corner is sharp enough to need smoothing
            if min_radius is not None:
                dx1 = p1[0] - p0[0]
                dy1 = p1[1] - p0[1]
                dx2 = p2[0] - p1[0]
                dy2 = p2[1] - p1[1]
                len1 = math.sqrt(dx1 * dx1 + dy1 * dy1) or 1e-9
                len2 = math.sqrt(dx2 * dx2 + dy2 * dy2) or 1e-9
                dot = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
                dot = max(-1.0, min(1.0, dot))
                angle = math.acos(dot)  # 0 = straight, π = reversal
                # Inscribed circle radius ≈ min(half-edge) * tan(angle/2)
                half_edge = min(len1, len2) * 0.5
                if angle > 0.01:
                    r = half_edge * math.tan((math.pi - angle) / 2)
                else:
                    r = float("inf")
                if r >= min_radius:
                    # Corner is already gentle enough
                    smoothed.append(p1)
                    continue

            # Chaikin: replace corner with two points at 25%/75%
            q0 = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            q1 = (0.25 * p1[0] + 0.75 * p2[0], 0.25 * p1[1] + 0.75 * p2[1])
            smoothed.append(q0)
            smoothed.append(q1)

        smoothed.append(positions[-1])
        positions = smoothed

    return positions


def _generate_cascade_path(
    config: dict[str, Any],
) -> list[tuple[float, float]]:
    """Generate a wide zigzag 'cascade' path with smooth U-turn transitions.

    Instead of sharp 90° corners at row ends, each transition uses a
    semicircular arc so the domino chain can propagate continuously.

    Dimensions scale with domino count so that U-turns stay gentle
    regardless of how many dominos are requested.
    """
    count = config.get("domino_count", 150)
    # Scale dimensions with count so the path is long enough without
    # relying on Step 1c's uniform scale-up (which squishes U-turns).
    scale = max(1.0, count / 100.0)
    width = 5.0 * WORLD_SCALE * scale
    height = 6.0 * WORLD_SCALE * scale
    rows = 5
    row_spacing = height / max(rows - 1, 1)
    # U-turn radius — true semicircle, no horizontal squish
    turn_radius = row_spacing / 2
    turn_points = 20  # more points for smoother arcs (was 12)

    positions: list[tuple[float, float]] = []
    # Distribute dominos across rows + turns
    turn_count = rows - 1
    turn_total_points = turn_count * turn_points
    row_total_points = count - turn_total_points
    per_row = max(4, row_total_points // rows)

    for row in range(rows):
        t_y = row / max(rows - 1, 1)
        y = -height / 2 + t_y * height
        going_right = (row % 2 == 0)

        # Straight section across the row
        for i in range(per_row):
            t_x = i / max(per_row - 1, 1)
            if not going_right:
                t_x = 1.0 - t_x
            x = -width / 2 + t_x * width
            positions.append((x, y))

        # Semicircular U-turn to next row (skip after last row)
        if row < rows - 1:
            # End of current row
            end_x = positions[-1][0]
            end_y = y
            next_y = -height / 2 + (row + 1) / max(rows - 1, 1) * height
            radius = abs(next_y - end_y) / 2

            for j in range(1, turn_points + 1):
                t = j / (turn_points + 1)
                angle = math.pi * t  # 0 to π
                if going_right:
                    # True semicircle — no 0.6 squish factor
                    arc_x = end_x + radius * math.sin(angle)
                    arc_y = end_y + radius * (1 - math.cos(angle))
                else:
                    arc_x = end_x - radius * math.sin(angle)
                    arc_y = end_y + radius * (1 - math.cos(angle))
                positions.append((arc_x, arc_y))

    # Fill remaining dominoes
    remaining = count - len(positions)
    if remaining > 0 and positions:
        last_x, last_y = positions[-1]
        dx = 0.3 * WORLD_SCALE
        for i in range(remaining):
            last_y += dx * 0.1
            positions.append((last_x + (i + 1) * dx * 0.05, last_y))

    return positions


def _generate_word_path(
    config: dict[str, Any],
) -> list[tuple[float, float]]:
    """Generate a flowing multi-wave path (placeholder for word-spelling).

    True word spelling would need font contour sampling which is complex.
    For now, uses a multi-S-curve that looks dynamic on camera and
    scales its length to fit the requested domino count.
    """
    count = config.get("domino_count", 150)
    amplitude = 1.5 * WORLD_SCALE
    # Scale length to domino count (1.4 BU spacing → ~50 dominos per 70 BU)
    y_extent = max(6.0 * WORLD_SCALE, count / 50.0 * 6.0 * WORLD_SCALE)
    cycles = max(1.5, count / 100.0)  # more waves for more dominos
    positions = []
    for i in range(count):
        t = i / max(count - 1, 1)
        y = -y_extent / 2 + t * y_extent
        x = amplitude * math.sin(2 * math.pi * cycles * t)
        positions.append((x, y))
    return positions


PATH_GENERATORS = {
    "spiral": _generate_spiral_path,
    "s_curve": _generate_s_curve_path,
    "branching": _generate_branching_path,
    "word_spell": _generate_word_path,
    "cascade": _generate_cascade_path,
}


# ---------------------------------------------------------------------------
# Domino placement
# ---------------------------------------------------------------------------

def _compute_tangent_angle(
    positions: list[tuple[float, float]], index: int,
) -> float:
    """Compute the Z-rotation angle so the domino faces perpendicular to path."""
    if len(positions) < 2:
        return 0.0
    if index == 0:
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
    elif index == len(positions) - 1:
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
    else:
        dx = positions[index + 1][0] - positions[index - 1][0]
        dy = positions[index + 1][1] - positions[index - 1][1]
    tangent = math.atan2(dy, dx)
    # Domino's thin face should be perpendicular to path direction
    return tangent


def _resample_path(
    raw: list[tuple[float, float]],
    spacing: float,
) -> list[tuple[float, float]]:
    """Resample a path with curvature-adaptive spacing.

    On straights the base *spacing* is used.  On curves the gap is
    widened by the factor ``1 / cos(θ/2)`` where θ is the local
    turning angle, so that the face-to-face distance between adjacent
    dominos stays ≥ *spacing* regardless of curvature.

    Formula derivation:
        Two dominos on a curve of radius R separated by arc length s
        face each other across a chord of length ``2R·sin(s/(2R))``.
        For small θ = s/R this simplifies to ``s·cos(θ/2)``.  To keep
        that chord ≥ spacing we need ``s ≥ spacing / cos(θ/2)``.
    """
    if len(raw) < 2:
        return raw

    # Build cumulative arc-length
    cum = [0.0]
    for i in range(1, len(raw)):
        dx = raw[i][0] - raw[i - 1][0]
        dy = raw[i][1] - raw[i - 1][1]
        cum.append(cum[-1] + math.sqrt(dx * dx + dy * dy))

    total_len = cum[-1]
    if total_len < spacing:
        return raw

    # Helper: interpolate (x, y) at cumulative distance *d*.
    def _interp(d: float, hint: int = 0) -> tuple[tuple[float, float], int]:
        seg = hint
        while seg < len(cum) - 2 and cum[seg + 1] < d:
            seg += 1
        seg_len = cum[seg + 1] - cum[seg]
        if seg_len < 1e-9:
            return raw[seg], seg
        t = (d - cum[seg]) / seg_len
        x = raw[seg][0] + t * (raw[seg + 1][0] - raw[seg][0])
        y = raw[seg][1] + t * (raw[seg + 1][1] - raw[seg][1])
        return (x, y), seg

    # Pre-compute turning angles at each raw-path vertex so we can
    # quickly look up curvature at any arc-length position.
    vertex_angles = [0.0] * len(raw)
    for i in range(1, len(raw) - 1):
        dx1 = raw[i][0] - raw[i - 1][0]
        dy1 = raw[i][1] - raw[i - 1][1]
        dx2 = raw[i + 1][0] - raw[i][0]
        dy2 = raw[i + 1][1] - raw[i][1]
        L1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        L2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        if L1 > 1e-9 and L2 > 1e-9:
            dot = (dx1 * dx2 + dy1 * dy2) / (L1 * L2)
            dot = max(-1.0, min(1.0, dot))
            vertex_angles[i] = math.acos(dot)  # 0 = straight, π = reversal

    def _curvature_at(d: float, seg: int) -> float:
        """Interpolate turning angle at arc-length *d*."""
        # Find the two nearest vertices bracketing *d*
        while seg < len(cum) - 2 and cum[seg + 1] < d:
            seg += 1
        seg_len = cum[seg + 1] - cum[seg]
        if seg_len < 1e-9:
            return vertex_angles[seg]
        t = (d - cum[seg]) / seg_len
        a0 = vertex_angles[seg]
        a1 = vertex_angles[min(seg + 1, len(vertex_angles) - 1)]
        return a0 + t * (a1 - a0)

    # Walk the path with adaptive step size
    resampled: list[tuple[float, float]] = []
    pt, seg = _interp(0.0)
    resampled.append(pt)
    current_d = 0.0

    while current_d < total_len - spacing * 0.5:
        theta = _curvature_at(current_d, seg)
        cos_half = math.cos(theta / 2.0)
        # Clamp multiplier to avoid infinite spacing on hairpin turns
        if cos_half > 0.3:
            step = spacing / cos_half
        else:
            step = spacing * 3.3
        current_d += step
        if current_d > total_len:
            break
        pt, seg = _interp(current_d, seg)
        resampled.append(pt)

    # Always include the endpoint
    if len(resampled) > 1:
        last = resampled[-1]
        end = raw[-1]
        dx = end[0] - last[0]
        dy = end[1] - last[1]
        if math.sqrt(dx * dx + dy * dy) > spacing * 0.3:
            resampled.append(end)

    return resampled


def _apply_material(
    obj: bpy.types.Object,
    colour_hex: str = "#808080",
    metallic: float = 0.1,
    roughness: float = 0.5,
    name: str | None = None,
) -> None:
    """Apply a principled BSDF material to an object."""
    mat_name = name or f"mat_{colour_hex}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            r, g, b = _hex_to_rgb(colour_hex)
            bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
            bsdf.inputs["Metallic"].default_value = metallic
            bsdf.inputs["Roughness"].default_value = roughness
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def _place_dominoes(
    config: dict[str, Any],
    positions: list[tuple[float, float]],
) -> list[bpy.types.Object]:
    """Place domino box objects at the given positions, standing upright."""
    S = WORLD_SCALE
    w = config.get("domino_width", 0.08) * S
    h = config.get("domino_height", 0.4) * S
    d = config.get("domino_depth", 0.06) * S
    mass = config.get("domino_mass", 0.3)
    friction = config.get("domino_friction", 0.6)
    bounce = config.get("domino_bounce", 0.1)

    palette_name = config.get("palette", "rainbow")
    colours = PALETTES.get(palette_name, PALETTES["rainbow"])

    # NOTE: primitive_cube_add(size=1) creates a 1×1×1 cube (verts ±0.5).
    # Setting scale=(a,b,c) and applying gives final dimensions (a,b,c).
    # Domino axes after Z-rotation to face the path:
    #   X (along path / tipping dir) = depth (thin)
    #   Y (perpendicular to path / visible face) = width
    #   Z = height
    # So we scale (X,Y,Z) = (d, w, h) — thin side faces the path.
    dominos: list[bpy.types.Object] = []
    for i, (px, py) in enumerate(positions):
        angle_z = _compute_tangent_angle(positions, i)

        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(px, py, h / 2),  # center at half-height → bottom at z=0
        )
        obj = bpy.context.active_object
        obj.name = f"Domino_{i:04d}"
        # X=depth (thin, along path), Y=width (face), Z=height
        obj.scale = (d, w, h)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.rotation_euler = Euler((0, 0, angle_z), "XYZ")

        # Material
        colour = colours[i % len(colours)]
        _apply_material(obj, colour, metallic=0.05, roughness=0.4,
                        name=f"domino_mat_{i % len(colours)}")

        # Rigid body — active (dynamic)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add(type="ACTIVE")
        rb = obj.rigid_body
        rb.mass = mass
        rb.friction = friction
        rb.restitution = bounce
        rb.collision_shape = "BOX"
        rb.collision_margin = 0.004 * S
        rb.use_margin = True

        dominos.append(obj)

    return dominos


# ---------------------------------------------------------------------------
# Ground plane
# ---------------------------------------------------------------------------

def _create_ground(
    config: dict[str, Any],
    ground_center: tuple[float, float] = (0.0, 0.0),
) -> bpy.types.Object:
    """Create a ground plane with passive rigid body, centered on the course."""
    S = WORLD_SCALE
    size = 100.0 * S  # ground must cover all paths with generous margin
    bpy.ops.mesh.primitive_plane_add(
        size=size, location=(ground_center[0], ground_center[1], 0),
    )
    ground = bpy.context.active_object
    ground.name = "Ground"

    _apply_material(ground, "#d4d4d8", metallic=0.0, roughness=0.8,
                    name="ground_mat")

    bpy.context.view_layer.objects.active = ground
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    rb = ground.rigid_body
    rb.friction = config.get("ground_friction", 0.8)
    rb.restitution = 0.0
    rb.collision_shape = "BOX"

    return ground


# ---------------------------------------------------------------------------
# Trigger — invisible pusher animated to topple the first domino
# ---------------------------------------------------------------------------

def _create_trigger(
    config: dict[str, Any],
    first_domino: bpy.types.Object,
    positions: list[tuple[float, float]],
) -> bpy.types.Object:
    """Trigger the domino chain by tilting the first domino.

    Primary method: Tilt the first domino ~15° toward the second domino
    so gravity naturally topples it. This is the most reliable trigger
    for Blender rigid body physics in background/bake mode.

    Secondary: Also create a small invisible kinematic pusher for belt-and-
    suspenders reliability — in case the tilt alone isn't enough.
    """
    S = WORLD_SCALE
    trigger_frame = config.get("trigger_frame", 60)

    # Direction: along path tangent at index 0 (toward second domino)
    if len(positions) >= 2:
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
    else:
        dx, dy = 1.0, 0.0
    length = math.sqrt(dx * dx + dy * dy) or 1.0
    dx /= length
    dy /= length

    h = config.get("domino_height", 0.4) * S

    # ── Primary trigger: tilt first domino toward path direction ──────
    # The tangent angle aligns the domino's X-axis with the path.
    # Tipping it forward means rotating around the LOCAL Y-axis
    # (perpendicular to the thin face). Since the domino already has
    # rotation_euler.z = tangent_angle, we add a small X rotation
    # that tilts it "forward" along the path.
    tilt_degrees = config.get("trigger_tilt_degrees", 8.0)
    tilt_rad = math.radians(tilt_degrees)

    # Current rotation is (0, 0, angle_z). We want to tilt forward
    # along the path direction. The domino's local X axis points along
    # the path tangent. Positive Y rotation (Ry matrix) tilts +Z toward
    # +X local, which after the Z rotation becomes path-forward.
    current_z = first_domino.rotation_euler.z
    first_domino.rotation_euler = Euler((0, tilt_rad, current_z), "XYZ")

    # Shift the domino slightly forward so its base stays on ground
    # when tilted (prevent foot from lifting off the ground plane)
    px, py = positions[0]
    shift = (h / 2) * math.sin(tilt_rad) * 0.3
    first_domino.location.x = px + dx * shift
    first_domino.location.y = py + dy * shift

    # ── Secondary trigger: gentle kinematic pusher ────────────────
    # Travels a very short distance and stops at the first domino
    # to give it a nudge rather than a shove.
    impulse_dist = config.get("trigger_impulse", 1.5) * 0.05 * S
    start_x = px - dx * impulse_dist * 2
    start_y = py - dy * impulse_dist * 2
    end_x = px - dx * impulse_dist * 0.1   # stop just before domino
    end_y = py - dy * impulse_dist * 0.1

    bpy.ops.mesh.primitive_cube_add(
        size=1,
        location=(start_x, start_y, h * 0.7),
    )
    pusher = bpy.context.active_object
    pusher.name = "Trigger_Pusher"
    pusher.scale = (0.05 * S, 0.05 * S, h * 0.2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    pusher.hide_render = True

    bpy.context.view_layer.objects.active = pusher
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    rb = pusher.rigid_body
    rb.kinematic = True
    rb.mass = 0.5
    rb.friction = 0.5
    rb.restitution = 0.0
    rb.collision_shape = "BOX"

    # Keyframe animation: stay still, then slide into first domino
    scene = bpy.context.scene
    wind_up_frame = max(1, trigger_frame - 5)

    scene.frame_set(1)
    pusher.location = (start_x, start_y, h * 0.7)
    pusher.keyframe_insert(data_path="location", frame=1)

    scene.frame_set(wind_up_frame)
    pusher.location = (start_x, start_y, h * 0.7)
    pusher.keyframe_insert(data_path="location", frame=wind_up_frame)

    scene.frame_set(trigger_frame)
    pusher.location = (end_x, end_y, h * 0.7)
    pusher.keyframe_insert(data_path="location", frame=trigger_frame)

    # Move pusher out of the way after triggering
    scene.frame_set(trigger_frame + 10)
    overshoot_x = end_x + dx * impulse_dist
    overshoot_y = end_y + dy * impulse_dist
    pusher.location = (overshoot_x, overshoot_y, h * 0.7 + 5 * S)
    pusher.keyframe_insert(data_path="location", frame=trigger_frame + 10)

    scene.frame_set(1)

    print(f"[trigger] First domino tilted {tilt_degrees}° toward path direction")
    print(f"[trigger] Pusher animated from frame {wind_up_frame} to {trigger_frame}")

    return pusher


# ---------------------------------------------------------------------------
# Finale object
# ---------------------------------------------------------------------------

def _create_finale(
    config: dict[str, Any],
    last_pos: tuple[float, float],
    positions: list[tuple[float, float]],
) -> bpy.types.Object | None:
    """Create a finale object at the end of the domino path."""
    finale_type = config.get("finale_type", "none")
    if finale_type == "none":
        return None

    S = WORLD_SCALE
    px, py = last_pos

    # Direction: last domino tangent (to place finale in path direction)
    if len(positions) >= 2:
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        dx /= length
        dy /= length
    else:
        dx, dy = 0.0, 1.0

    offset = 0.5 * S
    fx = px + dx * offset
    fy = py + dy * offset

    if finale_type == "tower":
        # Stack of small cubes
        tower_h = 1.0 * S
        cube_size = 0.15 * S
        layers = int(tower_h / cube_size)
        for layer in range(layers):
            z = cube_size / 2 + layer * cube_size
            bpy.ops.mesh.primitive_cube_add(
                size=cube_size,
                location=(fx, fy, z),
            )
            obj = bpy.context.active_object
            obj.name = f"Finale_Tower_{layer:02d}"
            _apply_material(obj, "#FFD700", metallic=0.3, roughness=0.3,
                            name="finale_gold")
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_add(type="ACTIVE")
            rb = obj.rigid_body
            rb.mass = 0.1
            rb.friction = 0.6
            rb.restitution = 0.1
        return bpy.context.active_object

    elif finale_type == "ball":
        radius = 0.3 * S
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius,
            location=(fx, fy, radius),
        )
        obj = bpy.context.active_object
        obj.name = "Finale_Ball"
        _apply_material(obj, "#FF4500", metallic=0.3, roughness=0.3,
                        name="finale_ball_mat")
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add(type="ACTIVE")
        rb = obj.rigid_body
        rb.mass = 2.0
        rb.friction = 0.5
        rb.restitution = 0.3
        rb.collision_shape = "SPHERE"
        return obj

    elif finale_type == "ramp":
        # A small ramp at the end
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(fx, fy, 0.15 * S),
        )
        obj = bpy.context.active_object
        obj.name = "Finale_Ramp"
        obj.scale = (0.5 * S, 0.3 * S, 0.15 * S)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # Tilt it
        angle = _compute_tangent_angle(positions, len(positions) - 1)
        obj.rotation_euler = Euler((0.3, 0, angle), "XYZ")
        _apply_material(obj, "#4169E1", metallic=0.2, roughness=0.4,
                        name="finale_ramp_mat")
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add(type="PASSIVE")
        rb = obj.rigid_body
        rb.friction = 0.6
        rb.restitution = 0.2
        return obj

    return None


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def _setup_camera(
    config: dict[str, Any],
    course_center: Vector,
    course_bounds: tuple[Vector, Vector],
    positions: list[tuple[float, float]] | None = None,
) -> bpy.types.Object:
    """Create and position the camera for 9:16 vertical output.

    When *positions* is provided and camera_style == "tracking", the camera
    is keyframed to follow the domino cascade along the path.  Otherwise it
    falls back to a static angled or top-down view.
    """
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_data.clip_end = max(500, 50 * WORLD_SCALE)
    cam_data.clip_start = 0.1

    scene = bpy.context.scene
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920

    bounds_min, bounds_max = course_bounds
    extent_x = bounds_max.x - bounds_min.x
    extent_y = bounds_max.y - bounds_min.y
    max_extent = max(extent_x, extent_y, 5.0 * WORLD_SCALE)

    camera_style = config.get("camera_style", "tracking")

    if camera_style == "top_down":
        # Straight down
        cam_obj.location = Vector((
            course_center.x,
            course_center.y,
            max_extent * 1.2,
        ))
        cam_obj.rotation_euler = Euler((0, 0, 0), "XYZ")
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = max_extent * 1.3

    elif camera_style == "tracking" and positions and len(positions) >= 4:
        # ── Tracking camera: follows the cascade wave front ──────────
        _setup_tracking_camera(config, cam_obj, cam_data, positions)

    else:
        # Static angled fallback
        cam_data.lens = 32
        tilt_deg = 30
        tilt_rad = math.radians(tilt_deg)
        h_half_fov = math.atan(cam_data.sensor_width / (2 * cam_data.lens))
        slant_dist = (max_extent * 1.15) / (2 * math.tan(h_half_fov))
        cam_height = slant_dist * math.cos(tilt_rad)
        cam_back = slant_dist * math.sin(tilt_rad)
        cam_obj.location = Vector((
            course_center.x,
            course_center.y - cam_back,
            cam_height,
        ))
        cam_obj.rotation_euler = Euler((tilt_rad, 0, 0), "XYZ")

    return cam_obj


def _setup_tracking_camera(
    config: dict[str, Any],
    cam_obj: bpy.types.Object,
    cam_data: bpy.types.Camera,
    positions: list[tuple[float, float]],
) -> None:
    """Set up a static placeholder camera for the tracking style.

    The actual physics-aware tracking keyframes are computed post-bake
    by ``physics_camera.py`` (called from ``bake_and_render.py``).
    This placeholder positions the camera looking at the start of
    the course so the scene is valid for smoke-testing.
    """
    S = WORLD_SCALE
    cam_data.lens = 40
    cam_data.clip_end = max(500, 50 * S)

    # Position: elevated + offset behind first domino
    first_x, first_y = positions[0]

    # Estimate tangent from first two dominos
    if len(positions) >= 2:
        dx = positions[1][0] - positions[0][0]
        dy = positions[1][1] - positions[0][1]
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        tx, ty = dx / length, dy / length
    else:
        tx, ty = 0.0, 1.0

    cam_height = 12.0 * S
    cam_back = 5.0 * S
    cam_obj.location = Vector((
        first_x - tx * cam_back,
        first_y - ty * cam_back,
        cam_height,
    ))

    # Look at a point slightly ahead of the start
    look_ahead = 4.0 * S
    target = Vector((
        first_x + tx * look_ahead,
        first_y + ty * look_ahead,
        0,
    ))
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler("XYZ")

    print(f"[camera] Tracking placeholder: height={cam_height:.0f} BU, "
          f"lens={cam_data.lens}mm — real keyframes applied post-bake")


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------

# Energy values must account for WORLD_SCALE distance (inverse-square).
# At S=10 distances are ~10x bigger -> energy needs ~S² = 100x more.
_E = WORLD_SCALE ** 2  # energy multiplier
LIGHTING_PRESETS = {
    "studio":   {"key": 800 * _E, "fill": 300 * _E, "rim": 500 * _E, "sun": 3.0, "world": 0.5},
    "dramatic": {"key": 1200 * _E, "fill": 100 * _E, "rim": 800 * _E, "sun": 2.0, "world": 0.2},
    "soft":     {"key": 500 * _E, "fill": 400 * _E, "rim": 300 * _E, "sun": 4.0, "world": 0.8},
    "soft_daylight": {"key": 600 * _E, "fill": 400 * _E, "rim": 400 * _E, "sun": 5.0, "world": 0.7},
}


def _setup_lighting(config: dict[str, Any], course_center: Vector) -> None:
    """Set up three-point lighting and world background."""
    preset_name = config.get("lighting_preset", "studio")
    preset = LIGHTING_PRESETS.get(preset_name, LIGHTING_PRESETS["studio"])
    S = WORLD_SCALE

    # Key light
    bpy.ops.object.light_add(type="AREA", location=(
        course_center.x + 5 * S,
        course_center.y - 3 * S,
        8 * S,
    ))
    key = bpy.context.active_object
    key.name = "Key_Light"
    key.data.energy = preset["key"]
    key.data.size = 3 * S
    key.rotation_euler = Euler((math.radians(60), 0, math.radians(30)), "XYZ")

    # Fill light
    bpy.ops.object.light_add(type="AREA", location=(
        course_center.x - 4 * S,
        course_center.y - 2 * S,
        6 * S,
    ))
    fill = bpy.context.active_object
    fill.name = "Fill_Light"
    fill.data.energy = preset["fill"]
    fill.data.size = 2 * S
    fill.rotation_euler = Euler((math.radians(50), 0, math.radians(-20)), "XYZ")

    # Rim light
    bpy.ops.object.light_add(type="AREA", location=(
        course_center.x,
        course_center.y + 4 * S,
        7 * S,
    ))
    rim = bpy.context.active_object
    rim.name = "Rim_Light"
    rim.data.energy = preset["rim"]
    rim.data.size = 2 * S
    rim.rotation_euler = Euler((math.radians(70), 0, math.radians(180)), "XYZ")

    # Sun light — does NOT attenuate with distance, provides
    # consistent base illumination for any WORLD_SCALE.
    bpy.ops.object.light_add(type="SUN", location=(
        course_center.x,
        course_center.y,
        10 * S,
    ))
    sun = bpy.context.active_object
    sun.name = "Sun_Light"
    sun.data.energy = preset.get("sun", 3.0)
    sun.rotation_euler = Euler((math.radians(50), 0, math.radians(20)), "XYZ")

    # World background — light neutral grey so scene is never black
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Strength"].default_value = preset["world"]
        bg.inputs["Color"].default_value = (0.35, 0.38, 0.45, 1.0)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_domino_course(config: dict[str, Any]) -> dict[str, Any]:
    """Main entry point: generate a complete domino course scene."""
    seed = config.get("seed", 42)
    random.seed(seed)

    archetype = config.get("archetype", "s_curve")
    generator = PATH_GENERATORS.get(archetype, _generate_s_curve_path)

    # Step 1: Generate raw path
    raw_positions = generator(config)

    # Step 1b: Smooth sharp corners on the RAW path (before resampling)
    # This must happen BEFORE resample so uniform spacing is preserved.
    domino_h = config.get("domino_height", 0.4) * WORLD_SCALE
    spacing = domino_h * config.get("spacing_ratio", 0.6)
    min_turn_radius = spacing * 5  # minimum curve radius = 5× spacing
    raw_positions = _smooth_path_corners(raw_positions, min_radius=min_turn_radius, iterations=2)

    # Step 1c: Scale path if too short for the requested domino count.
    # Compute arc length of the raw path — if it can't fit enough dominos
    # at the configured spacing, uniformly scale the path up.
    target_count = config.get("domino_count", 150)
    needed_length = target_count * spacing * 1.05  # 5% margin

    arc_length = 0.0
    for i in range(1, len(raw_positions)):
        dx = raw_positions[i][0] - raw_positions[i - 1][0]
        dy = raw_positions[i][1] - raw_positions[i - 1][1]
        arc_length += math.sqrt(dx * dx + dy * dy)

    if arc_length > 0 and arc_length < needed_length:
        scale_factor = needed_length / arc_length
        # Compute centroid for scale-from-center
        cx = sum(p[0] for p in raw_positions) / len(raw_positions)
        cy = sum(p[1] for p in raw_positions) / len(raw_positions)
        raw_positions = [
            (cx + (p[0] - cx) * scale_factor, cy + (p[1] - cy) * scale_factor)
            for p in raw_positions
        ]
        print(f"Path scaled {scale_factor:.2f}x to fit {target_count} dominos "
              f"(arc {arc_length:.0f} -> {needed_length:.0f} BU)")

    # Step 2: Resample to uniform spacing
    positions = _resample_path(raw_positions, spacing)

    # Clamp count if resampling produced more/fewer
    if len(positions) > target_count:
        positions = positions[:target_count]

    print(f"Path: {archetype}, {len(positions)} dominoes, spacing={spacing:.2f}")

    # Step 3: Clear and set up scene
    _clear_scene()
    _setup_rigid_body_world(config)

    # Step 4: Pre-compute course bounds (for ground centering)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    bounds_min = Vector((min(xs), min(ys), 0))
    bounds_max = Vector((max(xs), max(ys), domino_h))
    center = (bounds_min + bounds_max) / 2

    # Step 5: Create ground — centered on the course with generous margin
    _create_ground(config, ground_center=(center.x, center.y))

    # Step 6: Place dominoes
    dominos = _place_dominoes(config, positions)
    print(f"Placed {len(dominos)} dominoes")

    # Step 7: Create trigger
    if dominos:
        _create_trigger(config, dominos[0], positions)

    # Step 8: Finale object
    if positions:
        _create_finale(config, positions[-1], positions)

    # Step 8: Camera and lighting (bounds already computed in Step 4)
    _setup_camera(config, center, (bounds_min, bounds_max), positions=positions)
    _setup_lighting(config, center)

    # Calculate path length
    path_length = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        path_length += math.sqrt(dx * dx + dy * dy)

    return {
        "seed": seed,
        "archetype": archetype,
        "domino_count": len(dominos),
        "path_length": round(path_length, 2),
        "course_center": [round(center.x, 3), round(center.y, 3), round(center.z, 3)],
        "course_bounds_min": [round(bounds_min.x, 3), round(bounds_min.y, 3), round(bounds_min.z, 3)],
        "course_bounds_max": [round(bounds_max.x, 3), round(bounds_max.y, 3), round(bounds_max.z, 3)],
        "duration_frames": config.get("fps", 30) * config.get("duration_sec", 65),
        "fps": config.get("fps", 30),
        "camera_style": config.get("camera_style", "tracking"),
        "lighting_preset": config.get("lighting_preset", "studio"),
        "palette": config.get("palette", "rainbow"),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run generation."""
    import argparse

    # Blender passes script args after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Generate a domino course")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--output-blend", default=None, help="Output .blend path")
    parser.add_argument("--output-json", default=None, help="Output result JSON path")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    result = generate_domino_course(config)

    # Save .blend
    blend_path = args.output_blend or str(config_path.parent / "domino_course.blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(blend_path).resolve()))
    result["blend_file"] = blend_path
    print(f"Saved .blend: {blend_path}")

    # Save result JSON
    json_path = args.output_json or str(config_path.parent / "generation_result.json")
    Path(json_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved result: {json_path}")

    # Print JSON for stdout capture
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
