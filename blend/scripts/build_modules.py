"""Marble-run module builders — zigzag ramp style.

Each module creates Blender mesh objects with rigid-body passive
colliders.  Modules expose entry/exit sockets for chaining.

Design: A real marble run is a series of tilted ramps that zigzag
left ↔ right, descending in Z.  The viewer watches from +Y looking
at the XZ plane (like a pinball machine).

This file runs **inside Blender's Python** (bpy available).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector, Matrix  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# World scale — Bullet needs objects ~1 m+ for reliable collision.
# We build at 10× real-world size with standard gravity.
# ---------------------------------------------------------------------------
WORLD_SCALE: int = 10

# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class Socket:
    """Connection point on a module (world-space)."""
    position: Vector
    direction: Vector

    def transformed(self, matrix: Matrix) -> "Socket":
        return Socket(
            position=matrix @ self.position,
            direction=(matrix.to_3x3() @ self.direction).normalized(),
        )


@dataclass
class ModuleDef:
    """Metadata returned after building a module."""
    name: str
    objects: list[bpy.types.Object]
    entry_socket: Socket
    exit_socket: Socket
    bounding_min: Vector
    bounding_max: Vector
    safe_slope_range: tuple[float, float] = (5.0, 25.0)
    safe_marble_radius: tuple[float, float] = (0.02, 0.08)
    required_wall_height: float = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_module_counter: dict[str, int] = {}


def _unique_name(base: str) -> str:
    _module_counter[base] = _module_counter.get(base, 0) + 1
    return f"{base}_{_module_counter[base]:03d}"


def _add_rigid_body_passive(obj: bpy.types.Object, friction: float = 0.6,
                             bounce: float = 0.3, shape: str = "BOX") -> None:
    """Passive rigid-body collider."""
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    rb = obj.rigid_body
    rb.collision_shape = shape
    rb.friction = friction
    rb.restitution = bounce
    if shape == "MESH":
        rb.collision_margin = 0.0
        rb.use_margin = True
    else:
        rb.collision_margin = 0.004 * WORLD_SCALE
        rb.use_margin = True


def _apply_material(obj: bpy.types.Object, colour_hex: str = "#808080",
                     metallic: float = 0.1, roughness: float = 0.5,
                     name: str | None = None) -> None:
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


def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Constants for the zigzag track
# ---------------------------------------------------------------------------
S = WORLD_SCALE

# Track dimensions (in Blender units = pre-scale × WORLD_SCALE)
TRACK_WIDTH    = 0.5 * S    # 5.0 — width of the track channel (Y-dir)
RAMP_LENGTH    = 1.8 * S    # 18.0 — horizontal run per ramp section (X-dir)
RAMP_SLOPE_DEG = 12.0       # gentle slope for cinematic roll
RAMP_THICKNESS = 0.05 * S   # 0.5 — thick enough for Bullet BOX collision
WALL_HEIGHT    = 0.20 * S   # 2.0 — side walls (~2.5× marble diameter)
WALL_THICK     = 0.03 * S   # 0.3 — wall thickness
TURN_RADIUS    = 0.4 * S    # 4.0 — radius of the U-turn at each end


# ---------------------------------------------------------------------------
# Module: Start Platform
# ---------------------------------------------------------------------------

def build_start_platform(
    *,
    width: float = TRACK_WIDTH,
    colour: str = "#B0B0B0",
    gate_open_frame: int = 30,
    **_kw: Any,
) -> ModuleDef:
    """Flat platform at the top where marbles are released.

    A short level floor with walls on three sides (left, right, back)
    and a gate at the front that opens at *gate_open_frame*.
    The floor tilts slightly (5°) toward the exit so marbles roll out.
    """
    name = _unique_name("start_platform")
    floor_len = 0.8 * S   # 8 units — short platform
    floor_w = width
    thick = RAMP_THICKNESS
    wh = WALL_HEIGHT
    wt = WALL_THICK
    slope_rad = math.radians(5.0)

    objects: list[bpy.types.Object] = []

    # ── Floor (slightly tilted toward +X = exit direction) ─────────
    bpy.ops.mesh.primitive_cube_add(size=1)
    floor = bpy.context.active_object
    floor.name = f"{name}_floor"
    floor.scale = Vector((floor_len, floor_w, thick))
    floor.location = Vector((floor_len / 2, 0, -thick / 2))
    floor.rotation_euler.y = -slope_rad  # right side (exit) is lower
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    _add_rigid_body_passive(floor, friction=0.8)
    _apply_material(floor, colour, metallic=0.3, roughness=0.4)
    objects.append(floor)

    # ── Back wall (at X=0, blocks marbles from rolling backward) ───
    bpy.ops.mesh.primitive_cube_add(size=1)
    bwall = bpy.context.active_object
    bwall.name = f"{name}_back"
    bwall.scale = Vector((wt, floor_w + 2 * wt, wh))
    bwall.location = Vector((-wt / 2, 0, wh / 2))
    bpy.ops.object.transform_apply(scale=True)
    _add_rigid_body_passive(bwall)
    _apply_material(bwall, colour, metallic=0.3, roughness=0.4)
    objects.append(bwall)

    # ── Side walls (front and back in Y) ────────────────────────────
    for sy, sfx in [(-floor_w / 2 - wt / 2, "fwall"), (floor_w / 2 + wt / 2, "bkwall")]:
        bpy.ops.mesh.primitive_cube_add(size=1)
        sw = bpy.context.active_object
        sw.name = f"{name}_{sfx}"
        sw.scale = Vector((floor_len + wt, wt, wh))
        sw.location = Vector((floor_len / 2, sy, wh / 2))
        bpy.ops.object.transform_apply(scale=True)
        _add_rigid_body_passive(sw)
        _apply_material(sw, colour, metallic=0.3, roughness=0.4)
        objects.append(sw)

    # ── Gate (at X = floor_len, opens upward) ──────────────────────
    bpy.ops.mesh.primitive_cube_add(size=1)
    gate = bpy.context.active_object
    gate.name = f"{name}_gate"
    gate.scale = Vector((wt, floor_w + 2 * wt, wh))
    gate.location = Vector((floor_len + wt / 2, 0, wh / 2))
    bpy.ops.object.transform_apply(scale=True)
    _add_rigid_body_passive(gate)
    _apply_material(gate, "#FFD700", metallic=0.5, roughness=0.3)

    rb = gate.rigid_body
    rb.kinematic = True
    gate_open_end = gate_open_frame + 10
    gate.keyframe_insert(data_path="location", frame=gate_open_frame - 1)
    gate.location.z += wh + 0.5
    gate.keyframe_insert(data_path="location", frame=gate_open_end)
    gate.location.z -= wh + 0.5  # reset to closed
    objects.append(gate)

    # Drop from the slope
    drop = floor_len * math.sin(slope_rad)
    exit_x = floor_len
    exit_z = -drop

    return ModuleDef(
        name=name,
        objects=objects,
        entry_socket=Socket(Vector((0, 0, 0)), Vector((1, 0, 0))),
        exit_socket=Socket(Vector((exit_x, 0, exit_z)), Vector((1, 0, -0.2)).normalized()),
        bounding_min=Vector((-wt, -floor_w / 2 - wt, -drop - thick)),
        bounding_max=Vector((floor_len + wt, floor_w / 2 + wt, wh)),
        required_wall_height=wh,
    )


# ---------------------------------------------------------------------------
# Module: Ramp Section (the core building block)
# ---------------------------------------------------------------------------

def build_ramp_section(
    *,
    direction: str = "right",
    length: float = RAMP_LENGTH,
    width: float = TRACK_WIDTH,
    slope_deg: float = RAMP_SLOPE_DEG,
    colour: str = "#4A90D9",
    **_kw: Any,
) -> ModuleDef:
    """A single tilted ramp going left or right.

    The ramp runs along X (positive = right, negative = left).
    It descends in Z.  Track width is in Y.
    Camera views from +Y direction.

    *direction*: "right" means ramp goes in +X direction (left-to-right).
                 "left" means ramp goes in -X direction (right-to-left).
    """
    name = _unique_name("ramp")
    thick = RAMP_THICKNESS
    wh = WALL_HEIGHT
    wt = WALL_THICK
    slope_rad = math.radians(slope_deg)

    # Sign: +1 for right, -1 for left
    sign = 1.0 if direction == "right" else -1.0

    # The ramp center is at (sign*length/2, 0, -drop/2) relative to entry
    drop = length * math.sin(slope_rad)
    run = length * math.cos(slope_rad)

    objects: list[bpy.types.Object] = []

    # ── Ramp floor ────────────────────────────────────────────────
    bpy.ops.mesh.primitive_cube_add(size=1)
    ramp = bpy.context.active_object
    ramp.name = f"{name}_floor"
    ramp.scale = Vector((length, width, thick))
    ramp.location = Vector((sign * run / 2, 0, -drop / 2))
    # Tilt: rotate around Y axis so the "forward" end (in X) is lower
    ramp.rotation_euler.y = -sign * slope_rad
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    _add_rigid_body_passive(ramp, friction=0.7, shape="BOX")
    _apply_material(ramp, colour, metallic=0.2, roughness=0.5)
    objects.append(ramp)

    # ── Side walls (along Y = ±width/2) ──────────────────────────
    for sy, sfx in [(-width / 2 - wt / 2, "front_wall"),
                     (width / 2 + wt / 2, "back_wall")]:
        bpy.ops.mesh.primitive_cube_add(size=1)
        wall = bpy.context.active_object
        wall.name = f"{name}_{sfx}"
        wall.scale = Vector((length, wt, wh))
        wall.location = Vector((sign * run / 2, sy, -drop / 2 + wh / 2))
        wall.rotation_euler.y = -sign * slope_rad
        bpy.ops.object.transform_apply(scale=True, rotation=True)
        _add_rigid_body_passive(wall, shape="BOX")
        _apply_material(wall, colour, metallic=0.15, roughness=0.55)
        objects.append(wall)

    # Entry and exit positions
    entry = Vector((0, 0, 0))
    exit_pos = Vector((sign * run, 0, -drop))
    exit_dir = Vector((sign, 0, -0.2)).normalized()

    # Bounding box
    x_lo = min(0, sign * run) - wt
    x_hi = max(0, sign * run) + wt
    return ModuleDef(
        name=name,
        objects=objects,
        entry_socket=Socket(entry, Vector((sign, 0, 0))),
        exit_socket=Socket(exit_pos, exit_dir),
        bounding_min=Vector((x_lo, -width / 2 - wt, -drop - thick)),
        bounding_max=Vector((x_hi, width / 2 + wt, wh)),
        required_wall_height=wh,
    )


# ---------------------------------------------------------------------------
# Module: U-Turn (connects two ramps going opposite directions)
# ---------------------------------------------------------------------------

def build_turn_section(
    *,
    side: str = "right",
    width: float = TRACK_WIDTH,
    colour: str = "#D94A4A",
    **_kw: Any,
) -> ModuleDef:
    """U-turn at the end of a ramp section.

    *side*: "right" means the turn is at the right end (+X) of a
    right-going ramp.  "left" for the left end of a left-going ramp.

    The turn is a simple flat platform with a curved outer wall.
    Marbles roll in from one direction, hit the curved wall, and
    exit going the opposite direction (slightly below, thanks to
    a small Z-drop built into the turn floor).
    """
    name = _unique_name("turn")
    thick = RAMP_THICKNESS
    wh = WALL_HEIGHT
    wt = WALL_THICK
    turn_drop = 0.15 * S  # 1.5 units — small drop through the turn
    # The turn is roughly a half-pipe shape — we approximate with
    # a flat floor + a curved outer wall made of segments
    turn_width = width + 0.2 * S  # slightly wider for the curve

    sign = 1.0 if side == "right" else -1.0

    objects: list[bpy.types.Object] = []

    # ── Turn floor — flat-ish with slight slope toward exit ────────
    floor_size_x = turn_width
    floor_size_y = width
    bpy.ops.mesh.primitive_cube_add(size=1)
    floor = bpy.context.active_object
    floor.name = f"{name}_floor"
    floor.scale = Vector((floor_size_x, floor_size_y, thick))
    # Slope the floor slightly toward the exit (negative Z on opposite side)
    floor.location = Vector((sign * floor_size_x / 2, 0, -turn_drop / 2))
    # Small tilt toward exit direction
    slope_rad = math.atan2(turn_drop, floor_size_x)
    floor.rotation_euler.y = sign * slope_rad
    bpy.ops.object.transform_apply(scale=True, rotation=True)
    _add_rigid_body_passive(floor, friction=0.8, shape="BOX")
    _apply_material(floor, colour, metallic=0.2, roughness=0.4)
    objects.append(floor)

    # ── Outer curved wall (semicircle of segments) ─────────────────
    n_segs = 6
    curve_radius = floor_size_x / 2
    for i in range(n_segs):
        a0 = math.pi * i / n_segs
        a1 = math.pi * (i + 1) / n_segs
        amid = (a0 + a1) / 2
        seg_len = curve_radius * (a1 - a0)

        # Position on arc — the arc is centered at (sign * floor_size_x / 2, 0)
        # and curves from entry to exit
        cx = sign * (floor_size_x / 2 + curve_radius * math.cos(amid) * 0.05)
        cy = curve_radius * math.sin(amid) - width / 2 if i < n_segs // 2 else \
             -(curve_radius * math.sin(math.pi - amid) - width / 2)

        # Simpler: outer wall is just a flat wall on the outer side (+X for right turn)
        # We'll place wall segments in a semicircle around the outer edge
        seg_angle = amid - math.pi / 2  # tangent direction
        wx = sign * floor_size_x / 2 + sign * (width / 2) * math.cos(amid)
        wy = (width / 2) * math.sin(amid) - width / 4

        bpy.ops.mesh.primitive_cube_add(size=1)
        seg = bpy.context.active_object
        seg.name = f"{name}_curve_{i}"
        seg.scale = Vector((wt, seg_len * 1.2, wh))
        seg.location = Vector((wx, wy, -turn_drop / 2 + wh / 2))
        seg.rotation_euler.z = amid if sign > 0 else -amid
        bpy.ops.object.transform_apply(scale=True, rotation=True)
        _add_rigid_body_passive(seg, friction=0.7, shape="CONVEX_HULL")
        _apply_material(seg, colour, metallic=0.2, roughness=0.4)
        objects.append(seg)

    # ── End wall (closes the far end) ──────────────────────────────
    bpy.ops.mesh.primitive_cube_add(size=1)
    endwall = bpy.context.active_object
    endwall.name = f"{name}_endwall"
    endwall.scale = Vector((wt, floor_size_y + 2 * wt, wh))
    endwall.location = Vector((sign * (floor_size_x + wt / 2), 0, -turn_drop / 2 + wh / 2))
    bpy.ops.object.transform_apply(scale=True)
    _add_rigid_body_passive(endwall, shape="BOX")
    _apply_material(endwall, colour, metallic=0.15, roughness=0.55)
    objects.append(endwall)

    # ── Inner side walls (Y direction) ─────────────────────────────
    for sy, sfx in [(-width / 2 - wt / 2, "fwall"), (width / 2 + wt / 2, "bwall")]:
        bpy.ops.mesh.primitive_cube_add(size=1)
        sw = bpy.context.active_object
        sw.name = f"{name}_{sfx}"
        sw.scale = Vector((floor_size_x, wt, wh))
        sw.location = Vector((sign * floor_size_x / 2, sy, -turn_drop / 2 + wh / 2))
        bpy.ops.object.transform_apply(scale=True)
        _add_rigid_body_passive(sw, shape="BOX")
        _apply_material(sw, colour, metallic=0.15, roughness=0.55)
        objects.append(sw)

    entry = Vector((0, 0, 0))
    # Exit goes back the other way, slightly lower
    exit_pos = Vector((0, 0, -turn_drop))
    exit_dir = Vector((-sign, 0, -0.1)).normalized()

    x_lo = min(0, sign * floor_size_x) - wt
    x_hi = max(0, sign * floor_size_x) + wt

    return ModuleDef(
        name=name,
        objects=objects,
        entry_socket=Socket(entry, Vector((sign, 0, 0))),
        exit_socket=Socket(exit_pos, exit_dir),
        bounding_min=Vector((x_lo, -width / 2 - wt, -turn_drop - thick)),
        bounding_max=Vector((x_hi, width / 2 + wt, wh)),
        required_wall_height=wh,
    )


# ---------------------------------------------------------------------------
# Module: Finish Tray
# ---------------------------------------------------------------------------

def build_finish_tray(
    *,
    width: float = TRACK_WIDTH,
    colour: str = "#2ECC71",
    **_kw: Any,
) -> ModuleDef:
    """Collection tray at the bottom of the course."""
    name = _unique_name("finish_tray")
    tray_len = 1.2 * S   # 12 units
    tray_w = width + 0.3 * S  # slightly wider than track
    thick = RAMP_THICKNESS
    wh = WALL_HEIGHT
    wt = WALL_THICK * 2  # thicker walls for the catch tray

    objects: list[bpy.types.Object] = []

    # ── Floor ──────────────────────────────────────────────────────
    bpy.ops.mesh.primitive_cube_add(size=1)
    floor = bpy.context.active_object
    floor.name = f"{name}_floor"
    floor.scale = Vector((tray_len, tray_w, thick))
    floor.location = Vector((0, 0, -thick / 2))
    bpy.ops.object.transform_apply(scale=True)
    _add_rigid_body_passive(floor, friction=0.8, shape="BOX")
    _apply_material(floor, colour, metallic=0.1, roughness=0.6)
    objects.append(floor)

    # ── Walls on all 4 sides ──────────────────────────────────────
    walls = [
        # (position, scale, name_suffix)
        (Vector((-tray_len / 2 - wt / 2, 0, wh / 2)),
         Vector((wt, tray_w + 2 * wt, wh)), "left"),
        (Vector((tray_len / 2 + wt / 2, 0, wh / 2)),
         Vector((wt, tray_w + 2 * wt, wh)), "right"),
        (Vector((0, -tray_w / 2 - wt / 2, wh / 2)),
         Vector((tray_len + 2 * wt, wt, wh)), "front"),
        (Vector((0, tray_w / 2 + wt / 2, wh / 2)),
         Vector((tray_len + 2 * wt, wt, wh)), "back"),
    ]

    for pos, scl, sfx in walls:
        bpy.ops.mesh.primitive_cube_add(size=1)
        w = bpy.context.active_object
        w.name = f"{name}_{sfx}"
        w.scale = scl
        w.location = pos
        bpy.ops.object.transform_apply(scale=True)
        _add_rigid_body_passive(w, shape="BOX")
        _apply_material(w, colour, metallic=0.1, roughness=0.6)
        objects.append(w)

    return ModuleDef(
        name=name,
        objects=objects,
        entry_socket=Socket(Vector((0, 0, wh)), Vector((0, 0, -1))),
        exit_socket=Socket(Vector((0, 0, 0)), Vector((0, 0, -1))),
        bounding_min=Vector((-tray_len / 2 - wt, -tray_w / 2 - wt, -thick)),
        bounding_max=Vector((tray_len / 2 + wt, tray_w / 2 + wt, wh)),
        required_wall_height=wh,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODULE_BUILDERS: dict[str, Any] = {
    "start_platform": build_start_platform,
    "ramp_section": build_ramp_section,
    "turn_section": build_turn_section,
    "finish_tray": build_finish_tray,
}
