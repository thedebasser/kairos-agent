"""Generate a marble course — zigzag ramp style.

Builds a series of alternating left/right ramps connected by U-turns,
with a start platform at top and finish tray at bottom.

Run via:
    blender --background --python generate_course.py -- --config <path.json>
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector, Matrix  # type: ignore[import-untyped]

# Ensure blend/scripts is on the path so we can import siblings
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from build_modules import (  # noqa: E402
    MODULE_BUILDERS,
    ModuleDef,
    Socket,
    WORLD_SCALE,
    _apply_material,
    _hex_to_rgb,
    _add_rigid_body_passive,
    TRACK_WIDTH,
    RAMP_LENGTH,
    RAMP_SLOPE_DEG,
    WALL_HEIGHT,
)


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
    duration_frames = config.get("duration_frames", 1950)
    scene.frame_start = 1
    scene.frame_end = duration_frames
    scene.render.fps = fps
    rbw.point_cache.frame_start = 1
    rbw.point_cache.frame_end = duration_frames
    scene.gravity = (0, 0, -9.81)


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

def _setup_camera(style: str, course_center: Vector,
                  course_height: float, config: dict[str, Any],
                  course_bounds_min: Vector | None = None,
                  course_bounds_max: Vector | None = None,
                  follow_target: bpy.types.Object | None = None) -> bpy.types.Object:
    """Create and position the camera for 9:16 vertical output.

    Default style is *marble_follow* — tracks the first marble with
    smooth constraints.  The zigzag course descends in XZ, and the
    camera sits at +Y looking -Y (looking at the "front" of the course
    like a pinball machine).
    """
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_data.clip_end = max(500, 50 * WORLD_SCALE)
    cam_data.clip_start = 0.1

    scene = bpy.context.scene
    res_x = config.get("resolution_x", 1080)
    res_y = config.get("resolution_y", 1920)
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    S = WORLD_SCALE

    if style == "marble_follow" and follow_target is not None:
        # ── Marble-follow camera ──────────────────────────────────
        # Camera is in front (+Y) of the course, looking at the marble.
        # COPY_LOCATION follows marble with offset.
        # DAMPED_TRACK + LOCKED_TRACK for stable aim without gimbal lock.
        cam_data.lens = 32.0

        # Offset: in front of course (+Y), slightly above
        offset = Vector((0, -5.0 * S, 1.0 * S))
        cam_obj.location = offset

        # Constraint 1: Follow marble position with offset
        con_loc = cam_obj.constraints.new(type="COPY_LOCATION")
        con_loc.name = "Follow_Marble"
        con_loc.target = follow_target
        con_loc.use_offset = True
        con_loc.influence = 0.85

        # Constraint 2: Always look at the marble
        con_track = cam_obj.constraints.new(type="DAMPED_TRACK")
        con_track.name = "Look_At_Marble"
        con_track.target = follow_target
        con_track.track_axis = "TRACK_NEGATIVE_Z"

        # Constraint 3: Stabilise roll
        con_roll = cam_obj.constraints.new(type="LOCKED_TRACK")
        con_roll.name = "Stabilise_Roll"
        con_roll.target = follow_target
        con_roll.track_axis = "TRACK_NEGATIVE_Z"
        con_roll.lock_axis = "LOCK_Y"

        # DOF on marble
        cam_data.dof.use_dof = True
        cam_data.dof.focus_object = follow_target
        cam_data.dof.aperture_fstop = 2.8

        print(f"[camera] marble_follow: tracking '{follow_target.name}', "
              f"offset={list(offset)}")

    elif style == "front_static" and course_bounds_min is not None and course_bounds_max is not None:
        # ── Front static view — auto-framed ──────────────────────
        # Camera sits in front, framing the entire course.
        cam_data.lens = 24.0
        # Compute distance needed to frame the course
        course_w = course_bounds_max.x - course_bounds_min.x
        course_h = course_bounds_max.z - course_bounds_min.z
        # Use the larger of width/height to determine distance
        sensor_w = cam_data.sensor_width  # 36mm default
        hfov = 2 * math.atan(sensor_w / (2 * cam_data.lens))
        vfov = 2 * math.atan((sensor_w * res_y / res_x) / (2 * cam_data.lens))
        dist_for_w = (course_w * 1.4) / (2 * math.tan(hfov / 2))
        dist_for_h = (course_h * 1.4) / (2 * math.tan(vfov / 2))
        dist = max(dist_for_w, dist_for_h, 10.0)

        cam_obj.location = Vector((
            course_center.x,
            course_center.y - dist,
            course_center.z,
        ))
        # Look straight at the course (+Y direction)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)

        print(f"[camera] front_static: dist={dist:.1f}, center={list(course_center)}")

    elif style == "dolly" and course_bounds_min is not None and course_bounds_max is not None:
        # ── Keyframed dolly cam — descends alongside course ──────
        cam_data.lens = 28.0
        offset_y = -5.0 * S
        z_top = course_bounds_max.z + 1.0 * S
        z_bot = course_bounds_min.z - 0.5 * S

        cam_obj.location = Vector((
            course_center.x,
            course_center.y + offset_y,
            z_top,
        ))
        cam_obj.keyframe_insert(data_path="location", frame=1)

        cam_obj.location = Vector((
            course_center.x,
            course_center.y + offset_y,
            z_bot,
        ))
        duration = config.get("duration_frames",
                              bpy.context.scene.frame_end)
        cam_obj.keyframe_insert(data_path="location", frame=duration)

        # Ease keyframes (Blender 5.0 layered actions)
        if cam_obj.animation_data and cam_obj.animation_data.action:
            act = cam_obj.animation_data.action
            for layer in act.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for fcurve in bag.fcurves:
                            for kp in fcurve.keyframe_points:
                                kp.interpolation = "BEZIER"
                                kp.easing = "EASE_IN_OUT"

        # Look at centre
        empty = bpy.data.objects.new("DollyTarget", None)
        bpy.context.collection.objects.link(empty)
        empty.location = course_center
        con = cam_obj.constraints.new(type="DAMPED_TRACK")
        con.target = empty
        con.track_axis = "TRACK_NEGATIVE_Z"
        con_roll = cam_obj.constraints.new(type="LOCKED_TRACK")
        con_roll.target = empty
        con_roll.track_axis = "TRACK_NEGATIVE_Z"
        con_roll.lock_axis = "LOCK_Y"

        cam_data.dof.use_dof = True
        cam_data.dof.focus_object = empty
        cam_data.dof.aperture_fstop = 4.0

        print(f"[camera] dolly: z_top={z_top:.1f} → z_bot={z_bot:.1f}")

    else:
        # Fallback: elevated front view
        cam_data.lens = 35.0
        cam_obj.location = course_center + Vector((2.0 * S, -4.0 * S, 1.5 * S))
        cam_obj.rotation_euler = (math.radians(75), 0, math.radians(15))

    return cam_obj


def _setup_lighting(preset_name: str, config: dict[str, Any],
                    course_center: Vector | None = None) -> None:
    """Three-point light rig, scaled for WORLD_SCALE geometry."""
    presets = config.get("lighting_presets", {})
    preset = presets.get(preset_name, presets.get("studio", {}))

    energy_scale = WORLD_SCALE ** 2
    light_center = course_center or Vector((0, 0, 0))

    # Key light — high and to the front-left
    key = bpy.data.lights.new("Key", type="AREA")
    key.energy = preset.get("key_energy", 800) * energy_scale
    key.color = (1.0, 0.95, 0.9)
    key.size = 5.0 * WORLD_SCALE
    key_obj = bpy.data.objects.new("Key_Light", key)
    key_obj.location = Vector((2, -3, 3)) * WORLD_SCALE + light_center
    key_obj.rotation_euler = (math.radians(55), 0, math.radians(30))
    bpy.context.collection.objects.link(key_obj)

    # Fill light — front-right
    fill = bpy.data.lights.new("Fill", type="AREA")
    fill.energy = preset.get("fill_energy", 300) * energy_scale
    fill.color = (0.9, 0.95, 1.0)
    fill.size = 5.0 * WORLD_SCALE
    fill_obj = bpy.data.objects.new("Fill_Light", fill)
    fill_obj.location = Vector((-2, -2, 2)) * WORLD_SCALE + light_center
    fill_obj.rotation_euler = (math.radians(60), 0, math.radians(-30))
    bpy.context.collection.objects.link(fill_obj)

    # Rim light — behind and above
    rim = bpy.data.lights.new("Rim", type="SPOT")
    rim.energy = preset.get("rim_energy", 500) * energy_scale
    rim.color = (1.0, 1.0, 1.0)
    rim.spot_size = math.radians(90)
    rim_obj = bpy.data.objects.new("Rim_Light", rim)
    rim_obj.location = Vector((0, 2, 3)) * WORLD_SCALE + light_center
    rim_obj.rotation_euler = (math.radians(120), 0, 0)
    bpy.context.collection.objects.link(rim_obj)

    # World background
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        base_strength = preset.get("world_strength", 0.3)
        bg.inputs["Strength"].default_value = max(base_strength, 0.4)
        bg.inputs["Color"].default_value = (0.12, 0.12, 0.16, 1.0)


# ---------------------------------------------------------------------------
# Zigzag module placement
# ---------------------------------------------------------------------------

def _build_zigzag_sequence(config: dict[str, Any],
                           rng: random.Random) -> list[dict[str, Any]]:
    """Build a deterministic zigzag ramp sequence.

    Structure: start_platform → [ramp → turn]×N → ramp → finish_tray
    The number of ramp pairs = n_zigzags (default 3).
    Ramps alternate right / left.
    """
    n_zigzags = config.get("n_zigzags", 3)
    gate_open = config.get("gate_open_frame", 30)
    palette = config.get("colour_palette", ["#4A90D9"])

    # Pick colours for ramps
    ramp_colours = []
    for i in range(n_zigzags * 2 + 1):
        ramp_colours.append(palette[i % len(palette)])

    sequence: list[dict[str, Any]] = []

    # Start platform
    sequence.append({
        "type": "start_platform",
        "params": {"gate_open_frame": gate_open, "colour": "#B0B0B0"},
    })

    # Alternating ramps and turns
    directions = []
    for i in range(n_zigzags * 2):
        d = "right" if i % 2 == 0 else "left"
        directions.append(d)

    for i, d in enumerate(directions):
        sequence.append({
            "type": "ramp_section",
            "params": {"direction": d, "colour": ramp_colours[i]},
        })
        # Add a turn after each ramp (except the last one)
        if i < len(directions) - 1:
            turn_side = d  # turn is at the end where ramp ended
            sequence.append({
                "type": "turn_section",
                "params": {"side": turn_side, "colour": ramp_colours[i]},
            })

    # Finish tray
    sequence.append({
        "type": "finish_tray",
        "params": {"colour": "#2ECC71"},
    })

    return sequence


def _place_modules(module_sequence: list[dict[str, Any]],
                   rng: random.Random) -> list[ModuleDef]:
    """Build and place modules socket-to-socket.

    Same transform_apply workflow as before: remove RB → move →
    transform_apply → re-add RB so Bullet reads baked mesh positions.
    """
    placed: list[ModuleDef] = []
    current_pos = Vector((0, 0, 0))

    for i, mod_spec in enumerate(module_sequence):
        mod_type = mod_spec["type"]
        if mod_type not in MODULE_BUILDERS:
            print(f"WARNING: Unknown module type '{mod_type}' — skipping")
            continue

        params = mod_spec.get("params", {})
        builder = MODULE_BUILDERS[mod_type]
        mod = builder(**params)

        # Compute placement offset
        if i == 0:
            offset = current_pos - mod.entry_socket.position
        else:
            prev_exit = placed[-1].exit_socket
            offset = prev_exit.position - mod.entry_socket.position

        for obj in mod.objects:
            # 1. Save & remove rigid body
            rb_props = None
            if obj.rigid_body:
                rb_props = {
                    "type":        obj.rigid_body.type,
                    "shape":       obj.rigid_body.collision_shape,
                    "friction":    obj.rigid_body.friction,
                    "restitution": obj.rigid_body.restitution,
                    "margin":      obj.rigid_body.collision_margin,
                    "kinematic":   obj.rigid_body.kinematic,
                }
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_remove()

            # 2. Clear animation data
            if obj.animation_data:
                obj.animation_data_clear()

            # 3. Move
            obj.location += offset

            # 4. Bake location into mesh
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True)
            obj.select_set(False)

            # 5. Re-add rigid body
            if rb_props:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_add(type=rb_props["type"])
                rb = obj.rigid_body
                rb.collision_shape = rb_props["shape"]
                rb.friction        = rb_props["friction"]
                rb.restitution     = rb_props["restitution"]
                rb.collision_margin = rb_props["margin"]
                rb.use_margin      = True
                rb.kinematic       = rb_props.get("kinematic", False)

            # 6. Re-create gate animation
            is_gate = (mod_type == "start_platform"
                       and obj.name.endswith("_gate"))
            if is_gate and rb_props:
                gate_open = params.get("gate_open_frame", 30)
                gate_open_end = gate_open + 10
                wh = WALL_HEIGHT
                rb = obj.rigid_body
                rb.kinematic = True

                obj.location = (0, 0, 0)
                obj.keyframe_insert(data_path="location",
                                    frame=gate_open - 1)
                obj.location = (0, 0, wh + 0.5)
                obj.keyframe_insert(data_path="location",
                                    frame=gate_open_end)
                obj.location = (0, 0, 0)

        # Update sockets & bounds
        mod.entry_socket.position += offset
        mod.exit_socket.position += offset
        mod.bounding_min += offset
        mod.bounding_max += offset

        placed.append(mod)
        current_pos = mod.exit_socket.position

    return placed


# ---------------------------------------------------------------------------
# Marble spawning
# ---------------------------------------------------------------------------

def _spawn_marbles(config: dict[str, Any], start_module: ModuleDef,
                   rng: random.Random) -> list[bpy.types.Object]:
    """Spawn marbles above the start platform floor."""
    count = config.get("marble_count", 5)
    radius = config.get("marble_radius", 0.04) * WORLD_SCALE
    bounce = config.get("marble_bounce", 0.4)
    friction = config.get("marble_friction", 0.6)
    mass = config.get("marble_mass", 1.0)
    palette = config.get("colour_palette",
                         ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"])

    # Spawn centered above start platform
    floor_cx = (start_module.bounding_min.x + start_module.bounding_max.x) / 2
    floor_cy = (start_module.bounding_min.y + start_module.bounding_max.y) / 2
    floor_top_z = start_module.bounding_max.z

    # Grid: 5 marbles in a row
    marbles = []
    spacing = radius * 2.8
    start_x = floor_cx - (count - 1) / 2 * spacing

    for i in range(count):
        x = start_x + i * spacing
        x += rng.uniform(-radius * 0.1, radius * 0.1)
        y = floor_cy + rng.uniform(-radius * 0.3, radius * 0.3)
        z = floor_top_z + radius * 1.5

        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius, location=(x, y, z),
            segments=16, ring_count=8,
        )
        marble = bpy.context.active_object
        marble.name = f"marble_{i:03d}"
        bpy.ops.object.shade_smooth()

        bpy.ops.rigidbody.object_add(type="ACTIVE")
        rb = marble.rigid_body
        rb.mass = mass
        rb.collision_shape = "SPHERE"
        rb.friction = friction
        rb.restitution = bounce
        rb.collision_margin = 0.002 * WORLD_SCALE
        rb.linear_damping = 0.01
        rb.angular_damping = 0.02
        rb.use_deactivation = False

        colour = palette[i % len(palette)]
        _apply_material(marble, colour, metallic=0.6, roughness=0.2,
                        name=f"marble_mat_{colour}")
        marbles.append(marble)

    return marbles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_course(config: dict[str, Any]) -> dict[str, Any]:
    """Generate a full marble course scene in Blender."""
    seed = config.get("seed", 42)
    rng = random.Random(seed)

    # Load configs
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    params_data = json.loads((configs_dir / "parameters.json").read_text())

    # Marble settings
    marble_count = config.get("marble_count", 5)
    marble_radius = config.get("marble_radius",
                                params_data["parameter_ranges"]["marble_radius"]["default"])

    # Palette
    palette_name = config.get("palette",
                              rng.choice(list(params_data["colour_palettes"].keys())))
    palette = params_data["colour_palettes"].get(
        palette_name, params_data["colour_palettes"]["rainbow"])

    # Duration
    fps = config.get("fps", 30)
    duration_sec = config.get("duration_sec", 65)
    duration_frames = fps * duration_sec

    # Build scene
    _clear_scene()
    _setup_rigid_body_world({
        "substeps_per_frame": config.get("substeps_per_frame", 20),
        "solver_iterations": config.get("solver_iterations", 20),
        "fps": fps,
        "duration_frames": duration_frames,
    })

    # Build zigzag module sequence
    course_config = {
        "n_zigzags": config.get("n_zigzags", 3),
        "gate_open_frame": config.get("gate_open_frame", 30),
        "colour_palette": palette,
    }
    module_sequence = _build_zigzag_sequence(course_config, rng)

    # Place modules
    placed_modules = _place_modules(module_sequence, rng)

    # Force depsgraph update
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()

    # Course bounds
    all_min = Vector((float("inf"),) * 3)
    all_max = Vector((float("-inf"),) * 3)
    for m in placed_modules:
        for i in range(3):
            all_min[i] = min(all_min[i], m.bounding_min[i])
            all_max[i] = max(all_max[i], m.bounding_max[i])
    course_center = (all_min + all_max) / 2
    course_height = all_max.z - all_min.z

    # Spawn marbles
    marble_config = {
        "marble_count": marble_count,
        "marble_radius": marble_radius,
        "marble_bounce": config.get("marble_bounce",
                                     params_data["parameter_ranges"]["marble_bounce"]["default"]),
        "marble_friction": config.get("marble_friction",
                                       params_data["parameter_ranges"]["marble_friction"]["default"]),
        "marble_mass": config.get("marble_mass",
                                   params_data["parameter_ranges"]["marble_mass"]["default"]),
        "colour_palette": palette,
    }
    marbles = _spawn_marbles(marble_config, placed_modules[0], rng)

    # Invisible catch basin below course
    basin_z = all_min.z - 5.0 * WORLD_SCALE
    basin_size = max(all_max.x - all_min.x, all_max.y - all_min.y) * 3
    bpy.ops.mesh.primitive_cube_add(size=1)
    basin = bpy.context.active_object
    basin.name = "catch_basin"
    basin.scale = (basin_size, basin_size, 0.5 * WORLD_SCALE)
    basin.location = (course_center.x, course_center.y, basin_z)
    bpy.ops.object.transform_apply(scale=True)
    _add_rigid_body_passive(basin, shape="BOX")
    basin.hide_render = True
    basin.hide_viewport = False

    # Camera — default marble_follow
    camera_style = config.get("camera_style", "marble_follow")
    lighting_preset = config.get("lighting_preset",
                                  rng.choice(list(params_data["lighting_presets"].keys())))

    cam_config = {
        "resolution_x": 1080,
        "resolution_y": 1920,
        "duration_frames": duration_frames,
    }
    follow_target = marbles[0] if marbles else None
    _setup_camera(camera_style, course_center, course_height, cam_config,
                  course_bounds_min=all_min, course_bounds_max=all_max,
                  follow_target=follow_target)
    _setup_lighting(lighting_preset, params_data, course_center=course_center)

    bpy.context.scene["marble_course_config"] = json.dumps(config)
    bpy.context.scene["marble_course_seed"] = seed

    return {
        "seed": seed,
        "archetype": "zigzag_race",
        "module_count": len(placed_modules),
        "module_names": [m.name for m in placed_modules],
        "marble_count": len(marbles),
        "course_center": list(course_center),
        "course_bounds_min": list(all_min),
        "course_bounds_max": list(all_max),
        "duration_frames": duration_frames,
        "fps": fps,
        "camera_style": camera_style,
        "lighting_preset": lighting_preset,
        "palette": palette_name,
    }


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
    parser.add_argument("--config", required=True,
                        help="Path to generation config JSON")
    parser.add_argument("--output-blend", default=None,
                        help="Save .blend file")
    parser.add_argument("--output-json", default=None,
                        help="Save result JSON")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    result = generate_course(config)

    if args.output_blend:
        bpy.ops.wm.save_as_mainfile(
            filepath=str(Path(args.output_blend).resolve()))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
