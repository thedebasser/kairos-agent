"""Apply environment theming to a Blender scene.

Run AFTER generate_domino_course.py and BEFORE bake_and_render.py:

    blender --background scene.blend --python setup_environment.py \
        -- --theme-config theme.json

This script:
  1. Loads the theme config JSON
  2. Applies HDRI world background (if available)
  3. Applies ground texture material (if available)
  4. Applies themed domino materials
  5. Sets up compositor post-processing nodes
  6. Adds decorative environment objects (NON-INTERACTIVE / no physics)
  7. Saves the modified .blend

ALL environment objects are passive decoration only — they have NO rigid body
physics so they cannot interfere with domino/ball simulations.
"""

from __future__ import annotations

import colorsys
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# HDRI World Background
# ---------------------------------------------------------------------------

def apply_hdri(hdri_path: str, strength: float = 1.0) -> bool:
    """Apply an HDRI image as the world background.

    Args:
        hdri_path: Absolute path to the .exr / .hdr file.
        strength: Background strength (default 1.0).

    Returns True on success.
    """
    path = Path(hdri_path)
    if not path.exists():
        print(f"[env] HDRI file not found: {path}")
        return False

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    # Create nodes
    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (400, 0)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.location = (200, 0)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.location = (-200, 0)
    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-600, 0)
    mapp = nt.nodes.new("ShaderNodeMapping")
    mapp.location = (-400, 0)

    # Load image
    env.image = bpy.data.images.load(str(path))
    bg.inputs["Strength"].default_value = strength

    # Random rotation for variety
    rotation_z = random.uniform(0, 2 * math.pi)
    mapp.inputs["Rotation"].default_value[2] = rotation_z

    # Link nodes
    nt.links.new(coord.outputs["Generated"], mapp.inputs["Vector"])
    nt.links.new(mapp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    print(f"[env] Applied HDRI: {path.name} (strength={strength:.2f}, "
          f"rot={math.degrees(rotation_z):.0f}°)")
    return True


# ---------------------------------------------------------------------------
# Ground Texture Material
# ---------------------------------------------------------------------------

def apply_ground_texture(
    ground_obj: bpy.types.Object,
    texture_maps: dict[str, str],
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    uv_scale: float = 1.0,
) -> bool:
    """Apply PBR texture maps to the ground plane.

    Args:
        ground_obj: The ground plane object.
        texture_maps: Dict mapping map type to file path:
            {"diff": "path/to/diffuse.jpg", "rough": "...", "nor_gl": "..."}
        tint: RGB tint to multiply with the diffuse map.
        uv_scale: UV tiling scale (higher = more repeats).

    Returns True on success.
    """
    if not texture_maps:
        return False

    mat = bpy.data.materials.new("Ground_Themed")
    mat.use_nodes = True
    nt = mat.node_tree

    # Get existing Principled BSDF
    bsdf = nt.nodes.get("Principled BSDF")
    if not bsdf:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")

    # UV scaling
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-800, 0)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-600, 0)
    # Randomise UV scale slightly for variety
    actual_scale = uv_scale * random.uniform(0.8, 1.2)
    mapping.inputs["Scale"].default_value = (actual_scale, actual_scale, 1.0)
    nt.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    # Diffuse map + tint
    if "diff" in texture_maps:
        diff_path = Path(texture_maps["diff"])
        if diff_path.exists():
            tex_d = nt.nodes.new("ShaderNodeTexImage")
            tex_d.location = (-400, 200)
            tex_d.image = bpy.data.images.load(str(diff_path))
            nt.links.new(mapping.outputs["Vector"], tex_d.inputs["Vector"])

            # Tint via MixRGB multiply
            if tint != (1.0, 1.0, 1.0):
                mix = nt.nodes.new("ShaderNodeMixRGB")
                mix.blend_type = "MULTIPLY"
                mix.location = (-150, 200)
                mix.inputs[0].default_value = 1.0  # factor
                mix.inputs[2].default_value = (*tint, 1.0)
                nt.links.new(tex_d.outputs["Color"], mix.inputs[1])
                nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
            else:
                nt.links.new(tex_d.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness map
    if "rough" in texture_maps:
        rough_path = Path(texture_maps["rough"])
        if rough_path.exists():
            tex_r = nt.nodes.new("ShaderNodeTexImage")
            tex_r.location = (-400, -100)
            tex_r.image = bpy.data.images.load(str(rough_path))
            tex_r.image.colorspace_settings.name = "Non-Color"
            nt.links.new(mapping.outputs["Vector"], tex_r.inputs["Vector"])
            nt.links.new(tex_r.outputs["Color"], bsdf.inputs["Roughness"])

    # Normal map
    if "nor_gl" in texture_maps:
        nor_path = Path(texture_maps["nor_gl"])
        if nor_path.exists():
            tex_n = nt.nodes.new("ShaderNodeTexImage")
            tex_n.location = (-400, -400)
            tex_n.image = bpy.data.images.load(str(nor_path))
            tex_n.image.colorspace_settings.name = "Non-Color"
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.location = (-150, -400)
            nt.links.new(mapping.outputs["Vector"], tex_n.inputs["Vector"])
            nt.links.new(tex_n.outputs["Color"], nmap.inputs["Color"])
            nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    # Apply material
    if ground_obj.data.materials:
        ground_obj.data.materials[0] = mat
    else:
        ground_obj.data.materials.append(mat)

    print(f"[env] Applied ground texture ({len(texture_maps)} maps, "
          f"tint={tint}, scale={actual_scale:.2f})")
    return True


# ---------------------------------------------------------------------------
# Themed Domino Materials
# ---------------------------------------------------------------------------

def apply_themed_domino_materials(
    palette: list[list[float]],
    roughness: float = 0.4,
    metallic: float = 0.05,
) -> int:
    """Replace domino materials with theme-specific colours and surface.

    Each domino gets a colour from the palette with slight HSV jitter
    for natural variation. Metallic + roughness come from the theme.

    Returns number of dominos updated.
    """
    dominos = sorted(
        [obj for obj in bpy.data.objects if obj.name.startswith("Domino_")],
        key=lambda o: o.name,
    )
    if not dominos or not palette:
        return 0

    count = 0
    for i, obj in enumerate(dominos):
        base_rgb = palette[i % len(palette)]
        r, g, b = base_rgb[0], base_rgb[1], base_rgb[2]

        # HSV jitter: ±5% hue, ±8% saturation for natural variation
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h = (h + random.uniform(-0.05, 0.05)) % 1.0
        s = max(0, min(1, s + random.uniform(-0.08, 0.08)))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        mat_name = f"DominoTheme_{i % len(palette)}"
        # Reuse material for same palette index (performance)
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
                bsdf.inputs["Roughness"].default_value = roughness + random.uniform(-0.05, 0.05)
                bsdf.inputs["Metallic"].default_value = metallic

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        count += 1

    print(f"[env] Applied themed materials to {count} dominos "
          f"(palette={len(palette)} colours, roughness={roughness:.2f}, "
          f"metallic={metallic:.2f})")
    return count


# ---------------------------------------------------------------------------
# Compositor Post-Processing
# ---------------------------------------------------------------------------

def setup_compositor(compositor_config: dict[str, Any]) -> bool:
    """Set up Blender compositor nodes for post-processing.

    Creates: Glare/Bloom → Color Balance → Vignette → Output.

    Blender 5.x moved compositor to a node group and relocated node
    properties into socket inputs with human-readable enum strings.
    """
    scene = bpy.context.scene

    # --- Obtain the compositor node tree (Blender 5.x vs 4.x) --------
    _is_5x = hasattr(scene, "compositing_node_group")
    if _is_5x:
        if scene.compositing_node_group is None:
            ng = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
            scene.compositing_node_group = ng
        nt = scene.compositing_node_group
    else:
        scene.use_nodes = True
        nt = scene.node_tree

    nt.nodes.clear()

    # --- Helper: map old enum constants to Blender 5.x display names --
    _GLARE_TYPE_MAP = {
        "FOG_GLOW": "Fog Glow",
        "BLOOM": "Bloom",
        "STREAKS": "Streaks",
        "GHOSTS": "Ghosts",
        "SIMPLE_STAR": "Simple Star",
        "SUN_BEAMS": "Sun Beams",
    }
    _QUALITY_MAP = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}
    _CB_TYPE_MAP = {
        "LIFT_GAMMA_GAIN": "Lift/Gamma/Gain",
        "OFFSET_POWER_SLOPE": "Offset/Power/Slope",
    }

    # --- Render Layers input ------------------------------------------
    render = nt.nodes.new("CompositorNodeRLayers")
    render.location = (-400, 0)

    # --- Output node --------------------------------------------------
    if _is_5x:
        out_node = nt.nodes.new("NodeGroupOutput")
    else:
        out_node = nt.nodes.new("CompositorNodeComposite")
    out_node.location = (800, 0)

    # Viewer (optional, 4.x style — skip in 5.x group if unavailable)
    try:
        viewer = nt.nodes.new("CompositorNodeViewer")
        viewer.location = (800, -200)
    except Exception:
        viewer = None

    current_out = render.outputs["Image"]
    x_offset = -100

    # --- 1. Bloom / Glare --------------------------------------------
    bloom_type = compositor_config.get("bloom_type", "NONE")
    if bloom_type != "NONE":
        glare = nt.nodes.new("CompositorNodeGlare")
        glare.location = (x_offset, 0)
        x_offset += 200

        if _is_5x:
            # Blender 5.x: properties exposed as inputs
            display_type = _GLARE_TYPE_MAP.get(bloom_type, bloom_type)
            try:
                glare.inputs["Type"].default_value = display_type
            except Exception:
                pass
            try:
                glare.inputs["Quality"].default_value = "High"
            except Exception:
                pass
            try:
                glare.inputs["Threshold"].default_value = 0.8
            except Exception:
                pass
            try:
                glare.inputs["Strength"].default_value = compositor_config.get(
                    "bloom_mix", 0.3
                )
            except Exception:
                pass
        else:
            # Blender 4.x: direct attributes
            glare.glare_type = bloom_type
            glare.mix = compositor_config.get("bloom_mix", 0.3)
            glare.quality = "HIGH"
            glare.threshold = 0.8

        nt.links.new(current_out, glare.inputs["Image"])
        current_out = glare.outputs["Image"]

    # --- 2. Color Balance (lift/gamma/gain = mood grading) ------------
    lift = compositor_config.get("color_balance_lift", [1.0, 1.0, 1.0])
    gamma = compositor_config.get("color_balance_gamma", [1.0, 1.0, 1.0])
    gain = compositor_config.get("color_balance_gain", [1.0, 1.0, 1.0])

    cb = nt.nodes.new("CompositorNodeColorBalance")
    cb.location = (x_offset, 0)
    x_offset += 200

    if _is_5x:
        # Blender 5.x: Type input → display string, Lift/Gamma/Gain as RGBA inputs
        try:
            cb.inputs["Type"].default_value = _CB_TYPE_MAP.get(
                "LIFT_GAMMA_GAIN", "Lift/Gamma/Gain"
            )
        except Exception:
            pass
        # Lift RGBA (index 4), Gamma RGBA (index 6), Gain RGBA (index 8)
        for idx, vals in [(4, lift), (6, gamma), (8, gain)]:
            try:
                rgba = (*vals, 1.0) if len(vals) == 3 else tuple(vals)
                cb.inputs[idx].default_value = rgba
            except Exception:
                pass
    else:
        cb.correction_method = "LIFT_GAMMA_GAIN"
        cb.lift = (*lift, 1.0) if len(lift) == 3 else tuple(lift)
        cb.gamma = (*gamma, 1.0) if len(gamma) == 3 else tuple(gamma)
        cb.gain = (*gain, 1.0) if len(gain) == 3 else tuple(gain)

    nt.links.new(current_out, cb.inputs["Image"])
    current_out = cb.outputs["Image"]

    # --- 3. Vignette (lens darkening via ellipse mask + blur) ---------
    vignette_strength = compositor_config.get("vignette_strength", 0.0)
    if vignette_strength > 0.0:
        ellipse = nt.nodes.new("CompositorNodeEllipseMask")
        ellipse.location = (x_offset, -300)
        if _is_5x:
            # Blender 5.x: Size input is VECTOR at index 4
            try:
                ellipse.inputs[4].default_value = (0.85, 0.85)
            except Exception:
                pass
        else:
            ellipse.width = 0.85
            ellipse.height = 0.85

        blur = nt.nodes.new("CompositorNodeBlur")
        blur.location = (x_offset + 150, -300)
        if _is_5x:
            # Blender 5.x: Size input is VECTOR2D at index 1
            try:
                blur.inputs[1].default_value = (80.0, 80.0)
            except Exception:
                pass
        else:
            blur.size_x = 80
            blur.size_y = 80

        invert = nt.nodes.new("CompositorNodeInvert")
        invert.location = (x_offset + 300, -300)

        # ShaderNodeMix replaces CompositorNodeMixRGB in Blender 5.x
        if _is_5x:
            mix = nt.nodes.new("ShaderNodeMix")
            mix.location = (x_offset + 200, 0)
            x_offset += 400
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = vignette_strength
            # ShaderNodeMix RGBA inputs: [6]=A, [7]=B
            nt.links.new(ellipse.outputs["Mask"], blur.inputs["Image"])
            nt.links.new(blur.outputs["Image"], invert.inputs["Color"])
            nt.links.new(current_out, mix.inputs[6])
            nt.links.new(invert.outputs["Color"], mix.inputs[7])
            current_out = mix.outputs[2]  # Result (RGBA)
        else:
            mix = nt.nodes.new("CompositorNodeMixRGB")
            mix.location = (x_offset + 200, 0)
            x_offset += 400
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = vignette_strength
            nt.links.new(ellipse.outputs["Mask"], blur.inputs["Image"])
            nt.links.new(blur.outputs["Image"], invert.inputs["Color"])
            nt.links.new(current_out, mix.inputs[1])
            nt.links.new(invert.outputs["Color"], mix.inputs[2])
            current_out = mix.outputs["Image"]

    # --- Final output links -------------------------------------------
    nt.links.new(current_out, out_node.inputs[0])
    if viewer is not None:
        try:
            nt.links.new(current_out, viewer.inputs["Image"])
        except Exception:
            pass

    print(f"[env] Compositor: bloom={bloom_type}, vignette={vignette_strength:.2f}")
    return True


# ---------------------------------------------------------------------------
# Decorative Environment Objects (NON-INTERACTIVE)
# ---------------------------------------------------------------------------

def add_environment_decorations(theme_name: str) -> int:
    """Add decorative background objects based on theme.

    CRITICAL: These are purely visual — NO rigid body physics.
    They use collection instancing and are placed outside the
    domino course bounds to avoid any visual interference.

    Returns number of decorations added.
    """
    # Get course bounds from existing dominos
    dominos = [obj for obj in bpy.data.objects if obj.name.startswith("Domino_")]
    if not dominos:
        return 0

    xs = [d.location.x for d in dominos]
    ys = [d.location.y for d in dominos]
    margin = 30.0  # BU margin outside course

    x_min, x_max = min(xs) - margin, max(xs) + margin
    y_min, y_max = min(ys) - margin, max(ys) + margin

    # Create a "Decorations" collection (for easy toggling)
    if "Decorations" not in bpy.data.collections:
        deco_col = bpy.data.collections.new("Decorations")
        bpy.context.scene.collection.children.link(deco_col)
    else:
        deco_col = bpy.data.collections["Decorations"]

    count = 0
    # Theme-specific decorations (simple geometric shapes, unlit)
    # All objects are placed OUTSIDE course bounds
    deco_configs = _get_decoration_config(theme_name)

    for deco in deco_configs:
        obj = _create_decoration_object(deco, x_min, x_max, y_min, y_max)
        if obj:
            # Move from scene collection to decorations collection
            for col in obj.users_collection:
                col.objects.unlink(obj)
            deco_col.objects.link(obj)

            # CRITICAL: No rigid body physics on decorations
            if obj.rigid_body:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_remove()

            count += 1

    print(f"[env] Added {count} decorative objects (theme={theme_name}, NON-INTERACTIVE)")
    return count


def _get_decoration_config(theme_name: str) -> list[dict[str, Any]]:
    """Return decoration configs per theme."""
    configs: dict[str, list[dict[str, Any]]] = {
        "deep_space": [
            {"type": "icosphere", "count": 5, "size_range": (1.0, 3.0),
             "color": (0.1, 0.1, 0.3), "emission": 0.5, "z_range": (5, 25)},
        ],
        "enchanted_forest": [
            {"type": "cone", "count": 8, "size_range": (2.0, 6.0),
             "color": (0.15, 0.3, 0.1), "emission": 0.0, "z_range": (0, 0)},
        ],
        "golden_hour": [
            {"type": "uv_sphere", "count": 3, "size_range": (1.5, 4.0),
             "color": (0.8, 0.6, 0.3), "emission": 0.0, "z_range": (0, 0)},
        ],
        "neon_city": [
            {"type": "cube", "count": 6, "size_range": (2.0, 8.0),
             "color": (0.2, 0.2, 0.25), "emission": 0.0, "z_range": (0, 0)},
        ],
        "lava_world": [
            {"type": "icosphere", "count": 4, "size_range": (2.0, 5.0),
             "color": (0.3, 0.15, 0.1), "emission": 0.0, "z_range": (0, 0)},
        ],
    }
    return configs.get(theme_name, [])


def _create_decoration_object(
    config: dict[str, Any],
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> bpy.types.Object | None:
    """Create a single decorative object at a random position outside the course."""
    obj_type = config.get("type", "cube")
    count = config.get("count", 1)
    size_range = config.get("size_range", (1.0, 3.0))
    colour = config.get("color", (0.5, 0.5, 0.5))
    emission = config.get("emission", 0.0)
    z_range = config.get("z_range", (0, 0))

    created = None
    for _ in range(count):
        size = random.uniform(*size_range)

        # Place outside the course bounds
        side = random.choice(["left", "right", "front", "back"])
        if side == "left":
            x = x_min - random.uniform(5, 20)
            y = random.uniform(y_min, y_max)
        elif side == "right":
            x = x_max + random.uniform(5, 20)
            y = random.uniform(y_min, y_max)
        elif side == "front":
            x = random.uniform(x_min, x_max)
            y = y_min - random.uniform(5, 20)
        else:
            x = random.uniform(x_min, x_max)
            y = y_max + random.uniform(5, 20)

        z = random.uniform(*z_range) if z_range[1] > z_range[0] else 0

        if obj_type == "cube":
            bpy.ops.mesh.primitive_cube_add(size=size, location=(x, y, z + size / 2))
        elif obj_type == "icosphere":
            bpy.ops.mesh.primitive_ico_sphere_add(radius=size, location=(x, y, z + size))
        elif obj_type == "uv_sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(x, y, z + size))
        elif obj_type == "cone":
            bpy.ops.mesh.primitive_cone_add(radius1=size * 0.4, depth=size * 2,
                                            location=(x, y, z + size))
        else:
            continue

        obj = bpy.context.active_object
        obj.name = f"Deco_{obj_type}_{_:03d}"

        # Apply material (NO rigid body — purely visual)
        mat = bpy.data.materials.new(f"Deco_mat_{obj.name}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.6
            if emission > 0:
                bsdf.inputs["Emission Color"].default_value = (*colour, 1.0)
                bsdf.inputs["Emission Strength"].default_value = emission

        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        created = obj

    return created


def _strip_physics_from_decorations() -> int:
    """Safety sweep: remove rigid body from any non-simulation object.

    Only Domino_*, Ground, and Trigger* objects should have physics.
    Everything else (decorations, lights, camera, environment) must NOT.

    Returns number of objects stripped.
    """
    physics_prefixes = ("Domino_", "Ground", "Trigger")
    stripped = 0
    for obj in bpy.data.objects:
        if obj.rigid_body and not any(obj.name.startswith(p) for p in physics_prefixes):
            try:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.rigidbody.object_remove()
                stripped += 1
                print(f"[env] Stripped physics from: {obj.name}")
            except Exception:
                pass  # Object may not be selectable
    if stripped:
        print(f"[env] Safety sweep: removed physics from {stripped} non-simulation objects")
    return stripped


# ---------------------------------------------------------------------------
# Main: Apply full environment theme
# ---------------------------------------------------------------------------

def apply_environment(theme_config: dict[str, Any]) -> dict[str, Any]:
    """Apply the complete environment theme to the current scene.

    Args:
        theme_config: Theme configuration dict (from ThemeConfig.to_dict()).

    Returns result dict with what was applied.
    """
    result: dict[str, Any] = {
        "theme_name": theme_config.get("theme_name", "unknown"),
        "hdri_applied": False,
        "ground_texture_applied": False,
        "domino_materials_applied": 0,
        "compositor_applied": False,
        "decorations_added": 0,
    }

    # 1. HDRI
    hdri_path = theme_config.get("hdri_path")
    if hdri_path:
        result["hdri_applied"] = apply_hdri(
            hdri_path,
            strength=theme_config.get("hdri_strength", 1.0),
        )

    # 2. Ground texture
    ground = bpy.data.objects.get("Ground")
    texture_maps = theme_config.get("ground_texture_maps")
    if ground and texture_maps:
        tint = tuple(theme_config.get("ground_tint", [1.0, 1.0, 1.0]))
        uv_scale = theme_config.get("ground_uv_scale", 12.0)
        result["ground_texture_applied"] = apply_ground_texture(
            ground, texture_maps, tint=tint, uv_scale=uv_scale,
        )

    # 3. Domino materials
    palette = theme_config.get("domino_palette", [])
    if palette:
        result["domino_materials_applied"] = apply_themed_domino_materials(
            palette,
            roughness=theme_config.get("domino_roughness", 0.4),
            metallic=theme_config.get("domino_metallic", 0.05),
        )

    # 4. Compositor
    compositor = theme_config.get("compositor")
    if compositor:
        try:
            result["compositor_applied"] = setup_compositor(compositor)
        except Exception as exc:
            print(f"[env] WARNING: Compositor setup failed: {exc}")

    # 5. Decorations (NON-INTERACTIVE)
    theme_name = theme_config.get("theme_name", "")
    try:
        result["decorations_added"] = add_environment_decorations(theme_name)
    except Exception as exc:
        print(f"[env] WARNING: Decorations failed: {exc}")

    # 6. SAFETY SWEEP: Ensure NO decoration objects have rigid body physics.
    # Only objects named "Domino_*", "Ground", or "Trigger*" should have physics.
    _strip_physics_from_decorations()

    # Ensure render film is not transparent (we want the HDRI background)
    bpy.context.scene.render.film_transparent = False

    print(f"[env] Environment applied: {json.dumps(result, indent=2)}")
    return result


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
    parser = argparse.ArgumentParser(description="Apply environment theme to Blender scene")
    parser.add_argument("--theme-config", required=True, help="Path to theme config JSON")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.theme_config).read_text(encoding="utf-8"))
    result = apply_environment(config)

    # Save modified scene
    bpy.ops.wm.save_mainfile()
    print(f"[env] Saved themed scene: {bpy.data.filepath}")

    print(json.dumps(result, indent=2))
