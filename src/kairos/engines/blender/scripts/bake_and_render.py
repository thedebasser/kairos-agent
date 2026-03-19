"""Bake rigid-body simulation and render to video.

Steps:
  1. Bake the rigid-body simulation (all frames)
  2. Apply render settings from preset
  3. Render animation to output path

Run via:
    blender --background <file.blend> --python bake_and_render.py -- --output <path.mp4> [--preset production|preview]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]


def apply_render_preset(preset_name: str = "render") -> None:
    """Apply render settings from presets config."""
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    presets_data = json.loads((configs_dir / "render_presets.json").read_text())
    preset = presets_data.get(preset_name, presets_data.get("render", {}))

    scene = bpy.context.scene

    # Engine
    engine = preset.get("engine", "CYCLES")
    engine_map = {
        "EEVEE": "BLENDER_EEVEE",
        "CYCLES": "CYCLES",
    }
    scene.render.engine = engine_map.get(engine, engine)

    # Cycles settings
    if engine == "CYCLES":
        scene.cycles.samples = preset.get("samples", 128)
        scene.cycles.use_denoising = preset.get("use_denoising", True)
        # Try GPU
        device = preset.get("device", "GPU")
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "CUDA"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = True
            scene.cycles.device = device
        except Exception:
            scene.cycles.device = "CPU"
    elif engine == "EEVEE":
        # Blender 5.0: taa_render_samples was REMOVED.
        # Use scene.eevee.samples (unified sample count).
        try:
            scene.eevee.samples = preset.get("samples", 32)
        except AttributeError:
            # Fallback for older Blender versions
            try:
                scene.eevee.taa_render_samples = preset.get("samples", 32)
            except AttributeError:
                pass
        # Blender 5.0 shadow settings (ray-traced shadows replaced
        # the old shadow map system).
        try:
            scene.eevee.use_shadows = True
            scene.eevee.shadow_ray_count = 2
            scene.eevee.shadow_step_count = 8
        except AttributeError:
            pass  # Property names may differ across Blender versions
        # Make sure film is not transparent (we want the world background)
        scene.render.film_transparent = False

    # Resolution
    scene.render.resolution_x = preset.get("resolution_x", 1080)
    scene.render.resolution_y = preset.get("resolution_y", 1920)
    scene.render.fps = preset.get("fps", 30)
    scene.render.resolution_percentage = 100

    # Ensure camera far-clip is sufficient for WORLD_SCALE geometry
    cam = scene.camera
    if cam and cam.data:
        cam.data.clip_end = max(cam.data.clip_end, 300)

    # Output format — Blender 5.0 only supports image formats;
    # video encoding is handled externally via ffmpeg.
    fmt = preset.get("output_format", "PNG")
    if fmt == "FFMPEG":
        fmt = "PNG"  # Fall back to PNG frames for Blender 5.0
    scene.render.image_settings.file_format = fmt
    if fmt == "PNG":
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.compression = 15


def bake_simulation() -> float:
    """Free existing cache and bake the rigid-body simulation.

    Returns bake time in seconds.
    """
    scene = bpy.context.scene
    rbw = scene.rigidbody_world
    if rbw is None:
        raise RuntimeError("No rigid body world to bake")

    # Free existing bake
    override = bpy.context.copy()
    override["point_cache"] = rbw.point_cache

    with bpy.context.temp_override(**override):
        bpy.ops.ptcache.free_bake_all()

    start = time.monotonic()
    with bpy.context.temp_override(**override):
        bpy.ops.ptcache.bake_all(bake=True)
    bake_time = time.monotonic() - start

    return bake_time


def render_single_frame_test(output_path: str, frame: int = 1) -> Path:
    """Render a single frame to validate settings before committing to a
    full animation render.

    Returns the path to the rendered test image.
    Raises RuntimeError if the output file is not created or is empty.
    """
    scene = bpy.context.scene
    output = Path(output_path).resolve()
    test_dir = output.parent / "test_frame"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_path = test_dir / f"test_{frame:04d}.png"
    scene.render.filepath = str(test_path)
    scene.frame_set(frame)

    bpy.ops.render.render(write_still=True)

    if not test_path.exists() or test_path.stat().st_size < 1024:
        raise RuntimeError(
            f"Single-frame render test failed: {test_path} not created or empty"
        )
    print(f"[render] Single-frame test OK: {test_path} ({test_path.stat().st_size} bytes)")
    return test_path


def render_animation(output_path: str) -> float:
    """Render the full animation as PNG frames.

    Frames are written to a subfolder next to *output_path*.
    The pipeline will combine them with ffmpeg externally.

    Returns render time in seconds.
    """
    scene = bpy.context.scene
    output = Path(output_path).resolve()
    frames_dir = output.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    scene.render.filepath = str(frames_dir / "frame_")

    start = time.monotonic()
    bpy.ops.render.render(animation=True)
    render_time = time.monotonic() - start

    return render_time


def apply_physics_camera(config_path: str | None = None) -> dict[str, Any]:
    """Apply physics-aware camera tracking after rigid-body bake.

    Imports and runs the physics_camera module, which scans the baked
    simulation to detect the actual wave front and creates smooth
    camera keyframes following the action.
    """
    # Import physics_camera from the same scripts directory
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from physics_camera import apply_physics_camera as _apply

    config: dict[str, Any] = {}
    if config_path:
        p = Path(config_path)
        if p.exists():
            config = json.loads(p.read_text(encoding="utf-8"))

    start = time.monotonic()
    result = _apply(config)
    elapsed = time.monotonic() - start
    result["camera_compute_time_sec"] = round(elapsed, 2)

    print(f"[bake_and_render] Physics camera: {result.get('status')} "
          f"({elapsed:.1f}s)")
    return result


def apply_environment_theme(config_path: str | None = None) -> dict[str, Any]:
    """Apply environment theme if theme_config.json exists in the work dir.

    This applies HDRI, ground textures, domino materials, and compositor
    AFTER physics bake (so baked data is not affected) and BEFORE render.
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from setup_environment import apply_environment

    # Look for theme_config.json alongside config.json
    theme_config_path = None
    if config_path:
        theme_config_path = Path(config_path).parent / "theme_config.json"
    if not theme_config_path or not theme_config_path.exists():
        print("[bake_and_render] No theme_config.json found — skipping environment")
        return {"status": "skipped"}

    try:
        theme_config = json.loads(theme_config_path.read_text(encoding="utf-8"))
        start = time.monotonic()
        result = apply_environment(theme_config)
        elapsed = time.monotonic() - start
        result["env_apply_time_sec"] = round(elapsed, 2)
        print(f"[bake_and_render] Environment theme applied: "
              f"{theme_config.get('theme_name', 'unknown')} ({elapsed:.1f}s)")
        return result
    except Exception as exc:
        print(f"[bake_and_render] Environment theme failed (non-fatal): {exc}")
        return {"status": "error", "error": str(exc)}


def export_tip_frames(config_path: str | None = None) -> dict[str, int]:
    """Export domino tip_frames from the baked simulation for SFX timing.

    Reuses physics_camera's scan_tip_frames function.
    Saves tip_frames.json alongside the config.
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from physics_camera import scan_tip_frames

    scene = bpy.context.scene
    dominos = sorted(
        [obj for obj in bpy.data.objects if obj.name.startswith("Domino_")],
        key=lambda o: o.name,
    )
    if len(dominos) < 2:
        return {}

    tip_frames = scan_tip_frames(dominos, scene.frame_start, scene.frame_end)

    # Save to disk for post-processing
    if config_path:
        tip_path = Path(config_path).parent / "tip_frames.json"
        tip_path.write_text(
            json.dumps(tip_frames, indent=2),
            encoding="utf-8",
        )
        print(f"[bake_and_render] Exported {len(tip_frames)} tip_frames -> {tip_path}")

    return tip_frames


def add_collision_speakers(config_path: str | None = None) -> dict[str, Any]:
    """Place Blender Speaker objects at each domino and mixdown to WAV.

    Uses the tip_frames exported by export_tip_frames() to create a
    Speaker at every domino's position, keyframed to fire at tip time
    with randomised pitch for natural variation.

    Returns dict with status and path to the mixed-down WAV.
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from add_collision_speakers import add_collision_audio

    if not config_path:
        print("[bake_and_render] No config_path — skipping collision speakers")
        return {"status": "skipped"}

    work_dir = Path(config_path).parent
    tip_frames_path = work_dir / "tip_frames.json"
    sfx_dir = work_dir / "speaker_sfx"
    output_wav = work_dir / "collision_audio.wav"

    if not tip_frames_path.exists():
        print("[bake_and_render] No tip_frames.json — skipping collision speakers")
        return {"status": "no_tip_frames"}

    start = time.monotonic()
    result = add_collision_audio(
        str(tip_frames_path),
        str(sfx_dir),
        str(output_wav),
        num_variations=24,
    )
    elapsed = time.monotonic() - start
    result["speaker_time_sec"] = round(elapsed, 2)

    print(f"[bake_and_render] Collision speakers: {result.get('status')} "
          f"({result.get('speakers_placed', 0)} speakers, {elapsed:.1f}s)")
    return result


def generate_ambient_bed_track(config_path: str | None = None) -> dict[str, Any]:
    """Generate a theme-matched ambient bed WAV for the rendered video.

    Uses ambient_bed.py to synthesise a room-tone + environmental texture
    track that plays underneath the collision audio.
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from ambient_bed import generate_ambient_bed

    if not config_path:
        print("[bake_and_render] No config_path — skipping ambient bed")
        return {"status": "skipped"}

    work_dir = Path(config_path).parent
    output_wav = work_dir / "ambient_bed.wav"

    # Read theme from theme_config.json
    theme_name = None
    theme_config_path = work_dir / "theme_config.json"
    if theme_config_path.exists():
        try:
            tc = json.loads(theme_config_path.read_text(encoding="utf-8"))
            theme_name = tc.get("theme_name")
        except Exception:
            pass

    # Determine duration from Blender scene
    scene = bpy.context.scene
    fps = scene.render.fps
    total_frames = scene.frame_end - scene.frame_start + 1
    duration_sec = total_frames / fps

    start = time.monotonic()
    try:
        generate_ambient_bed(
            output_wav=str(output_wav),
            duration_sec=duration_sec,
            theme_name=theme_name,
        )
        elapsed = time.monotonic() - start
        size = output_wav.stat().st_size if output_wav.exists() else 0
        print(f"[bake_and_render] Ambient bed: {output_wav.name} "
              f"({size:,} bytes, {elapsed:.1f}s)")
        return {
            "status": "ok",
            "wav_path": str(output_wav),
            "duration_sec": round(duration_sec, 2),
            "theme": theme_name,
            "time_sec": round(elapsed, 2),
        }
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"[bake_and_render] Ambient bed failed (non-fatal): {exc}")
        return {"status": "error", "error": str(exc), "time_sec": round(elapsed, 2)}


def bake_and_render(
    output_path: str,
    preset: str = "render",
    config_path: str | None = None,
) -> dict[str, Any]:
    """Full pipeline: apply preset, bake, compute camera, apply env, render.

    Returns dict with timings and output info.
    """
    apply_render_preset(preset)

    bake_time = bake_simulation()

    # --- Physics-aware camera (post-bake) ---
    # Recompute camera keyframes using actual baked rigid-body data.
    camera_result = apply_physics_camera(config_path)

    # --- Export tip_frames for SFX timing ---
    tip_frames = export_tip_frames(config_path)

    # --- Collision speakers (post-bake, post-tip-frames) ---
    # Place Speaker objects at each domino, keyframed to fire at tip time.
    # Blender's audio engine handles spatial mixing → single WAV mixdown.
    speaker_result = add_collision_speakers(config_path)

    # --- Ambient bed DISABLED ---
    # Removed: the continuous room-tone sine wave was being amplified ~30 dB
    # by FFmpeg loudnorm in quiet sections, creating audible background noise.
    # See docs/white-noise-diagnosis.md for full analysis.
    # ambient_result = generate_ambient_bed_track(config_path)

    # --- Apply environment theme (post-bake, pre-render) ---
    # This applies HDRI, ground texture, domino materials, compositor.
    # Done AFTER bake so physics are not affected.
    env_result = apply_environment_theme(config_path)

    # Save the .blend with the updated camera + environment before rendering
    bpy.ops.wm.save_mainfile()

    # Single-frame render test — catch blank/broken renders before
    # committing to 1950+ frames.
    test_frame_path = render_single_frame_test(output_path, frame=1)

    render_time = render_animation(output_path)

    scene = bpy.context.scene
    total_frames = scene.frame_end - scene.frame_start + 1

    return {
        "output_path": output_path,
        "frames_dir": str(Path(output_path).resolve().parent / "frames"),
        "preset": preset,
        "bake_time_sec": round(bake_time, 2),
        "render_time_sec": round(render_time, 2),
        "camera_result": camera_result,
        "speaker_result": speaker_result,
        "ambient_result": None,  # ambient bed disabled — see docs/white-noise-diagnosis.md
        "env_result": env_result,
        "tip_frames_count": len(tip_frames),
        "total_frames": total_frames,
        "resolution": f"{scene.render.resolution_x}x{scene.render.resolution_y}",
        "fps": scene.render.fps,
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
    parser.add_argument("--output", required=True, help="Output video path (e.g. renders/video.mp4)")
    parser.add_argument("--preset", default="render", choices=["render", "render_preview"])
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--bake-only", action="store_true", help="Only bake, skip render")
    args = parser.parse_args(argv)

    if args.bake_only:
        apply_render_preset(args.preset)
        t = bake_simulation()
        print(json.dumps({"bake_time_sec": round(t, 2)}))
    else:
        result = bake_and_render(args.output, args.preset, config_path=args.config)
        print(json.dumps(result, indent=2))
