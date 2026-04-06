"""Render key calibration frames from a baked .blend file.

Designed for fast EEVEE preview rendering of physics simulations.
Uses the existing baked rigid-body cache in the .blend — no re-baking.
Resolution: 540×960 (half vertical-short), 16 EEVEE samples.

Run via:
    blender --background <file.blend> --python render_calibration_frames.py -- \\
        --auto 8 --output-dir /path/to/frames --output-json /path/result.json

Or with explicit frame numbers:
    blender --background <file.blend> --python render_calibration_frames.py -- \\
        --frames 1,200,400,600,900 --output-dir /path/to/frames --output-json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]


def _set_eevee_preview() -> None:
    """Configure EEVEE at low resolution for fast preview renders."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 540
    scene.render.resolution_y = 960
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15
    scene.render.use_motion_blur = False

    try:
        scene.eevee.samples = 16
    except AttributeError:
        pass

    # Ensure camera far-clip covers large WORLD_SCALE geometry (×10 units)
    cam = scene.camera
    if cam and cam.data:
        cam.data.clip_end = max(cam.data.clip_end, 500)


def _auto_frame_numbers(total_frames: int, count: int = 8) -> list[int]:
    """Compute evenly-distributed frame numbers spanning the simulation.

    Always includes frame 1 (rest state before physics fires) and the
    last frame (final state of the chain).
    """
    if count <= 1:
        return [1]
    if count == 2:
        return [1, total_frames]

    step = (total_frames - 1) / (count - 1)
    frames = [max(1, round(1 + i * step)) for i in range(count)]
    frames[-1] = total_frames  # guarantee last frame is included
    return sorted(set(frames))


def _physics_sample_end(scene: "bpy.types.Scene", sample_end_override: int | None) -> int:
    """Determine the last frame to use when computing auto sample points.

    The raw scene.frame_end is often 1950 (Blender default) even when the
    entire domino cascade finishes within the first 200-400 frames.  Sampling
    evenly across the full 1950 frames means most target frames fall in the
    'all fallen, static' region, making the VLM unable to see the cascade.

    Priority order:
    1. Explicit --sample-end CLI override.
    2. ptcache bake end frame (if smaller than scene.frame_end and baked).
    3. scene.frame_end capped at 500 as a safe default.
    """
    rbw = scene.rigidbody_world
    cap = sample_end_override if sample_end_override and sample_end_override > 0 else 500
    if rbw:
        pc_end = rbw.point_cache.frame_end
        if pc_end < scene.frame_end:
            cap = min(cap, pc_end)
    return min(scene.frame_end, cap)


def _bake_physics(up_to_frame: int) -> None:
    """Ensure rigid-body physics is evaluated up to *up_to_frame*.

    The smoke test uses in-memory frame-stepping (no persistent ptcache)
    so the bake doesn't survive across Blender processes.  We mirror
    the same approach here so the render sees the correct physics state.
    """
    scene = bpy.context.scene
    rbw = scene.rigidbody_world
    if rbw is None:
        print("[render_calibration] No rigidbody world — skipping bake")
        return

    end_frame = min(up_to_frame, scene.frame_end)
    # Try the proper ptcache bake API first (creates persistent cache files)
    try:
        original_end = rbw.point_cache.frame_end
        rbw.point_cache.frame_end = end_frame
        override = bpy.context.copy()
        override["point_cache"] = rbw.point_cache
        with bpy.context.temp_override(**override):
            bpy.ops.ptcache.free_bake_all()
            bpy.ops.ptcache.bake_all(bake=True)
        rbw.point_cache.frame_end = original_end
        print(f"[render_calibration] Baked via ptcache API up to frame {end_frame}")
        return
    except Exception as exc:
        print(f"[render_calibration] ptcache bake failed ({exc}), using frame-stepping...")

    # Fallback: step frame-by-frame to evaluate physics in memory
    scene.frame_set(scene.frame_start)
    for f in range(scene.frame_start, end_frame + 1):
        scene.frame_set(f)
    print(f"[render_calibration] Frame-stepped physics to frame {end_frame}")


def render_frames(frame_numbers: list[int], output_dir: Path) -> list[dict[str, Any]]:
    """Render specific frames to PNG and return per-frame metadata.

    Steps through the simulation sequentially so that rigid-body physics
    is properly evaluated at each frame (Blender's ptcache bake may not
    persist across processes, so we re-step in-order here).
    Frames that are in *frame_numbers* are rendered when reached.
    """
    scene = bpy.context.scene
    output_dir.mkdir(parents=True, exist_ok=True)

    rbw = scene.rigidbody_world
    targets = sorted(set(
        max(scene.frame_start, min(f, scene.frame_end))
        for f in frame_numbers
    ))
    max_target = targets[-1]

    results: list[dict[str, Any]] = []
    results_map: dict[int, dict[str, Any]] = {}

    def _render_at(frame_num: int) -> None:
        out_path = output_dir / f"frame_{frame_num:05d}.png"
        scene.render.filepath = str(out_path)
        try:
            bpy.ops.render.render(write_still=True)
            size = out_path.stat().st_size if out_path.exists() else 0
            ok = size > 1024
            entry = {"frame": frame_num, "path": str(out_path), "size_bytes": size, "ok": ok}
            print(f"[render_calibration] Frame {frame_num} → {out_path.name} ({size:,} bytes)")
        except Exception as exc:
            print(f"[render_calibration] Frame {frame_num} failed: {exc}")
            entry = {"frame": frame_num, "path": str(out_path), "size_bytes": 0, "ok": False,
                     "error": str(exc)}
        results_map[frame_num] = entry

    if rbw and not rbw.point_cache.is_baked:
        # No persistent cache — step frame-by-frame and capture at target frames.
        # This is the only reliable way to get correct physics state in background mode.
        print(f"[render_calibration] Stepping physics and rendering at frames: {targets}")
        target_set = set(targets)
        scene.frame_set(scene.frame_start)
        for f in range(scene.frame_start, max_target + 1):
            scene.frame_set(f)
            if f in target_set:
                _render_at(f)
    else:
        # ptcache bake is loaded — can jump directly to any frame
        print(f"[render_calibration] ptcache available — jumping to frames: {targets}")
        for frame_num in targets:
            scene.frame_set(frame_num)
            _render_at(frame_num)

    # Return in original frame order
    for f in targets:
        if f in results_map:
            results.append(results_map[f])

    return results


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    import argparse
    parser = argparse.ArgumentParser(description="Render calibration frames from baked .blend")
    parser.add_argument("--frames", default="",
                        help="Comma-separated frame numbers (e.g. 1,200,400)")
    parser.add_argument("--auto", type=int, default=8,
                        help="Auto-select N evenly-distributed frames (default: 8)")
    parser.add_argument("--sample-end", type=int, default=0,
                        help="Cap the auto-frame sampling window at this scene frame (default: auto-detect)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write PNG frames")
    parser.add_argument("--output-json", required=True,
                        help="Path to write result JSON")
    args = parser.parse_args(argv)

    scene = bpy.context.scene

    if args.frames:
        frame_numbers = [int(f.strip()) for f in args.frames.split(",") if f.strip()]
    else:
        sample_end = _physics_sample_end(scene, args.sample_end if args.sample_end else None)
        total_frames = sample_end - scene.frame_start + 1
        frame_numbers = _auto_frame_numbers(total_frames, args.auto)

    print(f"[render_calibration] Scene has {total_frames} frames, rendering: {frame_numbers}")

    _set_eevee_preview()
    rendered = render_frames(frame_numbers, Path(args.output_dir))

    result = {
        "total_scene_frames": total_frames,
        "frames_requested": frame_numbers,
        "frames_rendered": rendered,
        "ok_count": sum(1 for r in rendered if r["ok"]),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print JSON so the executor can parse it from stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
