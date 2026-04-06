"""Bake rigid-body physics state as per-frame animation keyframes.

Steps the physics simulation frame-by-frame, captures the evaluated world
transforms at each target frame, inserts them as location/rotation keyframes,
then disables the rigid-body world and saves a copy of the .blend.

The resulting file has NO active physics simulation — every domino position is
encoded as a plain animation keyframe.  This means the file can be rendered
correctly from Blender CLI ``--render-frame N`` without needing a display
context or a live physics evaluation.

Usage (called from within Blender):
    blender --background <file.blend> --python bake_physics_keyframes.py -- \\
        <output_blend> <frame1,frame2,...>

Example:
    blender --background domino.blend --python bake_physics_keyframes.py -- \\
        /tmp/baked.blend 1,50,100,200,300
"""

from __future__ import annotations

import sys

import bpy  # type: ignore[import-untyped]
from mathutils import Matrix  # type: ignore[import-untyped]


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    if len(argv) < 1:
        print("[bake_kf] ERROR: pass <output_blend> [frame1,frame2,...]")
        sys.exit(1)

    out_blend = argv[0]
    if len(argv) > 1:
        target_frames = sorted(set(int(f.strip()) for f in argv[1].split(",") if f.strip()))
    else:
        target_frames = [1, 50, 100, 150, 200, 300]

    scene = bpy.context.scene
    rbw = scene.rigidbody_world

    active_rbs = [
        obj for obj in scene.objects
        if obj.rigid_body and obj.rigid_body.type == "ACTIVE"
    ]
    print(f"[bake_kf] {len(active_rbs)} active rigid bodies | frames: {target_frames}")

    if not active_rbs:
        print("[bake_kf] No active rigid bodies found — saving unmodified blend.")
        bpy.ops.wm.save_as_mainfile(filepath=out_blend, copy=True)
        return

    target_set = set(target_frames)
    max_frame = max(target_frames)

    # ── Step physics and insert keyframes ────────────────────────────
    scene.frame_set(scene.frame_start)
    for f in range(scene.frame_start, max_frame + 1):
        scene.frame_set(f)
        if f not in target_set:
            continue

        dg = bpy.context.evaluated_depsgraph_get()
        print(f"[bake_kf] Keyframing {len(active_rbs)} objects at frame {f}...")
        for obj in active_rbs:
            obj_eval = obj.evaluated_get(dg)
            mw: Matrix = obj_eval.matrix_world.copy()
            loc, rot, scale = mw.decompose()
            obj.location = loc
            obj.rotation_euler = rot.to_euler()
            obj.scale = scale
            obj.keyframe_insert(data_path="location", frame=f)
            obj.keyframe_insert(data_path="rotation_euler", frame=f)

    print(f"[bake_kf] Keyframing complete.")

    # ── Disable physics world so keyframes drive transforms ──────────
    if rbw:
        rbw.enabled = False
        print("[bake_kf] Rigid body world disabled — keyframes now control positions.")

    # ── Save copy ────────────────────────────────────────────────────
    bpy.ops.wm.save_as_mainfile(filepath=out_blend, copy=True)
    print(f"[bake_kf] Saved baked .blend → {out_blend}")


if __name__ == "__main__":
    main()
