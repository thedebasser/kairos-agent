"""Validate a domino course .blend file.

Structural checks:
  1. Ground plane exists
  2. Domino objects exist and have rigid bodies
  3. Trigger object exists
  4. Camera exists
  5. Rigid body world is configured

Run via:
    blender --background <file.blend> --python validate_domino_course.py [-- --strict]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # type: ignore[import-untyped]


def validate() -> dict:
    """Run all validation checks and return result dict."""
    checks = []

    # 1. Ground plane
    ground = bpy.data.objects.get("Ground")
    checks.append({
        "name": "ground_exists",
        "passed": ground is not None,
        "message": "Ground plane found" if ground else "Ground plane missing",
    })

    # 2. Domino objects
    dominos = [o for o in bpy.data.objects if o.name.startswith("Domino_")]
    has_dominos = len(dominos) >= 10
    checks.append({
        "name": "domino_count",
        "passed": has_dominos,
        "message": f"Found {len(dominos)} dominos" + ("" if has_dominos else " (need >=10)"),
    })

    # 3. Rigid bodies on dominos
    rb_count = sum(1 for d in dominos if d.rigid_body is not None)
    all_rb = rb_count == len(dominos) and len(dominos) > 0
    checks.append({
        "name": "domino_rigid_bodies",
        "passed": all_rb,
        "message": f"{rb_count}/{len(dominos)} dominos have rigid bodies",
    })

    # 4. Trigger object
    trigger = bpy.data.objects.get("Trigger_Pusher")
    checks.append({
        "name": "trigger_exists",
        "passed": trigger is not None,
        "message": "Trigger found" if trigger else "Trigger_Pusher missing",
    })

    # 5. Camera
    cam = bpy.context.scene.camera
    checks.append({
        "name": "camera_exists",
        "passed": cam is not None,
        "message": "Camera found" if cam else "No active camera",
    })

    # 6. Rigid body world
    rbw = bpy.context.scene.rigidbody_world
    checks.append({
        "name": "rigid_body_world",
        "passed": rbw is not None,
        "message": "RB world configured" if rbw else "No rigid body world",
    })

    # 7. Frame range
    scene = bpy.context.scene
    frame_count = scene.frame_end - scene.frame_start + 1
    good_frames = frame_count >= 1800  # ~60s at 30fps
    checks.append({
        "name": "frame_range",
        "passed": good_frames,
        "message": f"Frame range: {scene.frame_start}-{scene.frame_end} ({frame_count} frames)",
    })

    all_passed = all(c["passed"] for c in checks)

    return {
        "passed": all_passed,
        "summary": f"{'PASSED' if all_passed else 'FAILED'}: "
                   f"{sum(c['passed'] for c in checks)}/{len(checks)} checks",
        "checks": checks,
    }


def main() -> None:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    strict = "--strict" in argv

    result = validate()

    print(json.dumps(result, indent=2))

    if strict and not result["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
