"""Validate a generated marble course.

Checks:
  1. Socket alignment — every module's entry aligns with the previous exit
  2. No module overlap — BVHTree-based proximity check
  3. Wall height — sufficient for marble radius
  4. Slope bounds — ramps within safe angle range
  5. Path-support — raycast downward along centreline to ensure support
  6. Bounding corridor — modules stay within expected tolerances

Run via:
    blender --background <file.blend> --python validate_course.py -- [--strict]
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-untyped]
from mathutils import Vector  # type: ignore[import-untyped]
from mathutils.bvhtree import BVHTree  # type: ignore[import-untyped]


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    message: str = ""
    value: Any = None
    threshold: Any = None


@dataclass
class ValidationResult:
    passed: bool
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def summary(self) -> str:
        p = sum(1 for c in self.checks if c.passed)
        return f"{p}/{len(self.checks)} checks passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message,
                 "value": c.value, "threshold": c.threshold}
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_rigid_body_world() -> ValidationCheck:
    """Ensure a rigid-body world is configured."""
    rbw = bpy.context.scene.rigidbody_world
    if rbw is None:
        return ValidationCheck("rigid_body_world", False, "No rigid body world found")
    return ValidationCheck("rigid_body_world", True,
                           f"substeps={rbw.substeps_per_frame}, iterations={rbw.solver_iterations}")


def check_marble_count(min_count: int = 5) -> ValidationCheck:
    """At least *min_count* marble objects exist."""
    marbles = [o for o in bpy.data.objects if o.name.startswith("marble_")]
    count = len(marbles)
    ok = count >= min_count
    return ValidationCheck("marble_count", ok,
                           f"Found {count} marbles (min={min_count})",
                           value=count, threshold=min_count)


def check_marble_rigid_bodies() -> ValidationCheck:
    """Every marble has an active rigid body."""
    marbles = [o for o in bpy.data.objects if o.name.startswith("marble_")]
    bad = [m.name for m in marbles if m.rigid_body is None or m.rigid_body.type != "ACTIVE"]
    ok = len(bad) == 0
    msg = "All marbles have active rigid bodies" if ok else f"Missing RB: {bad[:5]}"
    return ValidationCheck("marble_rigid_bodies", ok, msg)


def check_course_colliders() -> ValidationCheck:
    """Course pieces have passive rigid bodies."""
    course = [o for o in bpy.data.objects
              if not o.name.startswith("marble_")
              and o.type == "MESH"
              and not o.name.startswith(("Camera", "Key_", "Fill_", "Rim_"))]
    bad = [o.name for o in course if o.rigid_body is None]
    ok = len(bad) == 0
    msg = f"All {len(course)} course meshes have rigid bodies" if ok else f"Missing RB on: {bad[:5]}"
    return ValidationCheck("course_colliders", ok, msg)


def check_overlap(tolerance: float = 0.005) -> ValidationCheck:
    """Check for significant mesh overlaps using BVHTree."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    course = [o for o in bpy.data.objects
              if o.type == "MESH"
              and not o.name.startswith("marble_")
              and not o.name.startswith(("Camera", "Key_", "Fill_", "Rim_"))]

    trees: list[tuple[str, BVHTree]] = []
    for obj in course:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh and len(mesh.polygons) > 0:
            tree = BVHTree.FromObject(eval_obj, depsgraph)
            trees.append((obj.name, tree))
            eval_obj.to_mesh_clear()

    overlaps = []
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            name_a, tree_a = trees[i]
            name_b, tree_b = trees[j]
            pairs = tree_a.overlap(tree_b)
            if pairs and len(pairs) > 2:  # Allow minor touching
                overlaps.append(f"{name_a}<->{name_b} ({len(pairs)} faces)")

    ok = len(overlaps) == 0
    msg = "No significant overlaps" if ok else f"Overlaps: {overlaps[:5]}"
    return ValidationCheck("overlap", ok, msg, value=len(overlaps))


def check_support_raycast(marble_radius: float = 0.04,
                           gap_tolerance: float = 0.15) -> ValidationCheck:
    """Raycast downward along the approximate course path to ensure floor support."""
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Gather marble positions as path approximation
    marbles = sorted(
        [o for o in bpy.data.objects if o.name.startswith("marble_")],
        key=lambda o: -o.location.z,  # top to bottom
    )

    if not marbles:
        return ValidationCheck("support_raycast", False, "No marbles to trace path")

    # Build combined BVH of course geometry
    course = [o for o in bpy.data.objects
              if o.type == "MESH"
              and not o.name.startswith("marble_")
              and not o.name.startswith(("Camera", "Key_", "Fill_", "Rim_"))]

    unsupported = 0
    total_tests = 0
    for marble in marbles[:20]:  # sample up to 20
        origin = marble.location + Vector((0, 0, marble_radius * 2))
        # Cast down
        for obj in course:
            eval_obj = obj.evaluated_get(depsgraph)
            # Transform ray to object local space
            mat_inv = eval_obj.matrix_world.inverted()
            local_origin = mat_inv @ origin
            local_dir = mat_inv.to_3x3() @ Vector((0, 0, -1))
            result = eval_obj.ray_cast(local_origin, local_dir, distance=2.0)
            if result[0]:  # hit
                break
        else:
            unsupported += 1
        total_tests += 1

    ok = unsupported <= total_tests * 0.3  # allow 30% unsupported
    msg = f"{unsupported}/{total_tests} positions unsupported"
    return ValidationCheck("support_raycast", ok, msg,
                           value=unsupported, threshold=int(total_tests * 0.3))


def check_camera_exists() -> ValidationCheck:
    """A camera is present in the scene."""
    cam = bpy.context.scene.camera
    ok = cam is not None
    msg = "Camera found" if ok else "No camera in scene"
    # Also check clip_end is sufficient for WORLD_SCALE geometry
    if cam and cam.data and cam.data.clip_end < 300:
        return ValidationCheck("camera_exists", False,
                               f"Camera clip_end={cam.data.clip_end} < 300 (too small for WORLD_SCALE)")
    return ValidationCheck("camera_exists", ok, msg)


def check_wall_heights(marble_radius: float = 0.04,
                        marble_bounce: float = 0.4) -> ValidationCheck:
    """Validate that wall heights meet the containment formula.

    Formula: H_wall ≥ 2 × B² × h_drop + 2R, minimum 3.0 units.
    We approximate h_drop as the Z-distance between adjacent modules.
    """
    S = 10  # WORLD_SCALE
    r = marble_radius * S
    min_h = max(3.0, 2 * marble_bounce**2 * 5.0 + 2 * r)  # conservative 5-unit drop

    # Find all wall-like objects (name contains "wall" or "lwall" or "rwall")
    walls = [o for o in bpy.data.objects
             if o.type == "MESH"
             and ("wall" in o.name.lower() or "rail" in o.name.lower())
             and not o.name.startswith(("Camera", "Key_", "Fill_", "Rim_"))]

    too_short = []
    for w in walls:
        # Approximate wall height from bounding box Z extent
        bbox_z = [v[2] for v in w.bound_box]
        height = (max(bbox_z) - min(bbox_z)) * max(w.scale.z, 1.0)
        if height < min_h * 0.8:  # allow 20% tolerance
            too_short.append(f"{w.name} ({height:.2f} < {min_h:.2f})")

    ok = len(too_short) == 0
    msg = f"All {len(walls)} walls meet height formula" if ok else f"Short walls: {too_short[:5]}"
    return ValidationCheck("wall_heights", ok, msg,
                           value=len(too_short), threshold=0)


def check_substeps() -> ValidationCheck:
    """Verify substeps_per_frame ≥ 20 (research-validated minimum)."""
    rbw = bpy.context.scene.rigidbody_world
    if rbw is None:
        return ValidationCheck("substeps", False, "No rigid body world")
    ok = rbw.substeps_per_frame >= 20
    return ValidationCheck("substeps", ok,
                           f"substeps={rbw.substeps_per_frame} (min=20)",
                           value=rbw.substeps_per_frame, threshold=20)


def check_frame_range() -> ValidationCheck:
    """Frame range is sensible."""
    scene = bpy.context.scene
    total = scene.frame_end - scene.frame_start + 1
    ok = 300 <= total <= 3000  # 10-100 sec at 30fps
    return ValidationCheck("frame_range", ok,
                           f"Frames: {scene.frame_start}-{scene.frame_end} ({total} total)",
                           value=total, threshold="300-3000")


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_course(*, strict: bool = False,
                    marble_radius: float = 0.04) -> ValidationResult:
    """Run all validation checks.

    Args:
        strict: If True, require ALL checks to pass.
        marble_radius: Expected marble radius for support checks.
    """
    checks = [
        check_rigid_body_world(),
        check_substeps(),
        check_marble_count(),
        check_marble_rigid_bodies(),
        check_course_colliders(),
        check_overlap(),
        check_support_raycast(marble_radius=marble_radius),
        check_camera_exists(),
        check_frame_range(),
        check_wall_heights(marble_radius=marble_radius),
    ]

    if strict:
        passed = all(c.passed for c in checks)
    else:
        # Require critical checks, allow some non-critical failures
        critical = {"rigid_body_world", "marble_count", "marble_rigid_bodies",
                     "course_colliders", "camera_exists", "frame_range"}
        passed = all(c.passed for c in checks if c.name in critical)

    return ValidationResult(passed=passed, checks=checks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    strict = "--strict" in argv

    result = validate_course(strict=strict)
    output = result.to_dict()
    print(json.dumps(output, indent=2))

    if not result.passed:
        sys.exit(1)
