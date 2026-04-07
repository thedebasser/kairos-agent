"""Camera Router — computes a tracking camera trajectory for the domino run.

Receives the ConnectorOutput (complete waypoint path) and SceneManifest
(placed objects for occlusion checks).  Produces a CameraOutput with
keyframes for smooth third-person follow, occlusion avoidance, and a
pull-back overview at the end.

The heavy Blender-specific camera keyframing lives in
``engines/blender/scripts/physics_camera.py`` — this module computes
the *logical* camera trajectory that the Blender script consumes.
"""

from __future__ import annotations

import logging
import math

from kairos.pipelines.domino.creative.models import (
    CameraKeyframe,
    CameraOutput,
    ConnectorOutput,
    OcclusionEvent,
    SceneManifest,
    Waypoint,
)

logger = logging.getLogger(__name__)

# ─── Defaults ────────────────────────────────────────────────────────

DEFAULT_FOLLOW_DISTANCE = 3.5
DEFAULT_CAMERA_HEIGHT = 4.5
DEFAULT_LOOK_AHEAD = 1.5
DEFAULT_ELEVATION_DEG = 35.0
DEFAULT_AZIMUTH_DEG = 20.0
DEFAULT_KEY_INTERVAL = 5
LERP_REPOSITION_FRAMES = 45   # ~1.5 s at 30 fps
PULLBACK_FRAMES = 90           # 3 s at 30 fps
HOLD_FRAMES = 30               # 1 s hold on overview


# ─── Math helpers ────────────────────────────────────────────────────


def _vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: tuple[float, float, float], s: float):
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec_len(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _vec_norm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _vec_len(v)
    if length < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _vec_cross(a: tuple[float, float, float], b: tuple[float, float, float]):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _lerp_tuple(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    inv = 1.0 - t
    return (a[0] * inv + b[0] * t, a[1] * inv + b[1] * t, a[2] * inv + b[2] * t)


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


# ─── Occlusion detection ────────────────────────────────────────────


def _sphere_intersects_ray(
    ray_origin: tuple[float, float, float],
    ray_dir: tuple[float, float, float],
    centre: tuple[float, float, float],
    radius: float,
) -> bool:
    """Check if a ray (origin + direction) passes within *radius* of *centre*.

    Simplified sphere-ray intersection used as a proxy for object occlusion.
    """
    oc = _vec_sub(ray_origin, centre)
    a = ray_dir[0] ** 2 + ray_dir[1] ** 2 + ray_dir[2] ** 2
    b = 2.0 * (oc[0] * ray_dir[0] + oc[1] * ray_dir[1] + oc[2] * ray_dir[2])
    c = oc[0] ** 2 + oc[1] ** 2 + oc[2] ** 2 - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    # Intersection parameter must be between 0 and 1 (along the ray segment)
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a) if a != 0 else 0
    t2 = (-b + sqrt_disc) / (2 * a) if a != 0 else 0
    return (0 < t1 < 1) or (0 < t2 < 1)


def detect_occlusions(
    cam_pos: tuple[float, float, float],
    target: tuple[float, float, float],
    objects: list[tuple[str, tuple[float, float, float], float]],
) -> list[str]:
    """Return names of objects that occlude the camera→target ray.

    Each object is represented as (name, centre, radius).
    """
    ray_dir = _vec_sub(target, cam_pos)
    occluders: list[str] = []
    for name, centre, radius in objects:
        if _sphere_intersects_ray(cam_pos, ray_dir, centre, radius):
            occluders.append(name)
    return occluders


# ─── Build object collision spheres from manifest ────────────────────


def _manifest_to_spheres(
    manifest: SceneManifest,
) -> list[tuple[str, tuple[float, float, float], float]]:
    """Convert manifest objects to (name, centre, approx_radius) tuples."""
    spheres: list[tuple[str, tuple[float, float, float], float]] = []
    for obj in manifest.objects:
        # Rough bounding radius from scale (default 0.5 m base * scale)
        radius = 0.5 * obj.scale
        spheres.append((obj.asset_id, obj.position, radius))
    return spheres


# ─── Camera Router ───────────────────────────────────────────────────


class CameraRouter:
    """Computes a tracking camera trajectory from the domino path.

    1. Walk along the complete_path_waypoints
    2. For each segment, place camera behind + to the side, elevated
    3. Check for occlusions against scene objects
    4. Reposition smoothly if occluded (lerp over LERP_REPOSITION_FRAMES)
    5. Append pull-back overview at the end
    """

    def __init__(
        self,
        *,
        follow_distance: float = DEFAULT_FOLLOW_DISTANCE,
        camera_height: float = DEFAULT_CAMERA_HEIGHT,
        look_ahead: float = DEFAULT_LOOK_AHEAD,
        key_interval: int = DEFAULT_KEY_INTERVAL,
    ) -> None:
        self._follow_distance = follow_distance
        self._camera_height = camera_height
        self._look_ahead = look_ahead
        self._key_interval = key_interval

    def compute_trajectory(
        self,
        connector_output: ConnectorOutput,
        manifest: SceneManifest,
        *,
        fps: int = 30,
        duration_sec: float = 60.0,
    ) -> CameraOutput:
        """Compute the full camera trajectory.

        Args:
            connector_output: Fully-connected domino path.
            manifest: Scene with placed objects (for occlusion).
            fps: Render frame rate.
            duration_sec: Target video duration.

        Returns:
            Frozen CameraOutput with keyframes and occlusion info.
        """
        waypoints = connector_output.complete_path_waypoints
        if len(waypoints) < 2:
            logger.warning("[camera_router] Too few waypoints (%d)", len(waypoints))
            return CameraOutput(total_frames=0)

        total_frames = int(fps * duration_sec)
        scene_spheres = _manifest_to_spheres(manifest)

        # Map frame → path position by interpolating along waypoints
        path_positions = self._interpolate_path(waypoints, total_frames)

        alpha = math.radians(DEFAULT_ELEVATION_DEG)
        beta = math.radians(DEFAULT_AZIMUTH_DEG)

        raw_keyframes: list[CameraKeyframe] = []
        occlusion_events: list[OcclusionEvent] = []
        repositions = 0

        # Track active reposition lerp
        reposition_start: int | None = None
        reposition_from: tuple[float, float, float] | None = None
        reposition_to: tuple[float, float, float] | None = None

        for frame in range(0, total_frames, self._key_interval):
            pos = path_positions[min(frame, len(path_positions) - 1)]

            # Tangent from finite difference
            next_idx = min(frame + self._key_interval, len(path_positions) - 1)
            tangent = _vec_norm(_vec_sub(
                path_positions[next_idx], pos,
            ))
            if _vec_len(tangent) < 1e-6:
                tangent = (0.0, 1.0, 0.0)

            # Camera offset: behind + to the side, elevated
            right = _vec_norm(_vec_cross(tangent, (0, 0, 1)))
            if _vec_len(right) < 1e-6:
                right = (1.0, 0.0, 0.0)

            neg_tan = _vec_scale(tangent, -1)
            horiz = _vec_norm(_vec_add(
                _vec_scale(neg_tan, math.cos(beta)),
                _vec_scale(right, math.sin(beta)),
            ))

            cam_pos = (
                pos[0] + horiz[0] * math.cos(alpha) * self._follow_distance,
                pos[1] + horiz[1] * math.cos(alpha) * self._follow_distance,
                self._camera_height + math.sin(alpha) * self._follow_distance * 0.3,
            )

            look_target = (
                pos[0] + tangent[0] * self._look_ahead,
                pos[1] + tangent[1] * self._look_ahead,
                pos[2] * 0.5,
            )

            # Occlusion check
            occluders = detect_occlusions(cam_pos, look_target, scene_spheres)
            if occluders:
                # Find a clear position on the opposite side
                alt_horiz = _vec_norm(_vec_add(
                    _vec_scale(neg_tan, math.cos(beta)),
                    _vec_scale(right, -math.sin(beta)),  # flip side
                ))
                alt_pos = (
                    pos[0] + alt_horiz[0] * math.cos(alpha) * self._follow_distance,
                    pos[1] + alt_horiz[1] * math.cos(alpha) * self._follow_distance,
                    self._camera_height + math.sin(alpha) * self._follow_distance * 0.3,
                )

                # Record occlusion event
                occlusion_events.append(OcclusionEvent(
                    frame_start=frame,
                    frame_end=min(frame + LERP_REPOSITION_FRAMES, total_frames),
                    occluder=occluders[0],
                ))
                repositions += 1

                # Start smooth reposition
                reposition_start = frame
                reposition_from = cam_pos
                reposition_to = alt_pos
                cam_pos = alt_pos

            # Apply active lerp reposition
            if reposition_start is not None and reposition_from and reposition_to:
                elapsed = frame - reposition_start
                if elapsed < LERP_REPOSITION_FRAMES:
                    t = _ease_in_out(elapsed / LERP_REPOSITION_FRAMES)
                    cam_pos = _lerp_tuple(reposition_from, reposition_to, t)
                else:
                    reposition_start = None

            raw_keyframes.append(CameraKeyframe(
                frame=frame,
                position=cam_pos,
                look_target=look_target,
            ))

        # Append pull-back overview
        pullback_kfs = self._compute_pullback(
            raw_keyframes, waypoints, total_frames,
        )
        raw_keyframes.extend(pullback_kfs)

        logger.info(
            "[camera_router] Computed %d keyframes, %d repositions, %d occlusion events",
            len(raw_keyframes), repositions, len(occlusion_events),
        )

        return CameraOutput(
            keyframes=raw_keyframes,
            occlusion_events=occlusion_events,
            repositions=repositions,
            follow_distance=self._follow_distance,
            camera_height=self._camera_height,
            total_frames=total_frames,
        )

    # ─── Path interpolation ──────────────────────────────────────────

    def _interpolate_path(
        self,
        waypoints: list[Waypoint],
        total_frames: int,
    ) -> list[tuple[float, float, float]]:
        """Distribute *total_frames* positions evenly along the waypoint path."""
        if len(waypoints) < 2:
            wp = waypoints[0] if waypoints else Waypoint(x=0, y=0, z=0)
            return [(wp.x, wp.y, wp.z)] * total_frames

        # Compute cumulative arc-length
        cum_lengths = [0.0]
        for i in range(1, len(waypoints)):
            dx = waypoints[i].x - waypoints[i - 1].x
            dy = waypoints[i].y - waypoints[i - 1].y
            dz = waypoints[i].z - waypoints[i - 1].z
            seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
            cum_lengths.append(cum_lengths[-1] + seg_len)

        total_length = cum_lengths[-1]
        if total_length < 1e-9:
            wp = waypoints[0]
            return [(wp.x, wp.y, wp.z)] * total_frames

        positions: list[tuple[float, float, float]] = []
        seg_idx = 0
        for f in range(total_frames):
            target_dist = (f / max(1, total_frames - 1)) * total_length

            # Advance to the right segment
            while (
                seg_idx < len(cum_lengths) - 2
                and cum_lengths[seg_idx + 1] < target_dist
            ):
                seg_idx += 1

            seg_start = cum_lengths[seg_idx]
            seg_end = cum_lengths[seg_idx + 1]
            seg_range = seg_end - seg_start
            t = (target_dist - seg_start) / seg_range if seg_range > 1e-9 else 0.0
            t = max(0.0, min(1.0, t))

            wp_a = waypoints[seg_idx]
            wp_b = waypoints[min(seg_idx + 1, len(waypoints) - 1)]
            positions.append((
                wp_a.x + (wp_b.x - wp_a.x) * t,
                wp_a.y + (wp_b.y - wp_a.y) * t,
                wp_a.z + (wp_b.z - wp_a.z) * t,
            ))

        return positions

    # ─── Pull-back overview ──────────────────────────────────────────

    def _compute_pullback(
        self,
        existing: list[CameraKeyframe],
        waypoints: list[Waypoint],
        total_frames: int,
    ) -> list[CameraKeyframe]:
        """Compute pull-back keyframes to a wide overview at the end."""
        if not existing or not waypoints:
            return []

        # Bounding box of all waypoints
        xs = [w.x for w in waypoints]
        ys = [w.y for w in waypoints]
        zs = [w.z for w in waypoints]
        centre = (
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            (min(zs) + max(zs)) / 2,
        )
        diag = math.sqrt(
            (max(xs) - min(xs)) ** 2
            + (max(ys) - min(ys)) ** 2
            + (max(zs) - min(zs)) ** 2
        )
        overview_dist = max(diag * 0.7, 5.0)
        overview_dir = _vec_norm((0.3, -0.5, 0.7))
        overview_pos = _vec_add(centre, _vec_scale(overview_dir, overview_dist))

        last_kf = existing[-1]
        pullback_start = last_kf.frame + self._key_interval
        pullback_end = min(pullback_start + PULLBACK_FRAMES, total_frames)

        kfs: list[CameraKeyframe] = []
        for frame in range(pullback_start, pullback_end, self._key_interval):
            t = (frame - pullback_start) / max(1, pullback_end - pullback_start)
            t = _ease_in_out(max(0.0, min(1.0, t)))
            pos = _lerp_tuple(last_kf.position, overview_pos, t)
            target = _lerp_tuple(last_kf.look_target, centre, t)
            kfs.append(CameraKeyframe(frame=frame, position=pos, look_target=target))

        # Hold on overview
        for frame in range(pullback_end, min(pullback_end + HOLD_FRAMES, total_frames), self._key_interval):
            kfs.append(CameraKeyframe(frame=frame, position=overview_pos, look_target=centre))

        return kfs
