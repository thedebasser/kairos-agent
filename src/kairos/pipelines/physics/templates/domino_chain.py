#!/usr/bin/env python3
"""Domino chain simulation template.

This is a FIXED template executed inside the Docker sandbox.
The simulation agent injects a JSON config dict at the CONFIG placeholder.
All physics, rendering, and FFmpeg logic is handled here — the LLM only
controls the config parameters.

Research-validated physics (van Leeuwen 2004, Cantor & Wojtacki 2022):
  - Spacing: 0.4× domino height
  - Elasticity: 0.0 (fully inelastic)
  - Floor friction: 0.8–1.0
  - Mass: 10 (relative to gravity scale)
  - Substeps: 3 per frame (thin body tunneling prevention)
  - Trigger impulse applied at domino top edge for torque
"""

import json
import math
import os
import random
import subprocess
import sys
import time

import pygame
import pymunk
import pymunk.pygame_util

# === CONFIG INJECTION POINT — replaced by simulation_agent ===
CONFIG = __CONFIG_PLACEHOLDER__
# === END CONFIG ===

# ---------------------------------------------------------------------------
# Unpack config
# ---------------------------------------------------------------------------
WIDTH = CONFIG.get("width", 1080)
HEIGHT = CONFIG.get("height", 1920)
FPS = CONFIG.get("fps", 30)
DURATION_SEC = CONFIG.get("duration_sec", 65)
SEED = CONFIG.get("seed", 42)
BG_COLOR = tuple(CONFIG.get("background_color", [26, 26, 46]))
PALETTE = [tuple(c) for c in CONFIG.get("palette", [
    [255, 107, 107], [254, 202, 87], [50, 255, 126],
    [72, 219, 251], [165, 94, 234], [255, 159, 243],
])]
GRAVITY_Y = CONFIG.get("gravity_y", 900.0)
FLOOR_Y_OFFSET = CONFIG.get("floor_y_offset", 100)

DOMINO_W = CONFIG.get("domino_width", 10)
DOMINO_H = CONFIG.get("domino_height", 60)
DOMINO_COUNT = CONFIG.get("domino_count", 80)
SPACING_RATIO = CONFIG.get("spacing_ratio", 0.4)

PATH_TYPE = CONFIG.get("path_type", "s_curve")
PATH_AMPLITUDE = CONFIG.get("path_amplitude", 150.0)
PATH_CYCLES = CONFIG.get("path_cycles", 1.0)

TRIGGER_TIME_SEC = CONFIG.get("trigger_time_sec", 3.0)
TRIGGER_IMPULSE = CONFIG.get("trigger_impulse", 200.0)

DOMINO_MASS = CONFIG.get("domino_mass", 10.0)
DOMINO_ELASTICITY = CONFIG.get("domino_elasticity", 0.0)
DOMINO_FRICTION = CONFIG.get("domino_friction", 0.5)
FLOOR_ELASTICITY = CONFIG.get("floor_elasticity", 0.0)
FLOOR_FRICTION = CONFIG.get("floor_friction", 0.9)
SUBSTEPS = CONFIG.get("substeps", 3)

random.seed(SEED)

# ---------------------------------------------------------------------------
# Pymunk space setup
# ---------------------------------------------------------------------------
pymunk.pygame_util.positive_y_is_up = False

FLOOR_Y = HEIGHT - FLOOR_Y_OFFSET


def create_space():
    """Create a fresh Pymunk space with floor."""
    space = pymunk.Space()
    space.gravity = (0, GRAVITY_Y)
    space.iterations = 20

    # Floor segment
    floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    floor_shape = pymunk.Segment(floor_body, (0, FLOOR_Y), (WIDTH, FLOOR_Y), 5)
    floor_shape.elasticity = FLOOR_ELASTICITY
    floor_shape.friction = FLOOR_FRICTION
    space.add(floor_body, floor_shape)

    # Side walls (prevent dominos from falling off screen)
    left_wall = pymunk.Segment(floor_body, (0, 0), (0, HEIGHT), 5)
    left_wall.elasticity = 0.0
    left_wall.friction = 0.5
    right_wall = pymunk.Segment(floor_body, (WIDTH, 0), (WIDTH, HEIGHT), 5)
    right_wall.elasticity = 0.0
    right_wall.friction = 0.5
    space.add(left_wall, right_wall)

    return space


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------

def generate_path_positions(count, path_type, amplitude, cycles):
    """Generate (x, y) positions for domino bases along a path.

    All paths run top-to-bottom (Y increases) with the dominos standing
    upright on the floor plane.  For vertical-stack visibility on a 1080×1920
    canvas, we use a top-down traversal with gentle horizontal deflection.
    """
    spacing = DOMINO_H * SPACING_RATIO
    positions = []

    # Margins
    margin_x = 100
    usable_w = WIDTH - 2 * margin_x
    center_x = WIDTH / 2

    # Vertical extent: dominos arranged from top to bottom of screen
    # Leave space for trigger build-up at top and floor at bottom
    y_start = 300  # start well below top
    y_end = FLOOR_Y - DOMINO_H / 2 - 10  # stop just above floor

    if path_type == "straight":
        for i in range(count):
            t = i / max(count - 1, 1)
            x = center_x
            y = y_start + t * (y_end - y_start)
            positions.append((x, y))

    elif path_type == "s_curve":
        for i in range(count):
            t = i / max(count - 1, 1)
            y = y_start + t * (y_end - y_start)
            x = center_x + amplitude * math.sin(2 * math.pi * cycles * t)
            # Clamp to screen bounds
            x = max(margin_x + DOMINO_H / 2, min(WIDTH - margin_x - DOMINO_H / 2, x))
            positions.append((x, y))

    elif path_type == "arc":
        for i in range(count):
            t = i / max(count - 1, 1)
            y = y_start + t * (y_end - y_start)
            x = center_x + amplitude * math.sin(math.pi * t)
            x = max(margin_x + DOMINO_H / 2, min(WIDTH - margin_x - DOMINO_H / 2, x))
            positions.append((x, y))

    else:
        # Fallback to straight
        for i in range(count):
            t = i / max(count - 1, 1)
            x = center_x
            y = y_start + t * (y_end - y_start)
            positions.append((x, y))

    return positions


def compute_domino_angle(positions, index):
    """Compute the angle a domino should face to be perpendicular to path tangent."""
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

    # Path tangent angle
    tangent = math.atan2(dy, dx)
    # Domino stands perpendicular to path direction
    # Since dominos fall "forward" along the path, we orient them perpendicular
    return tangent - math.pi / 2


# ---------------------------------------------------------------------------
# Domino placement
# ---------------------------------------------------------------------------

def place_dominos(space, positions):
    """Place dominos at the given positions, standing upright."""
    dominos = []
    for i, (px, py) in enumerate(positions):
        angle = compute_domino_angle(positions, i)

        moment = pymunk.moment_for_box(DOMINO_MASS, (DOMINO_W, DOMINO_H))
        body = pymunk.Body(DOMINO_MASS, moment)
        body.position = (px, py)
        body.angle = angle

        shape = pymunk.Poly.create_box(body, (DOMINO_W, DOMINO_H))
        shape.elasticity = DOMINO_ELASTICITY
        shape.friction = DOMINO_FRICTION

        space.add(body, shape)
        dominos.append((body, shape, i))

    return dominos


# ---------------------------------------------------------------------------
# Headless physics pre-validation
# ---------------------------------------------------------------------------

def prevalidate_physics(positions):
    """Run a fast headless simulation to verify the chain actually completes.

    Returns (passed, fallen_count, total_count, message).
    """
    space = create_space()
    dominos = place_dominos(space, positions)

    # Simulate trigger at t=0 for pre-validation
    first_body = dominos[0][0]
    # Apply impulse at the top edge of the first domino
    local_top = (0, -DOMINO_H / 2)
    world_top = first_body.local_to_world(local_top)
    first_body.apply_impulse_at_world_point((TRIGGER_IMPULSE, 0), world_top)

    # Run physics for the expected duration (skip rendering)
    dt = 1.0 / FPS
    total_frames = int(DURATION_SEC * FPS)
    # Only simulate up to 80% of frames for quick check
    check_frames = int(total_frames * 0.8)

    for _ in range(check_frames):
        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)

    # Count how many dominos have fallen (angle changed significantly)
    initial_angles = [compute_domino_angle(positions, i) for i in range(len(positions))]
    fallen = 0
    for body, shape, idx in dominos:
        angle_diff = abs(body.angle - initial_angles[idx])
        # A domino is "fallen" if it has rotated more than 30 degrees
        if angle_diff > math.radians(30):
            fallen += 1

    total = len(dominos)
    ratio = fallen / total if total > 0 else 0
    passed = ratio >= 0.75  # At least 75% must fall

    msg = f"Pre-validation: {fallen}/{total} dominos fell ({ratio:.0%})"
    if not passed:
        msg += " — CHAIN STALLED, physics config may need adjustment"

    return passed, fallen, total, msg


# ---------------------------------------------------------------------------
# Drawing utilities (gfxdraw for anti-aliased rendering)
# ---------------------------------------------------------------------------

def draw_domino(surface, body, shape, color):
    """Draw a single domino with anti-aliased edges."""
    vertices = [body.local_to_world(v) for v in shape.get_vertices()]
    points = [(int(v.x), int(v.y)) for v in vertices]

    if len(points) >= 3:
        # Filled polygon
        pygame.draw.polygon(surface, color, points)
        # Anti-aliased outline (subtle highlight)
        highlight = tuple(min(c + 40, 255) for c in color)
        pygame.draw.aalines(surface, highlight, True, points, 1)


def draw_floor(surface):
    """Draw the floor with a subtle gradient line."""
    pygame.draw.line(surface, (60, 60, 80), (0, FLOOR_Y), (WIDTH, FLOOR_Y), 3)
    # Subtle glow below floor line
    for i in range(5):
        alpha_color = (40 + i * 4, 40 + i * 4, 60 + i * 4)
        pygame.draw.line(
            surface, alpha_color,
            (0, FLOOR_Y + 3 + i * 2),
            (WIDTH, FLOOR_Y + 3 + i * 2),
            1,
        )


def draw_trigger_indicator(surface, pos, progress):
    """Draw a countdown / trigger indicator before the push."""
    if progress >= 1.0:
        return
    # Pulsing circle at the first domino
    radius = int(30 + 10 * math.sin(progress * math.pi * 4))
    alpha = int(200 * (1 - progress))
    color = (255, 255, 255)
    x, y = int(pos[0]), int(pos[1])
    pygame.draw.circle(surface, color, (x, y), radius, 2)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Generate path
    positions = generate_path_positions(DOMINO_COUNT, PATH_TYPE, PATH_AMPLITUDE, PATH_CYCLES)

    # --- Headless pre-validation ---
    print("Running physics pre-validation...")
    pv_passed, pv_fallen, pv_total, pv_msg = prevalidate_physics(positions)
    print(pv_msg)
    if not pv_passed:
        print(f"COMPLETION_RATIO={pv_fallen / pv_total:.3f}")
        print("ERROR: Domino chain fails pre-validation. Exiting.")
        sys.exit(1)

    # --- Full simulation ---
    space = create_space()
    dominos = place_dominos(space, positions)

    # Assign colours from palette
    domino_colors = []
    for i in range(len(dominos)):
        color_idx = i % len(PALETTE)
        domino_colors.append(PALETTE[color_idx])

    total_frames = int(DURATION_SEC * FPS)
    trigger_frame = int(TRIGGER_TIME_SEC * FPS)
    triggered = False
    peak_body_count = len(dominos)
    payoff_timestamp = TRIGGER_TIME_SEC

    # Track fallen count for stdout reporting
    initial_angles = [compute_domino_angle(positions, idx) for idx in range(len(positions))]

    # --- FFmpeg pipe ---
    output_path = "/workspace/output/simulation.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    print(f"Rendering {total_frames} frames at {FPS} FPS...")
    render_start = time.time()

    for frame in range(total_frames):
        # --- Trigger ---
        if frame == trigger_frame and not triggered:
            first_body = dominos[0][0]
            local_top = (0, -DOMINO_H / 2)
            world_top = first_body.local_to_world(local_top)
            first_body.apply_impulse_at_world_point((TRIGGER_IMPULSE, 0), world_top)
            triggered = True
            payoff_timestamp = frame / FPS
            print(f"Trigger applied at frame {frame} ({payoff_timestamp:.1f}s)")

        # --- Physics step with substeps ---
        dt = 1.0 / FPS
        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)

        # --- Draw ---
        surface.fill(BG_COLOR)
        draw_floor(surface)

        # Trigger indicator (before trigger)
        if not triggered:
            progress = frame / trigger_frame if trigger_frame > 0 else 1.0
            first_pos = positions[0]
            draw_trigger_indicator(surface, first_pos, progress)

        # Draw dominos
        for (body, shape, idx), color in zip(dominos, domino_colors):
            draw_domino(surface, body, shape, color)

        # --- Write frame ---
        frame_data = pygame.image.tobytes(surface, "RGB")
        try:
            ffmpeg_proc.stdin.write(frame_data)
        except BrokenPipeError:
            print("FFmpeg pipe broken, stopping render.")
            break

        # Progress logging every 5 seconds
        if frame > 0 and frame % (FPS * 5) == 0:
            elapsed = time.time() - render_start
            fallen = sum(
                1 for (body, shape, idx) in dominos
                if abs(body.angle - initial_angles[idx]) > math.radians(30)
            )
            print(f"  Frame {frame}/{total_frames} ({frame/FPS:.0f}s) — "
                  f"elapsed: {elapsed:.1f}s — fallen: {fallen}/{len(dominos)}")

    # --- Finalize ---
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    # Final fallen count
    fallen = sum(
        1 for (body, shape, idx) in dominos
        if abs(body.angle - initial_angles[idx]) > math.radians(30)
    )
    completion_ratio = fallen / len(dominos) if dominos else 0

    print(f"PAYOFF_TIMESTAMP={payoff_timestamp:.2f}")
    print(f"PEAK_BODY_COUNT={peak_body_count}")
    print(f"FALLEN={fallen}/{len(dominos)}")
    print(f"COMPLETION_RATIO={completion_ratio:.3f}")

    render_elapsed = time.time() - render_start
    print(f"Render complete in {render_elapsed:.1f}s — {output_path}")

    if ffmpeg_proc.returncode != 0:
        stderr = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
        print(f"FFmpeg stderr: {stderr[-500:]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
