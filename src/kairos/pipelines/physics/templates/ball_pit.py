#!/usr/bin/env python3
"""Ball pit simulation template.

Fixed template executed inside the Docker sandbox.
The simulation agent injects a JSON config dict at the CONFIG placeholder.

Physics: bouncy balls dropped into a container with optional climax event
(gate drop, big ball, shake). Satisfying accumulation + bounce physics.
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

BALL_RADIUS_MIN = CONFIG.get("ball_radius_min", 15)
BALL_RADIUS_MAX = CONFIG.get("ball_radius_max", 35)
BALL_COUNT = CONFIG.get("ball_count", 200)
DROP_RATE_P1 = CONFIG.get("drop_rate_phase1", 3.0)
DROP_RATE_P2 = CONFIG.get("drop_rate_phase2", 10.0)
PHASE2_START = CONFIG.get("phase2_start_sec", 15.0)

CONTAINER_TYPE = CONFIG.get("container_type", "box")
CONTAINER_W_RATIO = CONFIG.get("container_width_ratio", 0.7)
CONTAINER_H_RATIO = CONFIG.get("container_height_ratio", 0.4)

CLIMAX_TYPE = CONFIG.get("climax_type", "gate_drop")
CLIMAX_TIME = CONFIG.get("climax_time_sec", 50.0)

BALL_ELASTICITY = CONFIG.get("ball_elasticity", 0.7)
BALL_FRICTION = CONFIG.get("ball_friction", 0.3)
BALL_MASS_MIN = CONFIG.get("ball_mass_min", 1.0)
BALL_MASS_MAX = CONFIG.get("ball_mass_max", 5.0)
WALL_ELASTICITY = CONFIG.get("wall_elasticity", 0.3)
WALL_FRICTION = CONFIG.get("wall_friction", 0.8)
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
    return space


def add_floor(space):
    """Add floor segment."""
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, (0, FLOOR_Y), (WIDTH, FLOOR_Y), 5)
    shape.elasticity = 0.1
    shape.friction = 0.9
    space.add(body, shape)
    return body


def add_container(space, floor_body):
    """Build the ball pit container walls.

    Returns list of (shape, is_gate) tuples.  Gate shapes can be removed
    for the gate_drop climax event.
    """
    container_w = int(WIDTH * CONTAINER_W_RATIO)
    container_h = int(HEIGHT * CONTAINER_H_RATIO)
    cx = WIDTH / 2
    left_x = cx - container_w / 2
    right_x = cx + container_w / 2
    bottom_y = FLOOR_Y
    top_y = bottom_y - container_h

    walls = []

    if CONTAINER_TYPE == "box":
        # Left wall
        s = pymunk.Segment(floor_body, (left_x, top_y), (left_x, bottom_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, False))

        # Right wall
        s = pymunk.Segment(floor_body, (right_x, top_y), (right_x, bottom_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, False))

        # Bottom (gate for climax)
        s = pymunk.Segment(floor_body, (left_x, bottom_y), (right_x, bottom_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, True))  # This is the gate

    elif CONTAINER_TYPE == "v_shape":
        mid_y = bottom_y
        # V-shaped funnel walls
        s = pymunk.Segment(floor_body, (left_x, top_y), (cx - 30, mid_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, False))

        s = pymunk.Segment(floor_body, (right_x, top_y), (cx + 30, mid_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, False))

        # Bottom gate
        s = pymunk.Segment(floor_body, (cx - 30, mid_y), (cx + 30, mid_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, True))

    elif CONTAINER_TYPE == "funnel":
        # Wide top, narrow bottom
        neck_w = 60
        s = pymunk.Segment(floor_body, (left_x, top_y), (cx - neck_w, bottom_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, False))

        s = pymunk.Segment(floor_body, (right_x, top_y), (cx + neck_w, bottom_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, False))

        # Bottom gate
        s = pymunk.Segment(floor_body, (cx - neck_w, bottom_y), (cx + neck_w, bottom_y), 5)
        s.elasticity = WALL_ELASTICITY
        s.friction = WALL_FRICTION
        space.add(s)
        walls.append((s, True))

    else:  # rounded — approximate with segments
        segments = 12
        for i in range(segments):
            t1 = i / segments
            t2 = (i + 1) / segments
            angle1 = math.pi + t1 * math.pi  # bottom semicircle
            angle2 = math.pi + t2 * math.pi
            r = container_w / 2
            x1 = cx + r * math.cos(angle1)
            y1 = bottom_y - container_h / 2 + (container_h / 2) * math.sin(angle1)
            x2 = cx + r * math.cos(angle2)
            y2 = bottom_y - container_h / 2 + (container_h / 2) * math.sin(angle2)
            s = pymunk.Segment(floor_body, (x1, y1), (x2, y2), 5)
            s.elasticity = WALL_ELASTICITY
            s.friction = WALL_FRICTION
            space.add(s)
            walls.append((s, i == segments // 2))  # middle segment is gate

    return walls, (left_x, top_y, right_x, bottom_y)


# ---------------------------------------------------------------------------
# Ball spawning
# ---------------------------------------------------------------------------

def spawn_ball(space, x, y, radius, color_idx):
    """Spawn a single ball at (x, y)."""
    t = (radius - BALL_RADIUS_MIN) / max(BALL_RADIUS_MAX - BALL_RADIUS_MIN, 1)
    mass = BALL_MASS_MIN + t * (BALL_MASS_MAX - BALL_MASS_MIN)
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = (x, y)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = BALL_ELASTICITY
    shape.friction = BALL_FRICTION
    space.add(body, shape)
    return body, shape, color_idx


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_ball(surface, body, shape, color):
    """Draw an anti-aliased ball."""
    pos = (int(body.position.x), int(body.position.y))
    radius = int(shape.radius)
    # Filled circle
    pygame.draw.circle(surface, color, pos, radius)
    # Highlight for 3D effect
    highlight_pos = (pos[0] - radius // 3, pos[1] - radius // 3)
    highlight_r = max(radius // 3, 2)
    highlight_color = tuple(min(c + 80, 255) for c in color)
    pygame.draw.circle(surface, highlight_color, highlight_pos, highlight_r)


def draw_walls(surface, walls, container_bounds):
    """Draw container walls."""
    wall_color = (80, 80, 120)
    for shape, is_gate in walls:
        a = shape.a
        b = shape.b
        pygame.draw.line(surface, wall_color, (int(a.x), int(a.y)), (int(b.x), int(b.y)), 4)


def draw_floor(surface):
    """Draw the floor line."""
    pygame.draw.line(surface, (60, 60, 80), (0, FLOOR_Y), (WIDTH, FLOOR_Y), 3)


# ---------------------------------------------------------------------------
# Climax events
# ---------------------------------------------------------------------------

def execute_climax(space, climax_type, walls, balls, container_bounds):
    """Execute the climax event."""
    if climax_type == "gate_drop":
        # Remove gate segments
        for shape, is_gate in walls:
            if is_gate:
                space.remove(shape)
        # Remove gate from wall list
        walls[:] = [(s, g) for s, g in walls if not g]
        print("CLIMAX: Gate dropped!")

    elif climax_type == "big_ball":
        # Drop a massive ball from above
        cx = WIDTH / 2
        radius = 80
        mass = 50.0
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = (cx, 100)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.5
        shape.friction = 0.3
        space.add(body, shape)
        balls.append((body, shape, 0))  # Use first palette color
        print("CLIMAX: Big ball dropped!")

    elif climax_type == "shake":
        # Apply random impulses to all balls
        for body, shape, _ in balls:
            ix = random.uniform(-500, 500)
            iy = random.uniform(-800, -200)
            body.apply_impulse_at_local_point((ix, iy))
        print("CLIMAX: Container shaken!")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))

    space = create_space()
    floor_body = add_floor(space)
    walls, container_bounds = add_container(space, floor_body)
    left_x, top_y, right_x, bottom_y = container_bounds

    balls = []
    balls_spawned = 0
    spawn_accumulator = 0.0
    climax_triggered = False

    total_frames = int(DURATION_SEC * FPS)
    climax_frame = int(CLIMAX_TIME * FPS)
    peak_body_count = 0
    payoff_timestamp = CLIMAX_TIME

    # --- FFmpeg pipe ---
    output_path = "/workspace/output/simulation.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path,
    ]

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    print(f"Rendering {total_frames} frames at {FPS} FPS...")
    render_start = time.time()

    for frame in range(total_frames):
        current_sec = frame / FPS
        dt = 1.0 / FPS

        # --- Spawn balls ---
        if balls_spawned < BALL_COUNT:
            rate = DROP_RATE_P2 if current_sec >= PHASE2_START else DROP_RATE_P1
            spawn_accumulator += rate * dt
            while spawn_accumulator >= 1.0 and balls_spawned < BALL_COUNT:
                spawn_accumulator -= 1.0
                radius = random.randint(BALL_RADIUS_MIN, BALL_RADIUS_MAX)
                # Spawn above container with some horizontal randomness
                spawn_x = random.uniform(left_x + radius + 10, right_x - radius - 10)
                spawn_y = top_y - radius - random.uniform(20, 100)
                color_idx = balls_spawned % len(PALETTE)
                ball = spawn_ball(space, spawn_x, spawn_y, radius, color_idx)
                balls.append(ball)
                balls_spawned += 1

        # --- Climax event ---
        if frame == climax_frame and not climax_triggered and CLIMAX_TYPE != "none":
            execute_climax(space, CLIMAX_TYPE, walls, balls, container_bounds)
            climax_triggered = True
            payoff_timestamp = current_sec

        # --- Physics ---
        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)

        # Track peak body count
        peak_body_count = max(peak_body_count, len(balls))

        # --- Draw ---
        surface.fill(BG_COLOR)
        draw_floor(surface)
        draw_walls(surface, walls, container_bounds)

        for body, shape, color_idx in balls:
            color = PALETTE[color_idx % len(PALETTE)]
            draw_ball(surface, body, shape, color)

        # --- Write frame ---
        frame_data = pygame.image.tobytes(surface, "RGB")
        try:
            ffmpeg_proc.stdin.write(frame_data)
        except BrokenPipeError:
            print("FFmpeg pipe broken, stopping render.")
            break

        # Progress logging
        if frame > 0 and frame % (FPS * 5) == 0:
            elapsed = time.time() - render_start
            print(f"  Frame {frame}/{total_frames} ({current_sec:.0f}s) — "
                  f"elapsed: {elapsed:.1f}s — balls: {len(balls)}")

    # --- Finalize ---
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    print(f"PAYOFF_TIMESTAMP={payoff_timestamp:.2f}")
    print(f"PEAK_BODY_COUNT={peak_body_count}")
    print(f"BALLS_SPAWNED={balls_spawned}")
    print(f"COMPLETION_RATIO=1.000")  # Ball pits always "complete"

    render_elapsed = time.time() - render_start
    print(f"Render complete in {render_elapsed:.1f}s — {output_path}")

    if ffmpeg_proc.returncode != 0:
        stderr = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
        print(f"FFmpeg stderr: {stderr[-500:]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
