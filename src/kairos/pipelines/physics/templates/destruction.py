#!/usr/bin/env python3
"""Block destruction simulation template.

Fixed template executed inside the Docker sandbox.
A block structure is built, allowed to settle, then destroyed by a projectile.
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

STRUCTURE_TYPE = CONFIG.get("structure_type", "tower")
CUSTOM_LAYERS = CONFIG.get("layers", [])
STRUCTURE_CX = CONFIG.get("structure_center_x", 540.0)
STRUCTURE_BASE_OFFSET = CONFIG.get("structure_base_y_offset", 100)
DEFAULT_BW = CONFIG.get("default_block_width", 80.0)
DEFAULT_BH = CONFIG.get("default_block_height", 40.0)
DEFAULT_ROWS = CONFIG.get("default_rows", 10)
DEFAULT_COLS = CONFIG.get("default_cols", 5)

SETTLE_TIME = CONFIG.get("settle_time_sec", 2.0)

PROJ_TYPE = CONFIG.get("projectile_type", "ball")
PROJ_RADIUS = CONFIG.get("projectile_radius", 40.0)
PROJ_MASS = CONFIG.get("projectile_mass", 50.0)
LAUNCH_TIME = CONFIG.get("launch_time_sec", 5.0)
LAUNCH_VX = CONFIG.get("launch_velocity_x", 800.0)
LAUNCH_VY = CONFIG.get("launch_velocity_y", -200.0)
LAUNCH_OX = CONFIG.get("launch_origin_x", 0.0)
LAUNCH_OY = CONFIG.get("launch_origin_y", 1400.0)

BLOCK_MASS = CONFIG.get("block_mass", 2.0)
BLOCK_ELASTICITY = CONFIG.get("block_elasticity", 0.1)
BLOCK_FRICTION = CONFIG.get("block_friction", 0.8)
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
    """Create Pymunk space."""
    space = pymunk.Space()
    space.gravity = (0, GRAVITY_Y)
    space.iterations = 30  # Higher iterations for stacking stability
    return space


def add_floor_and_walls(space):
    """Add floor and side walls."""
    static = pymunk.Body(body_type=pymunk.Body.STATIC)

    floor = pymunk.Segment(static, (0, FLOOR_Y), (WIDTH, FLOOR_Y), 5)
    floor.elasticity = FLOOR_ELASTICITY
    floor.friction = FLOOR_FRICTION

    left = pymunk.Segment(static, (0, 0), (0, HEIGHT), 5)
    left.elasticity = 0.0
    left.friction = 0.5

    right = pymunk.Segment(static, (WIDTH, 0), (WIDTH, HEIGHT), 5)
    right.elasticity = 0.0
    right.friction = 0.5

    space.add(static, floor, left, right)
    return static


# ---------------------------------------------------------------------------
# Structure building
# ---------------------------------------------------------------------------

def build_tower(space):
    """Build a tower from default parameters."""
    blocks = []
    base_y = FLOOR_Y - STRUCTURE_BASE_OFFSET
    total_w = DEFAULT_COLS * DEFAULT_BW
    start_x = STRUCTURE_CX - total_w / 2

    for row in range(DEFAULT_ROWS):
        y = base_y - row * DEFAULT_BH - DEFAULT_BH / 2
        cols = DEFAULT_COLS
        row_offset = (DEFAULT_BW / 2) * (row % 2)  # Alternate brick pattern

        for col in range(cols):
            x = start_x + col * DEFAULT_BW + DEFAULT_BW / 2 + row_offset
            color_idx = row % len(PALETTE)
            block = create_block(space, x, y, DEFAULT_BW, DEFAULT_BH, color_idx)
            blocks.append(block)

    return blocks


def build_pyramid(space):
    """Build a pyramid structure."""
    blocks = []
    base_y = FLOOR_Y - STRUCTURE_BASE_OFFSET

    for row in range(DEFAULT_ROWS):
        cols = DEFAULT_COLS - row
        if cols <= 0:
            break
        y = base_y - row * DEFAULT_BH - DEFAULT_BH / 2
        total_w = cols * DEFAULT_BW
        start_x = STRUCTURE_CX - total_w / 2

        for col in range(cols):
            x = start_x + col * DEFAULT_BW + DEFAULT_BW / 2
            color_idx = row % len(PALETTE)
            block = create_block(space, x, y, DEFAULT_BW, DEFAULT_BH, color_idx)
            blocks.append(block)

    return blocks


def build_wall(space):
    """Build a simple wall."""
    blocks = []
    base_y = FLOOR_Y - STRUCTURE_BASE_OFFSET

    for row in range(DEFAULT_ROWS):
        y = base_y - row * DEFAULT_BH - DEFAULT_BH / 2

        for col in range(DEFAULT_COLS):
            x = STRUCTURE_CX + (col - DEFAULT_COLS / 2) * DEFAULT_BW + DEFAULT_BW / 2
            color_idx = (row + col) % len(PALETTE)
            block = create_block(space, x, y, DEFAULT_BW, DEFAULT_BH, color_idx)
            blocks.append(block)

    return blocks


def build_from_layers(space):
    """Build structure from LLM-defined layers."""
    blocks = []
    base_y = FLOOR_Y - STRUCTURE_BASE_OFFSET
    y_cursor = base_y

    for layer_def in CUSTOM_LAYERS:
        count = layer_def.get("block_count", 5)
        bw = layer_def.get("block_width", DEFAULT_BW)
        bh = layer_def.get("block_height", DEFAULT_BH)
        offset_x = layer_def.get("offset_x", 0)
        color_idx = layer_def.get("color_index", 0)

        y = y_cursor - bh / 2
        total_w = count * bw
        start_x = STRUCTURE_CX - total_w / 2 + offset_x

        for col in range(count):
            x = start_x + col * bw + bw / 2
            block = create_block(space, x, y, bw, bh, color_idx)
            blocks.append(block)

        y_cursor -= bh

    return blocks


def create_block(space, x, y, w, h, color_idx):
    """Create a single block."""
    moment = pymunk.moment_for_box(BLOCK_MASS, (w, h))
    body = pymunk.Body(BLOCK_MASS, moment)
    body.position = (x, y)

    shape = pymunk.Poly.create_box(body, (w, h))
    shape.elasticity = BLOCK_ELASTICITY
    shape.friction = BLOCK_FRICTION

    space.add(body, shape)
    return body, shape, color_idx, w, h


# ---------------------------------------------------------------------------
# Projectile
# ---------------------------------------------------------------------------

def launch_projectile(space):
    """Create and launch the projectile."""
    if PROJ_TYPE == "ball" or PROJ_TYPE == "wrecking_ball":
        moment = pymunk.moment_for_circle(PROJ_MASS, 0, PROJ_RADIUS)
        body = pymunk.Body(PROJ_MASS, moment)
        body.position = (LAUNCH_OX, LAUNCH_OY)
        body.velocity = (LAUNCH_VX, LAUNCH_VY)

        shape = pymunk.Circle(body, PROJ_RADIUS)
        shape.elasticity = 0.3
        shape.friction = 0.5

        space.add(body, shape)
        return body, shape

    elif PROJ_TYPE == "explosion":
        # Apply outward impulse to all nearby blocks
        # (No separate body needed)
        return None, None

    elif PROJ_TYPE == "beam":
        # Horizontal beam sweeps through structure
        moment = pymunk.moment_for_box(PROJ_MASS * 2, (WIDTH, 20))
        body = pymunk.Body(PROJ_MASS * 2, moment)
        body.position = (LAUNCH_OX, LAUNCH_OY)
        body.velocity = (LAUNCH_VX, 0)

        shape = pymunk.Poly.create_box(body, (200, 20))
        shape.elasticity = 0.2
        shape.friction = 0.3

        space.add(body, shape)
        return body, shape

    return None, None


def apply_explosion(space, blocks):
    """Apply explosion force centered on the structure."""
    center_x = STRUCTURE_CX
    center_y = FLOOR_Y - STRUCTURE_BASE_OFFSET - DEFAULT_ROWS * DEFAULT_BH / 2

    for body, shape, _, _, _ in blocks:
        dx = body.position.x - center_x
        dy = body.position.y - center_y
        dist = max(math.sqrt(dx * dx + dy * dy), 1.0)
        force = 50000 / dist
        nx = dx / dist
        ny = dy / dist
        body.apply_impulse_at_local_point((nx * force, ny * force))


# ---------------------------------------------------------------------------
# Pre-settle
# ---------------------------------------------------------------------------

def pre_settle(space, settle_time):
    """Run physics without rendering to let the structure settle."""
    dt = 1.0 / FPS
    frames = int(settle_time * FPS)
    print(f"Pre-settling structure for {settle_time}s ({frames} frames)...")
    for _ in range(frames):
        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)
    print("  Structure settled.")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_block(surface, body, shape, color, w, h):
    """Draw a block."""
    vertices = [body.local_to_world(v) for v in shape.get_vertices()]
    points = [(int(v.x), int(v.y)) for v in vertices]
    if len(points) >= 3:
        pygame.draw.polygon(surface, color, points)
        highlight = tuple(min(c + 30, 255) for c in color)
        pygame.draw.aalines(surface, highlight, True, points, 1)


def draw_projectile(surface, proj_body, proj_shape, frame, launch_frame):
    """Draw the projectile."""
    if proj_body is None:
        return
    pos = (int(proj_body.position.x), int(proj_body.position.y))
    if hasattr(proj_shape, 'radius'):
        r = int(proj_shape.radius)
        color = (220, 220, 220)
        pygame.draw.circle(surface, color, pos, r)
        # Glow effect
        glow_color = (180, 180, 200)
        pygame.draw.circle(surface, glow_color, pos, r + 4, 2)
    else:
        vertices = [proj_body.local_to_world(v) for v in proj_shape.get_vertices()]
        points = [(int(v.x), int(v.y)) for v in vertices]
        if len(points) >= 3:
            pygame.draw.polygon(surface, (220, 220, 220), points)


def draw_floor(surface):
    """Draw floor."""
    pygame.draw.line(surface, (60, 60, 80), (0, FLOOR_Y), (WIDTH, FLOOR_Y), 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))

    space = create_space()
    add_floor_and_walls(space)

    # Build structure
    if CUSTOM_LAYERS:
        blocks = build_from_layers(space)
    elif STRUCTURE_TYPE == "tower":
        blocks = build_tower(space)
    elif STRUCTURE_TYPE == "pyramid":
        blocks = build_pyramid(space)
    elif STRUCTURE_TYPE == "wall":
        blocks = build_wall(space)
    else:
        blocks = build_tower(space)

    print(f"Built {len(blocks)} blocks ({STRUCTURE_TYPE})")

    # Pre-settle the structure
    pre_settle(space, SETTLE_TIME)

    # Record initial positions for displacement tracking
    initial_positions = [(b.position.x, b.position.y) for b, _, _, _, _ in blocks]

    total_frames = int(DURATION_SEC * FPS)
    launch_frame = int(LAUNCH_TIME * FPS)
    projectile_launched = False
    proj_body = None
    proj_shape = None
    peak_body_count = len(blocks)
    payoff_timestamp = LAUNCH_TIME

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

        # --- Launch projectile ---
        if frame == launch_frame and not projectile_launched:
            if PROJ_TYPE == "explosion":
                apply_explosion(space, blocks)
                print(f"Explosion applied at frame {frame} ({current_sec:.1f}s)")
            else:
                proj_body, proj_shape = launch_projectile(space)
                print(f"Projectile launched at frame {frame} ({current_sec:.1f}s)")
            projectile_launched = True
            payoff_timestamp = current_sec

        # --- Physics ---
        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)

        peak_body_count = max(peak_body_count, len(blocks) + (1 if proj_body else 0))

        # --- Draw ---
        surface.fill(BG_COLOR)
        draw_floor(surface)

        # Draw blocks
        for body, shape, color_idx, w, h in blocks:
            color = PALETTE[color_idx % len(PALETTE)]
            draw_block(surface, body, shape, color, w, h)

        # Draw projectile
        if proj_body is not None:
            draw_projectile(surface, proj_body, proj_shape, frame, launch_frame)

        # Pre-launch indicator
        if not projectile_launched:
            progress = frame / launch_frame if launch_frame > 0 else 1.0
            # Draw aiming line
            if PROJ_TYPE != "explosion":
                ox, oy = int(LAUNCH_OX), int(LAUNCH_OY)
                tx = int(STRUCTURE_CX)
                ty = int(FLOOR_Y - DEFAULT_ROWS * DEFAULT_BH / 2)
                alpha = int(100 * (0.5 + 0.5 * math.sin(progress * math.pi * 6)))
                aim_color = (alpha, alpha, alpha + 50)
                pygame.draw.line(surface, aim_color, (ox, oy), (tx, ty), 2)

        # --- Write frame ---
        frame_data = pygame.image.tobytes(surface, "RGB")
        try:
            ffmpeg_proc.stdin.write(frame_data)
        except BrokenPipeError:
            print("FFmpeg pipe broken, stopping render.")
            break

        if frame > 0 and frame % (FPS * 5) == 0:
            elapsed = time.time() - render_start
            displaced = sum(
                1 for (body, _, _, _, _), (ix, iy) in zip(blocks, initial_positions)
                if math.sqrt((body.position.x - ix) ** 2 + (body.position.y - iy) ** 2) > 20
            )
            print(f"  Frame {frame}/{total_frames} ({current_sec:.0f}s) — "
                  f"elapsed: {elapsed:.1f}s — displaced: {displaced}/{len(blocks)}")

    # --- Finalize ---
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    # Count displaced blocks
    displaced = sum(
        1 for (body, _, _, _, _), (ix, iy) in zip(blocks, initial_positions)
        if math.sqrt((body.position.x - ix) ** 2 + (body.position.y - iy) ** 2) > 20
    )
    ratio = displaced / len(blocks) if blocks else 0

    print(f"PAYOFF_TIMESTAMP={payoff_timestamp:.2f}")
    print(f"PEAK_BODY_COUNT={peak_body_count}")
    print(f"DISPLACED={displaced}/{len(blocks)}")
    print(f"COMPLETION_RATIO={ratio:.3f}")

    render_elapsed = time.time() - render_start
    print(f"Render complete in {render_elapsed:.1f}s — {output_path}")

    if ffmpeg_proc.returncode != 0:
        stderr = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
        print(f"FFmpeg stderr: {stderr[-500:]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
