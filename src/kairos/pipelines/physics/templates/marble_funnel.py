#!/usr/bin/env python3
"""Marble funnel / ramp simulation template.

Fixed template executed inside the Docker sandbox.
Marbles spawn at the top and cascade through ramps/funnels into collection bins.
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

MARBLE_R_MIN = CONFIG.get("marble_radius_min", 10)
MARBLE_R_MAX = CONFIG.get("marble_radius_max", 20)
MARBLE_COUNT = CONFIG.get("marble_count", 60)
SPAWN_RATE = CONFIG.get("spawn_rate", 3.0)
SPAWN_START = CONFIG.get("spawn_start_sec", 2.0)
SPAWN_X_MIN = CONFIG.get("spawn_x_min", 350.0)
SPAWN_X_MAX = CONFIG.get("spawn_x_max", 730.0)
SPAWN_Y = CONFIG.get("spawn_y", 50.0)

LAYOUT_TYPE = CONFIG.get("layout_type", "zig_zag")
RAMP_COUNT = CONFIG.get("ramp_count", 6)
CUSTOM_RAMPS = CONFIG.get("ramps", [])
CUSTOM_FUNNELS = CONFIG.get("funnels", [])

BIN_TYPE = CONFIG.get("bin_type", "divided")
BIN_COUNT = CONFIG.get("bin_count", 3)

MARBLE_ELASTICITY = CONFIG.get("marble_elasticity", 0.5)
MARBLE_FRICTION = CONFIG.get("marble_friction", 0.4)
MARBLE_MASS = CONFIG.get("marble_mass", 3.0)
RAMP_ELASTICITY = CONFIG.get("ramp_elasticity", 0.2)
RAMP_FRICTION = CONFIG.get("ramp_friction", 0.7)
SUBSTEPS = CONFIG.get("substeps", 3)

random.seed(SEED)

# ---------------------------------------------------------------------------
# Pymunk space setup
# ---------------------------------------------------------------------------
pymunk.pygame_util.positive_y_is_up = False

FLOOR_Y = HEIGHT - FLOOR_Y_OFFSET


def create_space():
    """Create Pymunk space with floor."""
    space = pymunk.Space()
    space.gravity = (0, GRAVITY_Y)
    space.iterations = 20
    return space


def add_floor_and_walls(space):
    """Add floor and side walls."""
    static = pymunk.Body(body_type=pymunk.Body.STATIC)

    floor = pymunk.Segment(static, (0, FLOOR_Y), (WIDTH, FLOOR_Y), 5)
    floor.elasticity = 0.1
    floor.friction = 0.9

    left = pymunk.Segment(static, (0, 0), (0, HEIGHT), 5)
    left.elasticity = 0.1
    left.friction = 0.5

    right = pymunk.Segment(static, (WIDTH, 0), (WIDTH, HEIGHT), 5)
    right.elasticity = 0.1
    right.friction = 0.5

    space.add(static, floor, left, right)
    return static


# ---------------------------------------------------------------------------
# Ramp generation
# ---------------------------------------------------------------------------

def generate_zig_zag_ramps(static_body, space):
    """Generate alternating left-right ramps spanning wall to wall."""
    ramps = []
    wall_margin = 20  # Ramps touch close to the walls
    usable_h = FLOOR_Y - 250
    ramp_spacing = usable_h / (RAMP_COUNT + 1)

    for i in range(RAMP_COUNT):
        y = 250 + (i + 1) * ramp_spacing
        if i % 2 == 0:
            x1 = wall_margin
            x2 = WIDTH - wall_margin
        else:
            x1 = WIDTH - wall_margin
            x2 = wall_margin

        # Slight downward slope in the direction of travel
        slope = 40
        y1 = y - slope / 2
        y2 = y + slope / 2

        seg = pymunk.Segment(static_body, (x1, y1), (x2, y2), 5)
        seg.elasticity = RAMP_ELASTICITY
        seg.friction = RAMP_FRICTION
        space.add(seg)
        ramps.append(((x1, y1), (x2, y2)))

        # Lip at the downhill end to redirect marbles downward
        lip_x = x2
        lip_y1 = y2
        lip_y2 = y2 + 50
        lip_seg = pymunk.Segment(static_body, (lip_x, lip_y1), (lip_x, lip_y2), 3)
        lip_seg.elasticity = RAMP_ELASTICITY
        lip_seg.friction = RAMP_FRICTION
        space.add(lip_seg)
        ramps.append(((lip_x, lip_y1), (lip_x, lip_y2)))

    return ramps


def generate_cascade_ramps(static_body, space):
    """Generate cascade-style ramps (angled shelves with gaps)."""
    ramps = []
    margin = 60
    usable_h = FLOOR_Y - 250
    ramp_spacing = usable_h / (RAMP_COUNT + 1)

    for i in range(RAMP_COUNT):
        y = 250 + (i + 1) * ramp_spacing
        shelf_w = WIDTH * 0.35
        gap = 80

        if i % 2 == 0:
            # Left shelf
            x1, x2 = margin, margin + shelf_w
            # Right shelf (with gap)
            x3, x4 = margin + shelf_w + gap, WIDTH - margin
        else:
            # Right shelf
            x1, x2 = WIDTH - margin - shelf_w, WIDTH - margin
            # Left shelf (with gap)
            x3, x4 = margin, WIDTH - margin - shelf_w - gap

        slope = 20
        for (sx1, sx2) in [(x1, x2), (x3, x4)]:
            seg = pymunk.Segment(
                static_body,
                (sx1, y - slope / 2), (sx2, y + slope / 2),
                4,
            )
            seg.elasticity = RAMP_ELASTICITY
            seg.friction = RAMP_FRICTION
            space.add(seg)
            ramps.append(((sx1, y - slope / 2), (sx2, y + slope / 2)))

    return ramps


def generate_custom_ramps(static_body, space):
    """Place ramps from the LLM-defined config, extended to walls for reliable catching."""
    ramps = []
    wall_margin = 20

    for ramp_def in CUSTOM_RAMPS:
        x1 = ramp_def.get("x_start", 100)
        y1 = ramp_def.get("y_start", 400)
        x2 = ramp_def.get("x_end", 900)
        y2 = ramp_def.get("y_end", 500)
        thickness = ramp_def.get("thickness", 4)

        # --- Extend ramp to reach both walls ---
        # Calculate slope so we can extend the line
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > 1:
            slope = dy / dx
            # Extend uphill end (lower y) to the wall
            if dx > 0:
                # Ramp goes left→right; extend left end to wall
                new_x1 = wall_margin
                new_y1 = y1 + slope * (new_x1 - x1)
                # Extend right end to wall
                new_x2 = WIDTH - wall_margin
                new_y2 = y1 + slope * (new_x2 - x1)
            else:
                # Ramp goes right→left; extend right end to wall
                new_x1 = WIDTH - wall_margin
                new_y1 = y1 + slope * (new_x1 - x1)
                # Extend left end to wall
                new_x2 = wall_margin
                new_y2 = y1 + slope * (new_x2 - x1)
            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2

        seg = pymunk.Segment(static_body, (x1, y1), (x2, y2), thickness)
        seg.elasticity = RAMP_ELASTICITY
        seg.friction = RAMP_FRICTION
        space.add(seg)
        ramps.append(((x1, y1), (x2, y2)))

        # --- Lip at downhill end to redirect marbles ---
        if y2 >= y1:
            lip_x, lip_y = x2, y2
        else:
            lip_x, lip_y = x1, y1

        lip_height = 50
        lip_seg = pymunk.Segment(
            static_body,
            (lip_x, lip_y),
            (lip_x, lip_y + lip_height),
            3,
        )
        lip_seg.elasticity = RAMP_ELASTICITY
        lip_seg.friction = RAMP_FRICTION
        space.add(lip_seg)
        ramps.append(((lip_x, lip_y), (lip_x, lip_y + lip_height)))

    return ramps


def add_funnels(static_body, space):
    """Add funnel shapes from config."""
    funnel_shapes = []
    for f_def in CUSTOM_FUNNELS:
        cx = f_def.get("center_x", WIDTH / 2)
        top_y = f_def.get("top_y", 400)
        mouth_w = f_def.get("mouth_width", 300)
        neck_w = f_def.get("neck_width", 40)
        h = f_def.get("height", 200)

        # Left side of funnel
        seg_l = pymunk.Segment(
            static_body,
            (cx - mouth_w / 2, top_y),
            (cx - neck_w / 2, top_y + h),
            4,
        )
        seg_l.elasticity = RAMP_ELASTICITY
        seg_l.friction = RAMP_FRICTION
        space.add(seg_l)

        # Right side of funnel
        seg_r = pymunk.Segment(
            static_body,
            (cx + mouth_w / 2, top_y),
            (cx + neck_w / 2, top_y + h),
            4,
        )
        seg_r.elasticity = RAMP_ELASTICITY
        seg_r.friction = RAMP_FRICTION
        space.add(seg_r)

        funnel_shapes.append(((cx, top_y), mouth_w, neck_w, h))

    return funnel_shapes


def add_collection_bins(static_body, space):
    """Add collection bins at the bottom."""
    bins = []
    bin_y_top = FLOOR_Y - 120
    bin_y_bot = FLOOR_Y
    total_w = WIDTH * 0.8
    start_x = (WIDTH - total_w) / 2
    bin_w = total_w / BIN_COUNT

    for i in range(BIN_COUNT):
        bx = start_x + i * bin_w

        # Left wall
        seg = pymunk.Segment(static_body, (bx, bin_y_top), (bx, bin_y_bot), 3)
        seg.elasticity = 0.1
        seg.friction = 0.8
        space.add(seg)

        bins.append((bx, bin_y_top, bx + bin_w, bin_y_bot))

    # Right wall of last bin
    last_x = start_x + BIN_COUNT * bin_w
    seg = pymunk.Segment(static_body, (last_x, bin_y_top), (last_x, bin_y_bot), 3)
    seg.elasticity = 0.1
    seg.friction = 0.8
    space.add(seg)

    return bins


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_marble(surface, body, shape, color):
    """Draw marble with highlight."""
    pos = (int(body.position.x), int(body.position.y))
    r = int(shape.radius)
    pygame.draw.circle(surface, color, pos, r)
    # Small highlight
    hl = (pos[0] - r // 3, pos[1] - r // 3)
    hl_r = max(r // 4, 2)
    hl_color = tuple(min(c + 70, 255) for c in color)
    pygame.draw.circle(surface, hl_color, hl, hl_r)


def draw_ramps(surface, ramps):
    """Draw ramp segments."""
    ramp_color = (70, 70, 110)
    for (x1, y1), (x2, y2) in ramps:
        pygame.draw.line(surface, ramp_color, (int(x1), int(y1)), (int(x2), int(y2)), 6)
        # Highlight edge
        hl_color = (90, 90, 140)
        pygame.draw.line(surface, hl_color, (int(x1), int(y1) - 2), (int(x2), int(y2) - 2), 2)


def draw_funnels(surface, funnels):
    """Draw funnel shapes."""
    funnel_color = (70, 70, 110)
    for (cx, ty), mw, nw, h in funnels:
        pygame.draw.line(surface, funnel_color,
                         (int(cx - mw / 2), int(ty)), (int(cx - nw / 2), int(ty + h)), 5)
        pygame.draw.line(surface, funnel_color,
                         (int(cx + mw / 2), int(ty)), (int(cx + nw / 2), int(ty + h)), 5)


def draw_bins(surface, bins):
    """Draw collection bins."""
    bin_color = (60, 60, 90)
    for bx1, by1, bx2, by2 in bins:
        pygame.draw.line(surface, bin_color, (int(bx1), int(by1)), (int(bx1), int(by2)), 3)
        pygame.draw.line(surface, bin_color, (int(bx2), int(by1)), (int(bx2), int(by2)), 3)


def draw_floor(surface):
    """Draw floor line."""
    pygame.draw.line(surface, (60, 60, 80), (0, FLOOR_Y), (WIDTH, FLOOR_Y), 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))

    space = create_space()
    static_body = add_floor_and_walls(space)

    # Build ramps
    if CUSTOM_RAMPS:
        ramps = generate_custom_ramps(static_body, space)
    elif LAYOUT_TYPE == "zig_zag":
        ramps = generate_zig_zag_ramps(static_body, space)
    elif LAYOUT_TYPE == "cascade":
        ramps = generate_cascade_ramps(static_body, space)
    else:
        ramps = generate_zig_zag_ramps(static_body, space)

    # Build funnels
    funnel_shapes = add_funnels(static_body, space)

    # Build collection bins
    bins = add_collection_bins(static_body, space)

    marbles = []
    marbles_spawned = 0
    spawn_acc = 0.0
    peak_body_count = 0
    payoff_timestamp = SPAWN_START

    total_frames = int(DURATION_SEC * FPS)
    spawn_start_frame = int(SPAWN_START * FPS)

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

        # --- Spawn marbles ---
        if frame >= spawn_start_frame and marbles_spawned < MARBLE_COUNT:
            spawn_acc += SPAWN_RATE * dt
            while spawn_acc >= 1.0 and marbles_spawned < MARBLE_COUNT:
                spawn_acc -= 1.0
                radius = random.randint(MARBLE_R_MIN, MARBLE_R_MAX)
                x = random.uniform(SPAWN_X_MIN, SPAWN_X_MAX)
                y = SPAWN_Y - random.uniform(0, 30)
                color_idx = marbles_spawned % len(PALETTE)

                moment = pymunk.moment_for_circle(MARBLE_MASS, 0, radius)
                body = pymunk.Body(MARBLE_MASS, moment)
                body.position = (x, y)
                shape = pymunk.Circle(body, radius)
                shape.elasticity = MARBLE_ELASTICITY
                shape.friction = MARBLE_FRICTION
                space.add(body, shape)
                marbles.append((body, shape, color_idx))
                marbles_spawned += 1

        # --- Physics ---
        for _ in range(SUBSTEPS):
            space.step(dt / SUBSTEPS)

        peak_body_count = max(peak_body_count, len(marbles))

        # --- Draw ---
        surface.fill(BG_COLOR)
        draw_floor(surface)
        draw_ramps(surface, ramps)
        draw_funnels(surface, funnel_shapes)
        draw_bins(surface, bins)

        for body, shape, color_idx in marbles:
            color = PALETTE[color_idx % len(PALETTE)]
            draw_marble(surface, body, shape, color)

        # --- Write frame ---
        frame_data = pygame.image.tobytes(surface, "RGB")
        try:
            ffmpeg_proc.stdin.write(frame_data)
        except BrokenPipeError:
            print("FFmpeg pipe broken, stopping render.")
            break

        if frame > 0 and frame % (FPS * 5) == 0:
            elapsed = time.time() - render_start
            print(f"  Frame {frame}/{total_frames} ({current_sec:.0f}s) — "
                  f"elapsed: {elapsed:.1f}s — marbles: {len(marbles)}")

    # --- Finalize ---
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    # Marbles that reached the bottom half of the screen
    reached_bottom = sum(
        1 for body, _, _ in marbles
        if body.position.y > HEIGHT * 0.7
    )
    ratio = reached_bottom / MARBLE_COUNT if MARBLE_COUNT > 0 else 1.0

    print(f"PAYOFF_TIMESTAMP={payoff_timestamp:.2f}")
    print(f"PEAK_BODY_COUNT={peak_body_count}")
    print(f"MARBLES_REACHED_BOTTOM={reached_bottom}/{MARBLE_COUNT}")
    print(f"COMPLETION_RATIO={ratio:.3f}")

    render_elapsed = time.time() - render_start
    print(f"Render complete in {render_elapsed:.1f}s — {output_path}")

    if ffmpeg_proc.returncode != 0:
        stderr = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else ""
        print(f"FFmpeg stderr: {stderr[-500:]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
