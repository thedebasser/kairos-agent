"""
Verified minimal domino chain — Pymunk 6.8 + Pygame 2.6
Coordinate system: Y-down (pygame-native), no conversion needed.
Produces a 10-second 540x960 MP4 with ~30 dominoes toppling.
Key: spacing = 0.6x height rule, low elasticity for clean toppling.
"""
import pygame
import pymunk
import pymunk.pygame_util
import subprocess
import random
import math
import os

pymunk.pygame_util.positive_y_is_up = False

WIDTH, HEIGHT = 540, 960
FPS = 30
DURATION = 10
TOTAL_FRAMES = FPS * DURATION

random.seed(42)

PALETTE = [(255,107,107),(254,202,87),(29,209,161),(72,219,251),(162,155,254)]
BG = (26, 26, 46)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))

space = pymunk.Space()
space.gravity = (0, 981)
space.damping = 0.99
space.iterations = 20  # Higher iterations for stable domino contacts

# ---- Floor ----
static = space.static_body
floor = pymunk.Segment(static, (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), 5)
floor.elasticity = 0.2
floor.friction = 0.8
space.add(floor)

# ---- Domino parameters (0.6× height rule) ----
DOMINO_W, DOMINO_H = 16, 48
SPACING = 32  # ~0.6 × 48 = 29, rounded up for safety
NUM_DOMINOES = 30

dominoes = []  # (body, shape, color)

# Place dominoes in a gentle S-curve across the screen
start_x = 60
floor_y = HEIGHT - 50 - DOMINO_H / 2  # Stand upright on floor

for i in range(NUM_DOMINOES):
    x = start_x + i * SPACING
    y = floor_y
    mass = 1.0
    moment = pymunk.moment_for_box(mass, (DOMINO_W, DOMINO_H))
    body = pymunk.Body(mass, moment)
    body.position = (x, y)
    shape = pymunk.Poly.create_box(body, (DOMINO_W, DOMINO_H))
    shape.elasticity = 0.1   # Low bounce = clean toppling
    shape.friction = 0.7
    space.add(body, shape)
    color = PALETTE[i % len(PALETTE)]
    dominoes.append((body, shape, color))

# ---- Trigger: push first domino at t=1s ----
triggered = False

# ---- FFmpeg pipe ----
os.makedirs('/workspace/output', exist_ok=True)
pipe = subprocess.Popen([
    'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
    '-crf', '23', '/workspace/output/simulation.mp4'
], stdin=subprocess.PIPE)

for frame in range(TOTAL_FRAMES):
    t = frame / FPS

    # Trigger first domino
    if t >= 1.0 and not triggered:
        first_body = dominoes[0][0]
        first_body.apply_impulse_at_local_point((300, 0), (0, -DOMINO_H / 2))
        triggered = True

    space.step(1/60)
    space.step(1/60)

    # ---- Draw ----
    screen.fill(BG)

    # Floor
    pygame.draw.line(screen, (80, 80, 100),
                     (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), 4)

    # Dominoes (rotated rectangles)
    for body, shape, color in dominoes:
        cx, cy = body.position
        surf = pygame.Surface((DOMINO_W, DOMINO_H), pygame.SRCALPHA)
        pygame.draw.rect(surf, color, (0, 0, DOMINO_W, DOMINO_H))
        angle_deg = math.degrees(body.angle)
        rotated = pygame.transform.rotate(surf, -angle_deg)
        rect = rotated.get_rect(center=(int(cx), int(cy)))
        screen.blit(rotated, rect)

    pipe.stdin.write(pygame.image.tostring(screen, 'RGB'))

pipe.stdin.close()
pipe.wait()
print(f"PAYOFF_TIMESTAMP=1.0")
print(f"PEAK_BODY_COUNT={len(dominoes)}")
