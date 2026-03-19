"""
Verified minimal destruction simulation — Pymunk 6.8 + Pygame 2.6
Coordinate system: Y-down (pygame-native), no conversion needed.
Produces a 10-second 540x960 MP4 with a block tower + wrecking ball.
Key: pre-settle the tower, then launch a heavy ball.
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

PALETTE = [(255,179,186),(255,223,186),(255,255,186),(186,255,201),(186,225,255)]
BG = (26, 26, 46)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))

space = pymunk.Space()
space.gravity = (0, 900)
space.damping = 0.99
space.iterations = 25  # High iterations for stable stacking

# ---- Floor ----
static = space.static_body
floor = pymunk.Segment(static, (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), 5)
floor.elasticity = 0.3
floor.friction = 0.8
space.add(floor)

# ---- Build tower: 8 layers × 3 blocks ----
BLOCK_W, BLOCK_H = 60, 30
blocks = []  # (body, shape, color)
tower_x = WIDTH // 2
floor_top = HEIGHT - 55  # just above floor segment

for row in range(8):
    for col in range(3):
        x = tower_x + (col - 1) * (BLOCK_W + 2)
        y = floor_top - (row + 0.5) * BLOCK_H
        mass = 2.0
        moment = pymunk.moment_for_box(mass, (BLOCK_W, BLOCK_H))
        body = pymunk.Body(mass, moment)
        body.position = (x, y)
        shape = pymunk.Poly.create_box(body, (BLOCK_W, BLOCK_H))
        shape.elasticity = 0.1
        shape.friction = 0.8
        space.add(body, shape)
        color = PALETTE[row % len(PALETTE)]
        blocks.append((body, shape, color))

# ---- Pre-settle tower (MANDATORY for destruction sims) ----
original_damping = space.damping
space.damping = 0.8
space.sleep_time_threshold = 0.5
for i in range(600):
    space.step(1/60)
    dynamic = [b for b in space.bodies if b.body_type == pymunk.Body.DYNAMIC]
    if dynamic and all(b.is_sleeping for b in dynamic):
        break
space.damping = original_damping

# Wake all bodies for recording
for body, _, _ in blocks:
    body.activate()

# ---- Wrecking ball (created but not yet moving) ----
ball_mass = 80
ball_radius = 40
ball_moment = pymunk.moment_for_circle(ball_mass, 0, ball_radius)
ball_body = pymunk.Body(ball_mass, ball_moment)
ball_body.position = (-100, HEIGHT - 300)  # Offscreen left
ball_shape = pymunk.Circle(ball_body, ball_radius)
ball_shape.elasticity = 0.3
ball_shape.friction = 0.5
space.add(ball_body, ball_shape)
ball_launched = False

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

    # Launch wrecking ball at t=4s
    if t >= 4.0 and not ball_launched:
        ball_body.velocity = (1500, -200)
        ball_launched = True

    space.step(1/60)
    space.step(1/60)

    # ---- Draw ----
    screen.fill(BG)

    # Floor
    pygame.draw.line(screen, (80, 80, 100),
                     (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), 4)

    # Blocks (rotated rectangles)
    for body, shape, color in blocks:
        cx, cy = body.position
        surf = pygame.Surface((BLOCK_W, BLOCK_H), pygame.SRCALPHA)
        pygame.draw.rect(surf, color, (0, 0, BLOCK_W, BLOCK_H))
        rotated = pygame.transform.rotate(surf, -math.degrees(body.angle))
        rect = rotated.get_rect(center=(int(cx), int(cy)))
        screen.blit(rotated, rect)

    # Wrecking ball
    bx, by = int(ball_body.position.x), int(ball_body.position.y)
    pygame.draw.circle(screen, (200, 50, 50), (bx, by), ball_radius)

    pipe.stdin.write(pygame.image.tostring(screen, 'RGB'))

pipe.stdin.close()
pipe.wait()
print(f"PAYOFF_TIMESTAMP=4.0")
print(f"PEAK_BODY_COUNT={len(blocks) + 1}")
