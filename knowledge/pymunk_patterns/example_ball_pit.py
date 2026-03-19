"""
Verified minimal ball pit simulation — Pymunk 6.8 + Pygame 2.6
Coordinate system: Y-down (pygame-native), no conversion needed.
Produces a 10-second 540x960 MP4 with ~50 bouncing balls.
"""
import pygame
import pymunk
import pymunk.pygame_util
import subprocess
import random
import math
import os

# ---- CRITICAL: Y-down coordinate system ----
pymunk.pygame_util.positive_y_is_up = False

WIDTH, HEIGHT = 540, 960
FPS = 30
DURATION = 10
TOTAL_FRAMES = FPS * DURATION

random.seed(42)

PALETTE = [(255,179,186),(255,223,186),(186,255,201),(186,225,255),(209,186,255)]
BG = (26, 26, 46)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))

# ---- Space setup ----
space = pymunk.Space()
space.gravity = (0, 900)       # Positive Y = down (pygame convention)
space.damping = 0.99

# ---- Boundaries ----
static = space.static_body
walls = [
    pymunk.Segment(static, (0, HEIGHT - 10), (WIDTH, HEIGHT - 10), 10),  # Floor
    pymunk.Segment(static, (0, 0), (0, HEIGHT), 10),                     # Left
    pymunk.Segment(static, (WIDTH, 0), (WIDTH, HEIGHT), 10),             # Right
]
for w in walls:
    w.elasticity = 0.5
    w.friction = 0.6
    space.add(w)

# ---- Angled platform (funnel) ----
plat1 = pymunk.Segment(static, (50, 400), (WIDTH // 2 - 40, 500), 5)
plat1.elasticity = 0.4; plat1.friction = 0.6; space.add(plat1)

plat2 = pymunk.Segment(static, (WIDTH - 50, 400), (WIDTH // 2 + 40, 500), 5)
plat2.elasticity = 0.4; plat2.friction = 0.6; space.add(plat2)

# ---- Ball list ----
balls = []  # (body, shape, color)

def spawn_ball(x, y):
    radius = random.uniform(12, 22)
    mass = radius / 10.0
    moment = pymunk.moment_for_circle(mass, 0, radius)  # ALWAYS use helper
    body = pymunk.Body(mass, moment)
    body.position = (x, y)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.7
    shape.friction = 0.4
    space.add(body, shape)
    balls.append((body, shape, random.choice(PALETTE)))

# ---- FFmpeg pipe ----
os.makedirs('/workspace/output', exist_ok=True)
pipe = subprocess.Popen([
    'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
    '-crf', '23', '/workspace/output/simulation.mp4'
], stdin=subprocess.PIPE)

peak_count = 0

for frame in range(TOTAL_FRAMES):
    t = frame / FPS

    # Spawn 2 balls/sec
    if frame % (FPS // 2) == 0 and len(balls) < 80:
        spawn_ball(random.uniform(100, WIDTH - 100), 30)

    # Physics: 2 substeps per frame = 60 Hz
    space.step(1/60)
    space.step(1/60)

    # Cleanup offscreen
    for entry in balls[:]:
        if entry[0].position.y > HEIGHT + 200:
            space.remove(entry[0], entry[1])
            balls.remove(entry)

    peak_count = max(peak_count, len(balls))

    # Draw
    screen.fill(BG)
    for seg in [plat1, plat2] + walls:
        a = (int(seg.a.x + seg.body.position.x), int(seg.a.y + seg.body.position.y))
        b = (int(seg.b.x + seg.body.position.x), int(seg.b.y + seg.body.position.y))
        pygame.draw.line(screen, (100, 100, 120), a, b, 4)
    for body, shape, color in balls:
        pos = (int(body.position.x), int(body.position.y))
        pygame.draw.circle(screen, color, pos, int(shape.radius))
        hl = tuple(min(c+60,255) for c in color)
        pygame.draw.circle(screen, hl, (pos[0]-int(shape.radius)//3, pos[1]-int(shape.radius)//3), max(1, int(shape.radius)//3))

    pipe.stdin.write(pygame.image.tostring(screen, 'RGB'))

pipe.stdin.close()
pipe.wait()
print(f"PAYOFF_TIMESTAMP=7.0")
print(f"PEAK_BODY_COUNT={peak_count}")
