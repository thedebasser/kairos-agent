"""
Verified minimal marble funnel — Pymunk 6.8 + Pygame 2.6
Coordinate system: Y-down (pygame-native), no conversion needed.
Produces a 10-second 540x960 MP4 with marbles rolling down ramps.
Key: Segment-based ramps, friction >= 0.5 for rolling.
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

PALETTE = [(255,179,186),(255,223,186),(186,255,201),(186,225,255),(209,186,255)]
BG = (26, 26, 46)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
pygame.init()
screen = pygame.Surface((WIDTH, HEIGHT))

space = pymunk.Space()
space.gravity = (0, 900)
space.damping = 0.99
space.iterations = 15

# ---- Static ramps (zig-zag funnel) ----
static = space.static_body
ramps = []

# Zig-zag ramps descending the screen
ramp_data = [
    ((50, 200), (490, 320)),    # Ramp 1: top-left → bottom-right
    ((490, 400), (50, 520)),    # Ramp 2: top-right → bottom-left
    ((50, 600), (490, 720)),    # Ramp 3: left → right
    ((490, 800), (50, 880)),    # Ramp 4: right → left (into collection)
]

for a, b in ramp_data:
    seg = pymunk.Segment(static, a, b, 5)
    seg.elasticity = 0.3
    seg.friction = 0.7  # >= 0.5 for satisfying rolling
    space.add(seg)
    ramps.append(seg)

# Collection floor
cfloor = pymunk.Segment(static, (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), 5)
cfloor.elasticity = 0.3; cfloor.friction = 0.7; space.add(cfloor)
ramps.append(cfloor)

# Side walls
for wall_pts in [((0, 0), (0, HEIGHT)), ((WIDTH, 0), (WIDTH, HEIGHT))]:
    w = pymunk.Segment(static, wall_pts[0], wall_pts[1], 5)
    w.elasticity = 0.3; w.friction = 0.5; space.add(w)
    ramps.append(w)

# ---- Marbles ----
marbles = []  # (body, shape, color)

def spawn_marble(x, y):
    radius = random.uniform(10, 16)
    mass = radius / 8.0
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = (x, y)
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.5
    shape.friction = 0.6  # >= 0.5 for rolling
    space.add(body, shape)
    marbles.append((body, shape, random.choice(PALETTE)))

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

    # Spawn marbles from top
    if frame % (FPS // 3) == 0 and len(marbles) < 100:
        spawn_marble(random.uniform(100, WIDTH - 100), 30)

    space.step(1/60)
    space.step(1/60)

    # Remove off-bottom
    for entry in marbles[:]:
        if entry[0].position.y > HEIGHT + 100:
            space.remove(entry[0], entry[1])
            marbles.remove(entry)

    peak_count = max(peak_count, len(marbles))

    # ---- Draw ----
    screen.fill(BG)

    # Ramps
    for seg in ramps:
        a = (int(seg.a.x + seg.body.position.x), int(seg.a.y + seg.body.position.y))
        b = (int(seg.b.x + seg.body.position.x), int(seg.b.y + seg.body.position.y))
        pygame.draw.line(screen, (100, 100, 130), a, b, 5)

    # Marbles
    for body, shape, color in marbles:
        pos = (int(body.position.x), int(body.position.y))
        r = int(shape.radius)
        pygame.draw.circle(screen, color, pos, r)
        hl = tuple(min(c+60,255) for c in color)
        pygame.draw.circle(screen, hl, (pos[0]-r//3, pos[1]-r//3), max(1, r//3))

    pipe.stdin.write(pygame.image.tostring(screen, 'RGB'))

pipe.stdin.close()
pipe.wait()
print(f"PAYOFF_TIMESTAMP=7.0")
print(f"PEAK_BODY_COUNT={peak_count}")
