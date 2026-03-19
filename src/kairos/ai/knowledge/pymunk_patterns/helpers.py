# ============================================================================
# Kairos — Verified Pymunk/Pygame Utility Functions
# ============================================================================
# These helper functions are TESTED and WORKING with Pymunk 6.8 + Pygame 2.6.
# They are injected into simulation prompts as ready-to-use code snippets.
#
# COORDINATE CONVENTION: Y-down (pygame-native)
#   pymunk.pygame_util.positive_y_is_up = False
#   space.gravity = (0, 900)
#   Positions map directly to screen pixels — no conversion needed.
# ============================================================================

import pygame
import pymunk
import pymunk.pygame_util
import random
import math

# ---- CRITICAL: Set Y-down coordinate system BEFORE creating any bodies ----
pymunk.pygame_util.positive_y_is_up = False

# ============================================================================
# COLOR PALETTES (dark background + bright objects = best contrast on feeds)
# ============================================================================

PASTEL_RAINBOW = [
    (255, 179, 186),  # Pink
    (255, 223, 186),  # Peach
    (255, 255, 186),  # Yellow
    (186, 255, 201),  # Mint
    (186, 225, 255),  # Sky blue
    (209, 186, 255),  # Lavender
]

CANDY_BRIGHT = [
    (255, 107, 107),  # Coral red
    (255, 159, 67),   # Orange
    (254, 202, 87),   # Yellow
    (29, 209, 161),   # Teal
    (72, 219, 251),   # Sky blue
    (162, 155, 254),  # Purple
]

NEON_POP = [
    (255, 0, 110),    # Hot pink
    (255, 183, 0),    # Amber
    (0, 255, 163),    # Mint green
    (0, 174, 255),    # Electric blue
    (184, 0, 255),    # Violet
    (255, 234, 0),    # Yellow
]

BG_DARK = (26, 26, 46)       # #1a1a2e — deep navy
BG_DARKER = (15, 15, 30)     # Very dark blue
BG_CHARCOAL = (30, 30, 40)   # Charcoal


def hex_to_rgb(h):
    """Convert hex colour string to RGB tuple."""
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ============================================================================
# BOUNDARY CREATION — walls that contain the simulation
# ============================================================================

def create_boundaries(space, width, height, margin=10, elasticity=0.4, friction=0.5):
    """Create floor, ceiling, and side walls as static segments.

    Args:
        space: pymunk.Space
        width, height: screen dimensions (e.g. 1080, 1920)
        margin: wall thickness offset from edges
        elasticity: bounce factor for walls
        friction: surface grip for walls
    """
    static = space.static_body
    walls = [
        # Floor
        pymunk.Segment(static, (0, height - margin), (width, height - margin), margin),
        # Ceiling
        pymunk.Segment(static, (0, margin), (width, margin), margin),
        # Left wall
        pymunk.Segment(static, (margin, 0), (margin, height), margin),
        # Right wall
        pymunk.Segment(static, (width - margin, 0), (width - margin, height), margin),
    ]
    for wall in walls:
        wall.elasticity = elasticity
        wall.friction = friction
        space.add(wall)
    return walls


# ============================================================================
# BALL CREATION — with correct moment of inertia
# ============================================================================

def create_ball(space, pos, radius=20, mass=1.0, elasticity=0.7, friction=0.5, color=None):
    """Create a dynamic circle body with correct moment of inertia.

    IMPORTANT: Always use pymunk.moment_for_circle() — never guess the moment.

    Returns:
        (body, shape, color) tuple
    """
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.elasticity = elasticity
    shape.friction = friction
    space.add(body, shape)
    if color is None:
        color = random.choice(PASTEL_RAINBOW)
    return body, shape, color


# ============================================================================
# BOX CREATION — for blocks, dominoes, structure elements
# ============================================================================

def create_box(space, pos, size=(60, 40), mass=2.0, elasticity=0.2, friction=0.7, color=None):
    """Create a dynamic box body with correct moment of inertia.

    Args:
        pos: (x, y) center position
        size: (width, height) of the box
    """
    moment = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Poly.create_box(body, size)
    shape.elasticity = elasticity
    shape.friction = friction
    space.add(body, shape)
    if color is None:
        color = random.choice(PASTEL_RAINBOW)
    return body, shape, color


# ============================================================================
# DRAWING HELPERS — anti-aliased rendering
# ============================================================================

def draw_aa_circle(surface, color, center, radius):
    """Draw an anti-aliased filled circle with highlight."""
    x, y = int(center[0]), int(center[1])
    r = int(radius)
    if r < 1:
        return
    try:
        import pygame.gfxdraw
        pygame.gfxdraw.aacircle(surface, x, y, r, color)
        pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    except (ImportError, OverflowError):
        pygame.draw.circle(surface, color, (x, y), r)
    # Specular highlight
    highlight = tuple(min(c + 60, 255) for c in color[:3])
    hr = max(1, r // 3)
    hx, hy = x - r // 3, y - r // 3
    try:
        pygame.gfxdraw.filled_circle(surface, hx, hy, hr, (*highlight, 120))
    except (ImportError, OverflowError):
        pass


def draw_rotated_rect(screen, color, center, width, height, angle_rad):
    """Draw a rotated rectangle (for dominoes, blocks, etc.).

    CRITICAL: pygame.transform.rotate() expands the bounding box.
    Must re-center with get_rect(center=...) to avoid visual drift.

    Args:
        angle_rad: angle in RADIANS (from pymunk body.angle)
    """
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0, 0, width, height))
    angle_deg = math.degrees(angle_rad)
    rotated = pygame.transform.rotate(surf, -angle_deg)  # negative for screen coords
    rect = rotated.get_rect(center=(int(center[0]), int(center[1])))
    screen.blit(rotated, rect)


def draw_segment(surface, seg, color=(100, 100, 120), width=4):
    """Draw a pymunk.Segment as a visible line."""
    # In Y-down mode, positions map directly to screen coords
    p1 = int(seg.a.x + seg.body.position.x), int(seg.a.y + seg.body.position.y)
    p2 = int(seg.b.x + seg.body.position.x), int(seg.b.y + seg.body.position.y)
    pygame.draw.line(surface, color, p1, p2, width)


# ============================================================================
# VELOCITY LIMITER — prevents tunneling through walls
# ============================================================================

def make_velocity_limiter(max_velocity=1200):
    """Create a velocity limiter function to prevent tunneling.

    Attach to bodies: body.velocity_func = limiter
    Fast objects can pass through walls with discrete collision detection.
    This caps velocity so objects always stay within detection range.
    """
    def limit_velocity(body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        if body.velocity.length > max_velocity:
            body.velocity = body.velocity.normalized() * max_velocity
    return limit_velocity


# ============================================================================
# PRE-SETTLE — stabilise stacked structures before recording
# ============================================================================

def pre_settle(space, max_steps=6000, dt=1/60, damping_override=0.8):
    """Run physics silently until all dynamic bodies are sleeping.

    Temporarily increases damping for faster settling.
    Use BEFORE the render loop for towers/structures.

    Args:
        max_steps: safety limit to prevent infinite loops
        dt: physics timestep
        damping_override: temporary damping (0.8 = fast settle, 1.0 = no extra damping)
    """
    original_damping = space.damping
    space.damping = damping_override
    space.sleep_time_threshold = 0.5

    for _ in range(max_steps):
        space.step(dt)
        dynamic_bodies = [b for b in space.bodies if b.body_type == pymunk.Body.DYNAMIC]
        if dynamic_bodies and all(b.is_sleeping for b in dynamic_bodies):
            break

    space.damping = original_damping


# ============================================================================
# OFFSCREEN CLEANUP — remove bodies that have left the visible area
# ============================================================================

def remove_offscreen(space, bodies_list, width, height, margin=200):
    """Remove bodies that have moved far offscreen to maintain performance.

    Args:
        bodies_list: list of (body, shape, ...) tuples to check
        width, height: screen dimensions
        margin: pixels beyond screen edge before removal

    Returns:
        Updated list with offscreen bodies removed.
    """
    keep = []
    for entry in bodies_list:
        body = entry[0]
        x, y = body.position
        if -margin < x < width + margin and -margin < y < height + margin:
            keep.append(entry)
        else:
            # Remove from space
            for shape in body.shapes:
                space.remove(shape)
            space.remove(body)
    return keep


# ============================================================================
# GLOW EFFECT — simple bloom for visual polish
# ============================================================================

def apply_glow(screen):
    """Apply a simple bloom/glow effect to the entire screen.

    Downscales, upscales (blurring), then blits additively.
    Call AFTER drawing all objects but BEFORE pygame.image.tostring().
    """
    w, h = screen.get_size()
    small = pygame.transform.smoothscale(screen, (w // 4, h // 4))
    glow = pygame.transform.smoothscale(small, (w, h))
    glow.set_alpha(80)
    screen.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
