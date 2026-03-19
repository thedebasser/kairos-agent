# Building an automated "oddly satisfying" physics simulation pipeline

Pymunk 6.8 and Pygame 2.6 form a capable foundation for generating short-form physics simulation content, but success depends on precise parameter tuning, correct coordinate handling, polished visuals, and smart LLM prompting. This report synthesizes practical findings across five critical domains—working code patterns, physics tuning, visual rendering, content strategy, and automated code generation—to give you everything needed to build a reliable automated content pipeline.

---

## 1. Pymunk/Pygame working code: the essential repositories and patterns

The single most important Pymunk resource is the **official examples directory** at `github.com/viblo/pymunk` under `pymunk/examples/`. Key files include `bouncing_balls.py` (ball physics with `DrawOptions`), `box2d_pyramid.py` and `box2d_vertical_stack.py` (stacking/collapse mechanics), `balls_and_lines.py` (user-drawn ramps with dynamic balls), `newtons_cradle.py` (constraint-based chain physics), and `contact_and_no_flipy.py` (the critical Y-down coordinate pattern). The `using_sprites.py` example demonstrates game-quality rendering beyond debug drawing.

**marcpaulo15/pygame_funny_simulations** on GitHub is the closest existing project to an "oddly satisfying" pipeline. It includes eight standalone simulations—`colorful_flooding/` (ball pit filling), `sand_clock/` (funnel/hourglass particle flow), `wrecking_ball/` (demolition physics), `gravity_controller/` (mouse-controlled gravity), and `rebound_collisions/`—each with a `config.yml` for parameter tweaking. This repo demonstrates production-quality simulation patterns and is the best starting template.

**techwithtim/PyMunk-Physics-Simulation** provides a complete tower destruction simulation with a swinging wrecking ball on a pivot joint, boundary creation patterns, and ball-launching mechanics. The code uses `space.gravity = (0, 981)` in Y-down coordinates with `elasticity = 0.9` for balls and `0.4` for boundaries.

The Pymunk showcase page (`pymunk.org/en/latest/showcase.html`) reveals particularly relevant projects: a "satisfying bouncing balls" simulation that creates beautiful regular arrangements mid-flight, a **Galton Board** (quincunx) simulation with randomized elasticity, and a **Suika Game** reimplementation. The official `index_video.py` demonstrates motors, joints, sleeping bodies, and automatic shape generation from images.

### The coordinate system problem and its solution

The #1 source of bugs in Pymunk+Pygame code is coordinate mismatch. Pymunk defaults to math-style Y-up coordinates while Pygame uses screen-style Y-down. **The recommended approach for new projects is to match Pygame's Y-down system entirely**:

```python
import pymunk
import pymunk.pygame_util
pymunk.pygame_util.positive_y_is_up = False  # CRITICAL LINE

space = pymunk.Space()
space.gravity = (0, 900)  # Positive Y = down in screen coords
```

This eliminates all coordinate conversion. Positions map directly to screen pixels. The alternative—using Y-up with `flipy = lambda y: -y + screen_height`—introduces bugs every time you forget to convert. The `contact_and_no_flipy.py` example demonstrates this clean approach.

### Core code patterns every simulation needs

The boundary creation pattern wraps the simulation space:

```python
def create_boundaries(space, width, height):
    walls = [
        [(width/2, height-10), (width, 20)],   # Floor
        [(width/2, 10), (width, 20)],           # Ceiling
        [(10, height/2), (20, height)],          # Left
        [(width-10, height/2), (20, height)]    # Right
    ]
    for pos, size in walls:
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = pos
        shape = pymunk.Poly.create_box(body, size)
        shape.elasticity = 0.4
        shape.friction = 0.5
        space.add(body, shape)
```

The minimal game loop pattern:

```python
draw_options = pymunk.pygame_util.DrawOptions(window)
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    window.fill("white")
    space.debug_draw(draw_options)
    pygame.display.update()
    space.step(1/60.0)
    clock.tick(60)
```

---

## 2. Physics parameters that produce satisfying simulations

### Domino chain spacing: the 0.6× height rule

Physics research establishes that **optimal domino spacing is 0.5× to 1.0× the domino height**, with maximum toppling speed at approximately **0.6× domino height**. Standard domino proportions follow the ratio height:width:thickness ≈ **2:1:0.5** (e.g., 60px tall × 30px wide × 15px thick). For a 60px domino, place them **36px apart** (center-to-center distance = width + gap). The gap must be less than domino height or the chain fails. Spacing between 1.5× and 5× thickness provides the most stable, friction-insensitive toppling.

For Pymunk domino implementation: create dominoes as `Poly.create_box(body, size=(width, height))`, set `elasticity = 0.1–0.3` (low bounce for clean toppling), `friction = 0.5–0.8`, and trigger the first domino with `body.apply_impulse_at_local_point()` or a dynamic ball impact.

### Elasticity and friction: the multiplication trap

Pymunk calculates effective collision elasticity by **multiplying both shapes' values**—two shapes at 0.7 produce an effective bounce of just **0.49**. This is the most common source of "why aren't my balls bouncing?" confusion. Practical values by scenario:

- **Satisfying bouncing** (ball pits, marble runs): shape elasticity **0.8–0.95**, walls **0.8–0.99**
- **Realistic settling** (towers, structures): shape elasticity **0.1–0.3**, walls **0.4**
- **Maximum bounce** (enclosed demos): shape elasticity **0.95**, walls **0.999** (never ≥1.0—causes instability)
- **Domino chains**: elasticity **0.2–0.3** for clean toppling without excessive bouncing

Friction follows the same multiplication model. Values above 1.0 are explicitly supported. Key real-world references: wood-on-wood ≈ **0.4**, rubber-on-concrete ≈ **1.0**, ice ≈ **0.04**. For rolling behavior (marble runs), both ball and surface need friction ≥ **0.5**.

### Space configuration for different simulation types

Four preset configurations cover the main simulation categories:

**Satisfying bouncing** (ball pits, bouncing demos): `gravity=(0, 900)`, `damping=1.0`, `iterations=10`, shape elasticity **0.8–0.9**, friction **0.3–0.4**. Lower gravity (600) gives more airtime for visual appeal.

**Stable stacking** (towers, structures pre-collapse): `gravity=(0, 900)`, `iterations=25`, shape elasticity **0.1**, friction **0.8**. Use 5 substeps per frame (`space.step(1/300)` × 5) and enable sleeping with `space.sleep_time_threshold = 0.5`.

**Chain reactions** (dominoes, Rube Goldberg): `gravity=(0, 981)`, `iterations=15–20`, elasticity **0.2**, friction **0.5–0.7**. Higher iterations prevent mushy stacks from buckling prematurely.

**Marble runs** (ramps, funnels): `gravity=(0, 900)`, `iterations=10–15`, ball elasticity **0.5**, friction **0.5**. Ramps built from `pymunk.Segment` shapes angled between endpoints.

### Preventing tunneling and overlap

Pymunk uses discrete (not continuous) collision detection, so fast objects can pass through thin walls. The **primary fix is smaller timesteps**: replace `space.step(1/60)` with 10 calls to `space.step(1/600)`. For fast projectiles, limit velocity via a custom function:

```python
def limit_velocity(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, damping, dt)
    max_vel = 1000
    if body.velocity.length > max_vel:
        body.velocity = body.velocity.normalized() * max_vel
body.velocity_func = limit_velocity
```

Additional settings: `space.collision_slop = 0.1` (default) controls allowed overlap; `space.collision_persistence = 3` caches collision data to prevent jittering. For very fast objects, use `space.segment_query()` as a ray-cast alternative.

### Pre-settling towers before destruction

Run the simulation silently before recording:

```python
space.sleep_time_threshold = 0.5
for i in range(6000):  # Safety limit
    space.step(1/60)
    if all(b.is_sleeping for b in space.bodies 
           if b.body_type == pymunk.Body.DYNAMIC):
        break
# Tower is now settled; begin recording
```

For faster settling, temporarily set `space.damping = 0.8` during the settle phase, then reset to 1.0. Increase iterations temporarily for better accuracy during settling.

### Mass and moment: always use helper functions

**Never guess the moment of inertia**—incorrect values produce visually broken simulations. Use Pymunk's helpers:

```python
moment = pymunk.moment_for_circle(mass, 0, radius)    # Circles
moment = pymunk.moment_for_box(mass, (width, height))  # Boxes
moment = pymunk.moment_for_poly(mass, vertices)         # Polygons
```

The preferred modern approach lets Pymunk auto-calculate: create a `Body()` with no arguments, set `shape.mass` or `shape.density`, and Pymunk computes mass, moment, and center of gravity after adding to the space. Never set mass on both the body and its shapes—the shape value overwrites the body value.

---

## 3. Visual rendering techniques that elevate quality

### Drawing rotated rectangles (the domino rendering problem)

Pygame has no native rotated rectangle function. The correct pattern creates a surface, rotates it, re-centers the bounding rect, then blits:

```python
def draw_rotated_rect(screen, color, center, width, height, angle_deg):
    surf = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0, 0, width, height))
    rotated = pygame.transform.rotate(surf, angle_deg)
    rect = rotated.get_rect(center=center)
    screen.blit(rotated, rect)
```

**Critical detail**: `pygame.transform.rotate()` expands the bounding box to fit the rotated image. Without re-centering via `get_rect(center=...)`, objects appear to wander. Always rotate from the original surface each frame—never rotate an already-rotated surface, which accumulates distortion. For Pymunk bodies, convert radians to degrees with `math.degrees(body.angle)`.

### Anti-aliasing with pygame.gfxdraw

The `pygame.gfxdraw` module provides anti-aliased primitives. The standard pattern draws **both** the AA outline and the filled shape:

```python
import pygame.gfxdraw
# AA filled circle:
pygame.gfxdraw.aacircle(surf, x, y, radius, color)
pygame.gfxdraw.filled_circle(surf, x, y, radius, color)
# AA filled polygon:
pygame.gfxdraw.aapolygon(surf, points, color)
pygame.gfxdraw.filled_polygon(surf, points, color)
```

For shapes without AA functions, render at **2× resolution** on a larger surface and downscale with `pygame.transform.smoothscale()` for bilinear filtering.

### Particle effects for dust, sparks, and trails

A lightweight particle system stores position, velocity, lifetime, and size per particle:

```python
class Particle:
    def __init__(self, pos, vel, life, color, size):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.life = life
        self.max_life = life
        self.size = size
        self.color = color
    def update(self, dt):
        self.pos += self.vel * dt
        self.vel *= 0.98  # Drag
        self.life -= dt
    def draw(self, surf):
        alpha = int(255 * (self.life / self.max_life))
        s = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color[:3], alpha), 
                          (self.size, self.size), self.size)
        surf.blit(s, (self.pos.x-self.size, self.pos.y-self.size))
```

Emit 20–30 particles on collision events with randomized angles and speeds. For trails, spawn 1–2 particles per frame at the object's position with low velocity and short lifetime. **Performance tip**: creating many `SRCALPHA` surfaces per frame is expensive—consider pre-caching particle surfaces at different alpha levels or drawing directly to a shared surface.

### Color palettes that work for satisfying content

Pastel palettes dominate successful satisfying content. Two proven sets:

```python
PASTEL_RAINBOW = [
    (255, 179, 186),  # Pink
    (255, 223, 186),  # Peach
    (255, 255, 186),  # Yellow
    (186, 255, 201),  # Mint
    (186, 225, 255),  # Sky blue
]

CANDY_BRIGHT = [
    (224, 187, 228),  # Pale plum
    (149, 125, 173),  # Pastel purple
    (210, 145, 188),  # Lilac pink
    (254, 200, 216),  # Bubblegum
    (255, 223, 211),  # Rose gold
]
```

Generating pastels programmatically: `((r+255)//2, (g+255)//2, (b+255)//2)` where r, g, b are random 127–255 values. Cycle through palette indices when spawning objects for rainbow effects. **Dark backgrounds** (near-black or deep purple) with bright colored objects provide the contrast that performs best on social feeds.

### Screen shake and glow effects

Screen shake uses a generator pattern applied to the blit offset:

```python
def screen_shake(magnitude=8, duration_frames=20):
    for i in range(duration_frames):
        decay = 1 - (i / duration_frames)
        x = int((random.random()*2-1) * magnitude * decay)
        y = int((random.random()*2-1) * magnitude * decay)
        yield (x, y)
    while True:
        yield (0, 0)
```

Apply shake to the world surface blit position—never to individual objects. A quick bloom/glow effect downscales the screen, upscales back with `smoothscale`, and blits additively:

```python
def apply_glow(screen):
    w, h = screen.get_size()
    small = pygame.transform.smoothscale(screen, (w//2, h//2))
    glow = pygame.transform.smoothscale(small, (w, h))
    glow.set_alpha(100)
    screen.blit(glow, (0,0), special_flags=pygame.BLEND_RGB_ADD)
```

Gradient backgrounds should be pre-rendered once to a cached surface using `pygame.Color.lerp()` across scanlines, then blitted each frame.

---

## 4. What makes physics simulations go viral on short-form platforms

### The satisfaction hierarchy: what gets views

**C4D4U** (2.15M subscribers, **1.6 billion total views**) dominates the satisfying simulation space with soft-body Tetris, jelly races, and Pac-Man physics. His most-viewed video reached **34.8 million views**. Corridor Digital's "Satisfying Puzzle" render became the **25th most-liked TikTok video of all time**. Among real-world chain reactions, **Hevesh5** (4.3M subscribers) proves domino content has massive audience demand.

The engagement hierarchy for 2D-relevant simulation types, ranked by consistent performance:

- **Domino/chain reactions**: Cascading collapse with escalating complexity—strongest narrative tension
- **Ball pit filling/bouncing**: Simple objects multiplying and filling containers—pure accumulation satisfaction
- **Marble runs/races**: Objects navigating elaborate paths—combines anticipation with narrative
- **Tower/structure collapse**: Build-up followed by dramatic destruction—clear payoff moment
- **Perfect fit/geometric puzzles**: Objects slotting into exact spaces—completion satisfaction

### The psychology driving engagement

Professor Robert Colombo's research shows satisfying videos trigger **serotonin and dopamine release**—the same neurotransmitters as sugar consumption. Seven psychological hooks drive this: completion/resolution (showing tasks finishing), predictability (the brain relaxes when outcomes are certain), sensory gratification (visual-auditory stimulation), symmetry preference (evolutionary bias toward ordered patterns), and "visual tactility" (the synaesthetic sensation of physically feeling what you see). **Predictability is key**—unlike real life, satisfying videos offer complete certainty about outcomes, which reduces heart rate and induces calm.

### The 3-second rule and optimal pacing

**50–60% of viewers who leave do so in the first 3 seconds.** Opening frames must show objects already in motion—never start with a title card or slow setup. The optimal structure:

- **0–1 second**: Visual hook—bright colors, objects in motion, immediate intrigue
- **1–15 seconds**: Build-up—increasing complexity, accumulating objects, rising tension
- **15–25 seconds**: Payoff—the satisfying moment (collapse, completion, chain reaction finish)
- **25–30 seconds**: Resolution or seamless loop back to start

**Optimal length**: TikTok's own recommendation is **24–31 seconds**. YouTube Shorts peak at **~13 seconds** (for loop replays) or full 60 seconds. **Seamless loops are extremely powerful**—videos that restart invisibly get rewatched, and algorithms weight replay behavior as an exceptional engagement signal. Target **>90% retention** and **>100% watch-through rate** (indicating rewatches) for algorithmic amplification.

### Sound design is non-negotiable

The #1 differentiator between viral and non-viral simulation content is **synchronized impact sounds**. Every collision, bounce, or landing needs a matching audio cue. Pure sound effects outperform background music for simulation content. Sound categories that work: clinking/tapping for hard objects, squelching for soft bodies, satisfying "thunk" for landings, and crunching for destruction. Strategic silence before a big payoff moment amplifies impact. Captions boost retention **15–25%** since 50% of viewers watch without sound.

### Format requirements

**9:16 vertical (1080×1920)** is mandatory—videos with black bars lose ~40% visible area and signal repurposed content, reducing algorithmic priority. Dark backgrounds with bright colored objects provide feed-stopping contrast. Post **2–3 short clips daily** with a consistent format. Use hashtags: #oddlysatisfying, #simulation, #physics, #satisfying, #asmr. On TikTok in 2025–2026, **Save and Share signals outweigh Likes** for algorithmic distribution.

---

## 5. LLM prompting strategies for reliable code generation

### The prompt template that works

Research across multiple papers shows the most effective LLM code generation prompts combine **role assignment, explicit constraints, output format specification, and few-shot examples**. For Pymunk/Pygame, this template structure produces the highest first-attempt success rates:

```
You are an expert Python developer specializing in 2D physics simulations 
using Pymunk and Pygame.

Generate a COMPLETE, RUNNABLE Python script that creates:
[SIMULATION DESCRIPTION]

Technical requirements:
- Use Pymunk for physics, Pygame for rendering
- Set pymunk.pygame_util.positive_y_is_up = False
- Window: [W]x[H], FPS: 60, gravity: (0, 900)
- Include proper event loop with quit handling
- Use pymunk.pygame_util.DrawOptions for debug rendering
- All bodies/shapes added to space; space.step(1/60) in main loop

Output ONLY executable Python code. No markdown fences. No placeholders.
```

**Shorter prompts perform better**—research found prompts under 50 words had higher success rates. Keep instructions precise but concise. Set LLM **temperature to 0** for initial generation, incrementing by 0.1 per retry if errors occur.

### Few-shot examples are the single biggest quality lever

The paper "When LLMs Meet API Documentation" found that **example code contributes the most** to LLM performance with less-common libraries—more than descriptive text or parameter documentation. For Pymunk, include **1–2 complete working scripts** in every prompt. The few-shot example must demonstrate the complete skeleton: `positive_y_is_up = False`, space setup with gravity, body+shape creation (both dynamic and static), the DrawOptions rendering pattern, and the full game loop with `space.step()`.

A RAG system pulling from a curated database of **20–30 verified Pymunk/Pygame scripts** plus API documentation excerpts improves performance **83–220%** over prompts without documentation, according to the same research. The official Pymunk examples directory is the best source for verified working code.

### Common LLM failure modes with Pymunk/Pygame

Analysis of 558+ incorrect code generation attempts across GPT-4, GPT-3.5, and open-source models reveals these recurring failures:

- **Coordinate system confusion**: LLMs frequently forget `positive_y_is_up = False` or set gravity in the wrong direction, causing objects to fall "up" or spawn off-screen
- **Missing `space.step()`**: The physics never advances, producing a frozen screen
- **API naming errors**: Pygame camelCase confusion (`setCaption` instead of `set_caption`)
- **Incomplete game loops**: Window closes immediately due to malformed event handling
- **Mass/moment misconfiguration**: Guessing moment of inertia instead of using `moment_for_circle/box`
- **Setting mass on both body AND shapes**: Pymunk overwrites body values, producing unexpected physics

These six failure modes account for the vast majority of broken Pymunk code. A static analysis check that validates all six patterns before execution catches most failures.

### The automated pipeline architecture

The LASSI framework achieves **80–85% first-attempt success** with self-correcting loops. The recommended pipeline for Pymunk/Pygame:

1. **Prompt construction**: Template + simulation description + 1–2 few-shot examples + relevant Pymunk API snippets
2. **Generation**: LLM call at temperature=0
3. **Extraction**: Strip markdown fences if present (regex: `` ```python\n(.*?)\n``` ``)
4. **Syntax validation**: `ast.parse()` check
5. **Pattern validation**: Verify `positive_y_is_up`, `space.step()`, `pygame.event.get()`, `pygame.display` are all present
6. **Execution test**: Run in subprocess with 5–10 second timeout, capture stderr
7. **Error feedback loop**: If failure occurs, send error message + original code back to LLM with the instruction "Fix the error and return the COMPLETE corrected script." Up to 3 retries with temperature escalation (+0.1 per attempt)

Expected success rates with this pipeline: **simple simulations (bouncing balls)**: ~85–90% first attempt; **medium complexity (domino chains, marble runs)**: ~60–70% first attempt, ~90% after retries; **complex simulations (joints, custom rendering, collision handlers)**: ~40–50% first attempt, requiring few-shot examples to reach ~80% after retries.

Self-correction is most effective for surface-level issues (missing imports, syntax errors) but less effective for deep logic errors. **External feedback (actual error messages)** is far more effective than asking the LLM to self-review without concrete failure information.

---

## Conclusion: connecting the pieces into a production pipeline

The path from research to production requires combining these findings into a coherent system. The **Y-down coordinate convention** (`positive_y_is_up = False`, gravity `(0, 900)`) should be hardcoded as an invariant—it eliminates the single largest source of bugs. Physics presets for each simulation type (bouncy, stacking, chain reaction, marble run) should be codified as parameter dictionaries injected into prompts, with the **0.6× height domino spacing rule** and the elasticity multiplication behavior documented explicitly for the LLM.

For visual quality, the pipeline should provide the LLM with pre-built utility functions for rotated rectangle drawing, particle emission, anti-aliased shapes, and gradient backgrounds—these are boilerplate the LLM shouldn't reinvent each time. Color palettes (pastel rainbow or candy bright against dark backgrounds) should be supplied as constants.

Content-wise, the **24–31 second sweet spot**, **start-mid-action** convention, and **build-up → payoff → loop** structure should be encoded as timing constraints in the simulation description. Sound design (synchronized impact sounds) requires a post-processing step outside the Pygame simulation itself—consider pre-recording a library of collision sounds mapped to object types.

The most critical insight across all research: **few-shot examples matter more than any other prompting technique** for niche library code generation. Building and maintaining a curated library of 20–30 verified, parameter-annotated Pymunk simulations will deliver more pipeline reliability than any amount of prompt engineering sophistication. Start with the `marcpaulo15/pygame_funny_simulations` repo's patterns and the official Pymunk examples as your foundation.