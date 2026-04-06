# Kairos Agentic Content Pipeline — Design Document

**Version:** 1.1  
**Date:** April 2026  
**Status:** Implementation-ready

---

## 1. Overview

Kairos is a multi-agent system that autonomously produces short-form physics simulation videos (domino runs, marble runs). The system replaces static, hard-coded simulation scripts with a creative pipeline where AI agents design scenes, route courses, calibrate physics, validate results, and produce camera-ready renders.

The pipeline is fully autonomous from creative brief to rendered video. The only human intervention is a final review before upload.

### Design Principles

- **Specialised agents with clear boundaries.** Each agent owns one domain. The set designer never adjusts physics. The connector agent never moves set pieces.
- **Validate early, validate often.** Lightweight per-step validation catches obvious failures. A final reviewer catches emergent issues from the full combination.
- **Learn from every run.** Successful calibrations are stored in ChromaDB. The system gets better over time, requiring fewer iterations for new scenarios.
- **Content-type extensible.** The architecture supports multiple content types (dominos, marble runs) with shared infrastructure and swappable content-specific agents.
- **Local-first.** All LLM inference runs on local models via Ollama. No external API costs for production runs.

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CREATIVE PIPELINE                             │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │
│  │ Set Designer  │───>│ Path Setter  │───>│ Connector Agent  │       │
│  │ (Agent 1)     │    │ (Agent 2)    │    │ (Agent 3)        │       │
│  └──────────────┘    └──────────────┘    └──────────────────┘       │
│         │                   │                     │                  │
│         ▼                   ▼                     ▼                  │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │              Per-Step Validator (VLM)                     │        │
│  │  Checks after each agent, catches obvious failures       │        │
│  └─────────────────────────────────────────────────────────┘        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        PHYSICS PIPELINE                              │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Placement Agent   │───>│ Physics Sim  │───>│ Physics      │       │
│  │ (Agent 4)         │    │ (Blender)    │    │ Validation   │       │
│  └──────────────────┘    └──────────────┘    └──────────────┘       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     CINEMATOGRAPHY PIPELINE                          │
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │ Camera Router     │───>│ Camera Validator  │                       │
│  │ (Agent 5)         │    │ (VLM)            │                       │
│  └──────────────────┘    └──────────────────┘                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FINAL REVIEW                                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │ Final Reviewer Agent (watches full rendered run)          │       │
│  │ Catches emergent issues from the full combination         │       │
│  │ Attributes failures to specific agents with reasoning     │       │
│  └──────────────────────────────────────────────────────────┘       │
│         │                                                            │
│         ├─── PASS ──> Final Render ──> Human Review ──> Upload       │
│         │                                                            │
│         └─── FAIL ──> Route feedback to responsible agent            │
│                       Cascade re-run from that point forward         │
└─────────────────────────────────────────────────────────────────────┘
```

### Content-Type Routing

Each content type (domino, marble_run) provides its own versions of Agents 3-5 and content-specific skill libraries. Agents 1-2 (Set Designer, Path Setter) are shared. The orchestrator receives a content_type parameter and wires up the appropriate agents.

```
orchestrator.run(content_type="domino")   → shared agents + domino agents
orchestrator.run(content_type="marble")   → shared agents + marble agents
```

---

## 3. Agent Specifications

### 3.1 Agent 1: Set Designer / Creative Director

**Domain:** Visual storytelling, scene composition, theme setting.  
**Knows about:** Asset catalogue, themes, visual composition, scale relative to content.  
**Does NOT know about:** Domino physics, path routing, connectors.

**Inputs:**
- Content type (domino or marble_run)
- Asset catalogue (full list of available objects with metadata)
- Previous runs (ChromaDB lookup of past scenes to avoid repetition)
- Reference domino/marble size for scale reasoning
- Scene bounds (fixed bounding box per content type)
- Approximate course length target (e.g. "roughly 10 metres of course")

**Prompt strategy:**

Rather than a generic "think of an idea," the set designer receives:

```
You are a set designer for miniature physics simulation videos.

Here are the objects/models you have available:
{asset_catalogue}

Here's what we've done in previous runs (avoid repeating):
{previous_scene_summaries}

Here are some creative approaches to get you started:
- Gigantified everyday scenes (dinner table, bedroom, kitchen)
- Miniature cityscapes (buildings, bridges, streets)
- Fantasy/abstract (floating platforms, impossible geometry)
- Themed sets (construction site, library, toy store)

Design a visually interesting set. Your dominos are {domino_height}m tall,
so a standard dining table would be about {table_height/domino_height:.0f} 
dominos high.

Rules:
- Place objects with enough space between them for a course to pass through
  (minimum gap: {min_gap}m between objects, unless intentionally creating 
  a bridge/plank scenario)
- Stay within scene bounds: {bounds}
- The course will be approximately {target_length}m long
- Choose a ground texture that fits your theme
- Choose appropriate lighting (indoor or outdoor)
- If outdoor: specify a skybox/HDRI
- If indoor: use neutral white/warm lighting, no skybox
- Place decorative objects as well as functional ones to make the scene 
  feel lived-in
```

**Outputs:**

```yaml
scene_manifest:
  theme: "giant's dinner table"
  environment:
    type: indoor  # or outdoor
    ground:
      texture: wooden_floorboards
    lighting:
      type: warm_ambient  # Mapped to Blender lighting preset
      # No skybox for indoor
  
  narrative: >
    Dominoes start on a napkin beside a plate, weave between cutlery,
    drop off the table edge, cross the floor past a toppled wine glass,
    climb a stack of books, and finish on a nearby chair.
  
  objects:
    # Functional set pieces (course will interact with these)
    - asset: dining_table
      position: [0, 0, 0]
      rotation: [0, 0, 0]
      scale: 3.0
      role: functional
    - asset: book_stack
      position: [3.0, 0.5, 0]
      rotation: [0, 0, 15]  # Slightly angled
      scale: 2.5
      role: functional
    - asset: wooden_chair
      position: [4.5, 1.0, 0]
      rotation: [0, 0, -30]
      scale: 3.0
      role: functional
    
    # Decorative objects (atmosphere, not course-functional)
    - asset: dinner_plate
      position: [0.3, 0.2, 0.75]
      rotation: [0, 0, 0]
      scale: 3.0
      role: decorative
    - asset: wine_glass_toppled
      position: [2.0, -0.3, 0]
      rotation: [90, 0, 45]
      scale: 3.0
      role: decorative
    - asset: napkin
      position: [-0.5, 0.1, 0.75]
      rotation: [0, 0, 12]
      scale: 3.0
      role: decorative
```

**Per-step validation criteria (after Agent 1):**
- No functional objects overlapping each other
- All objects within scene bounds
- Minimum gap between functional objects respected
- At least 2 functional objects at different elevations (otherwise the course is boring)
- Ground texture and lighting type specified
- Indoor scenes have no skybox; outdoor scenes have skybox/HDRI

---

### 3.2 Agent 2: Path Setter

**Domain:** Spatial reasoning, 3D routing, gradient management.  
**Knows about:** Object positions and surface geometry, gradient constraints, scene bounds, camera frustum.  
**Does NOT know about:** Domino physics, connector construction, theme/aesthetics.

**Inputs:**
- Scene manifest from Agent 1 (object positions, surfaces, roles)
- Surface metadata from asset catalogue (usable areas, heights, boundaries)
- Path constraints (min/max gradient, no self-intersection, approximate target length)
- Content type (affects gradient limits — dominoes max ~30°, marbles can handle steeper)

**How it reasons:**

The path setter receives the scene with placed objects and their surface metadata. It knows:
- The dining table has a flat surface at height 2.25m (0.75m × 3.0 scale), dimensions 3.6m × 1.8m
- The book stack has a sloped surface from 0.0m to 0.9m
- The chair seat is at 1.35m

It needs to find a route that visits the functional objects in a plausible order, respecting gradient constraints. It doesn't place connectors — it just flags where they're needed.

**Outputs:**

```yaml
path:
  total_length_estimate: 11.2  # metres
  
  segments:
    - id: seg_01
      type: flat_surface
      surface_ref: dining_table.tabletop
      waypoints:
        - position: [-0.3, 0.1, 2.25]
        - position: [0.8, 0.4, 2.25]
        - position: [1.5, 0.3, 2.25]
      gradient: 0
      notes: "Weaving path across table between decorative objects"
    
    - id: seg_02
      type: height_transition_down
      from_height: 2.25
      to_height: 0.0
      horizontal_distance: 1.2
      required_gradient: -62  # Too steep for direct ramp
      needs_connector: true
      connector_hint: "spiral_or_staircase"
      available_footprint: [1.0, 1.0]  # Space available for connector
      notes: "Table edge to floor — needs spiral or staircase"
    
    - id: seg_03
      type: ground_level
      waypoints:
        - position: [2.2, 0.3, 0.0]
        - position: [2.8, 0.4, 0.0]
      gradient: 0
      notes: "Across floor past toppled wine glass"
    
    - id: seg_04
      type: height_transition_up
      from_height: 0.0
      to_height: 0.9
      horizontal_distance: 0.8
      required_gradient: 48  # Steep but possible with ramp
      needs_connector: true
      connector_hint: "ramp_or_small_staircase"
      available_footprint: [0.8, 0.6]
      notes: "Ground to book stack top"
    
    - id: seg_05
      type: flat_surface
      surface_ref: book_stack.top
      waypoints:
        - position: [3.0, 0.5, 0.9]
        - position: [3.3, 0.6, 0.9]
      gradient: 0
      notes: "Across book stack"
    
    - id: seg_06
      type: height_transition_up
      from_height: 0.9
      to_height: 1.35
      horizontal_distance: 1.5
      required_gradient: 17  # Gentle enough for ramp
      needs_connector: true
      connector_hint: "ramp_or_plank_bridge"
      available_footprint: [1.5, 0.5]
      notes: "Book stack to chair seat"
    
    - id: seg_07
      type: flat_surface
      surface_ref: chair.seat
      waypoints:
        - position: [4.3, 0.9, 1.35]
        - position: [4.5, 1.0, 1.35]
      gradient: 0
      notes: "Across chair seat — finale"
```

**Per-step validation criteria (after Agent 2):**
- All segments connect (no gaps between segment end and next segment start)
- No gradient exceeds content-type maximum
- Path doesn't self-intersect
- Path visits all functional objects marked in the scene manifest
- Total estimated length is within ±30% of target
- Every segment is within scene bounds
- needs_connector segments have valid available_footprint (enough space for a connector)

---

### 3.3 Agent 3: Connector Agent

**Domain:** Structural engineering, physics-aware construction.  
**Knows about:** Skill library (ramps, spirals, stairs, platforms, planks), physics constraints, calibration data from ChromaDB.  
**Does NOT know about:** Theme/aesthetics, scene composition, camera.

**Inputs:**
- Path from Agent 2 (with needs_connector flags)
- Skill library (available connector primitives)
- Calibration data from ChromaDB (correction factors for similar transitions)
- Content type (domino vs marble — affects connector choices)

**How it reasons:**

For each flagged transition, it evaluates the available connector types against the constraints:

```
Transition seg_02: Table (2.25m) → Floor (0.0m)
Height difference: 2.25m down
Available footprint: 1.0m × 1.0m
Available connectors:
  - ramp: needs 2.25/tan(25°) = 4.8m horizontal → REJECTED (footprint too small)
  - staircase: 15 steps × 0.15m = 2.25m, needs 15 × 0.1m = 1.5m horizontal → REJECTED
  - spiral: radius 0.4m, ~3 turns, fits in 0.8m × 0.8m → ACCEPTED
  - spiral: radius 0.3m, ~4 turns, fits in 0.6m × 0.6m → ACCEPTED (tighter)
  
Selecting: spiral, radius 0.35m, 3.5 turns (balance of space and gradient)
ChromaDB lookup: similar spiral descent found, confidence 0.87
  → spacing_correction: 0.92 (8% tighter than formula on spirals)
  → recommended friction: 0.65
```

**Outputs:**

```yaml
connectors:
  - id: conn_01
    for_segment: seg_02
    type: spiral_descent
    params:
      center: [1.8, 0.3, 0.0]
      radius: 0.35
      turns: 3.5
      direction: clockwise
      start_height: 2.25
      end_height: 0.0
      gradient_per_turn: -18.4  # degrees
    calibration:
      source: chromadb
      match_confidence: 0.87
      spacing_correction: 0.92
      friction_override: 0.65
    generated_waypoints:
      # Dense waypoints through the spiral
      - [1.8, 0.65, 2.25]
      - [2.15, 0.3, 2.09]
      - [1.8, -0.05, 1.93]
      # ... (many more, computed from spiral geometry)
      - [1.8, 0.65, 0.0]
  
  - id: conn_02
    for_segment: seg_04
    type: ramp
    params:
      start: [2.8, 0.4, 0.0]
      end: [3.0, 0.5, 0.9]
      width: 0.15
      angle: 48
      # 48° is steep — adding side rails for stability
      has_rails: true
    calibration:
      source: formula_default
      match_confidence: null  # No prior calibration for this angle
      notes: "First steep ramp — will need calibration run"
    generated_waypoints:
      - [2.8, 0.4, 0.0]
      - [2.85, 0.42, 0.15]
      - [2.9, 0.45, 0.45]
      - [2.95, 0.47, 0.67]
      - [3.0, 0.5, 0.9]

# Updated complete path (original segments + connector waypoints merged)
complete_path:
  waypoints: [...]  # Full ordered list
  segment_types: [...]  # Type annotation per waypoint
```

**Procedural generation approach:**

Connectors are generated via Blender Python scripts in the skill library. Each connector type is a parametric function:

```python
# Example: spiral connector primitive (simplified)
def create_spiral(center, radius, turns, start_height, end_height, direction):
    """
    Generates spiral ramp geometry and returns waypoints + mesh.
    """
    points_per_turn = 32
    total_points = int(turns * points_per_turn)
    height_step = (end_height - start_height) / total_points
    
    waypoints = []
    for i in range(total_points):
        angle = (i / points_per_turn) * 2 * pi
        if direction == "clockwise":
            angle = -angle
        x = center[0] + radius * cos(angle)
        y = center[1] + radius * sin(angle)
        z = start_height + (i * height_step)
        waypoints.append([x, y, z])
    
    # Generate ramp mesh along waypoints
    mesh = generate_ramp_mesh(waypoints, width=0.12, thickness=0.02)
    
    return waypoints, mesh
```

**Per-step validation criteria (after Agent 3):**
- All flagged transitions have connectors assigned
- Connector geometry fits within available footprint
- Gradient at every point within content-type limits
- No connector geometry intersects with scene objects
- No connector geometry intersects with other connectors
- Complete path (segments + connectors) is continuous — no gaps

---

### 3.4 Agent 4: Placement Agent (Content-Specific)

**Domain:** Physics-aware object placement along a path.  
**Content-specific:** Separate implementations for domino and marble.

#### Domino Placement Agent

**Knows about:** Domino physics (spacing, mass, friction, rigid body config), scaling laws, calibration data from ChromaDB.

**Inputs:**
- Complete path with segment types and waypoints
- Calibration data from ChromaDB (spacing corrections, physics overrides per segment type)
- Domino dimensions (default: 0.08m × 0.04m × 0.006m — height × width × depth)

**How it works:**

For each path segment, the placement agent:
1. Looks up calibration data for that segment type
2. Computes spacing using the scaling law formula + correction factors
3. Places dominos along the waypoints with appropriate rotation (facing path tangent)
4. Configures rigid body physics per domino
5. Sets up the trigger (first domino push)

**Spacing computation:**

```python
def compute_spacing(segment_type, domino_height, gradient, curvature, calibration):
    # Base formula: spacing ≈ domino_height × spacing_ratio
    base_spacing = domino_height * 0.35  # Default ratio
    
    # Apply gradient correction
    if gradient > 0:  # Uphill
        base_spacing *= (1.0 - 0.01 * abs(gradient))  # Tighter uphill
    elif gradient < 0:  # Downhill
        base_spacing *= (1.0 + 0.005 * abs(gradient))  # Wider downhill
    
    # Apply curvature correction
    if curvature > 0:
        base_spacing *= (1.0 - 0.1 * curvature)  # Tighter on curves
    
    # Apply learned correction factor from ChromaDB
    if calibration and calibration.spacing_correction:
        base_spacing *= calibration.spacing_correction
    
    return base_spacing
```

#### Marble Run Placement Agent

**Knows about:** Track piece connector system, ball physics, momentum requirements.

Marble runs use the connector-based approach (pieces must physically join). The placement agent selects track pieces from the skill library, validates connector compatibility (diameter, direction), and ensures sufficient momentum throughout.

Not detailed in this version — to be specced when marble run content type is implemented.

---

### 3.5 Agent 5: Camera Router

**Domain:** Cinematography, third-person follow camera, occlusion avoidance.

**Inputs:**
- Complete path (the spline the camera follows)
- Physics bake results (wavefront position per frame)
- Scene object positions (for occlusion detection)
- Content type (affects camera distance — marbles are smaller, camera needs to be closer)

**Default camera behaviour: Third-person follow (Mario 64 style)**

```python
# Camera follows the domino wavefront
def compute_camera_position(frame, wavefront_pos, path_tangent, scene_objects):
    # Default: behind and above
    follow_distance = 1.5  # metres behind wavefront
    camera_height = 0.8    # metres above wavefront
    look_ahead = 0.5       # metres ahead of wavefront
    
    # Camera position
    offset = -path_tangent * follow_distance
    offset.z += camera_height
    camera_pos = wavefront_pos + offset
    
    # Look target
    look_target = wavefront_pos + path_tangent * look_ahead
    
    # Occlusion check (every 24 frames = 1 second)
    if frame % 24 == 0:
        if is_occluded(camera_pos, wavefront_pos, scene_objects):
            camera_pos = find_clear_position(
                wavefront_pos, path_tangent, scene_objects,
                preferred_side="right",  # Try right side first
                min_distance=1.0,
                max_distance=2.5
            )
    
    return camera_pos, look_target
```

**Camera speed matching:**

The camera travels along the path spline at the same speed as the domino wavefront. Speed is derived from the physics bake — the wavefront position is known at every frame, so the camera just tracks it with a fixed offset.

**Smooth transitions:**

When the camera needs to reposition (due to occlusion), it lerps over 30-60 frames using bezier interpolation. Blender keyframes handle this naturally — set a keyframe at the current position and the new position with bezier curves.

**Camera validation criteria:**
- Dominos visible in every frame (no fully occluded frames)
- No more than 10% of frames with partial occlusion
- Camera movement is smooth (no velocity spikes > 2× average)
- Key moments (elevation changes, bridge crossings, finale) are well-framed

---

### 3.6 Validation Agent (Per-Step)

**Domain:** Quick quality checks between pipeline steps.  
**Model:** Qwen3-VL-8B via Ollama

The per-step validator runs lightweight checks after each agent completes. It does NOT render the full scene — it works from the structured data (YAML outputs) and optionally a quick viewport screenshot from Blender.

**Validation criteria per step:**

| After Agent | Check | Method |
|---|---|---|
| Set Designer | Objects don't overlap | Bounding box intersection test |
| Set Designer | Objects within scene bounds | Position vs bounds check |
| Set Designer | Minimum gap between functional objects | Distance calculation |
| Set Designer | At least 2 elevation levels | Height analysis |
| Set Designer | Environment specified (ground, lighting) | Schema validation |
| Path Setter | Segments connect without gaps | Endpoint distance check |
| Path Setter | Gradient within limits | Gradient calculation per segment |
| Path Setter | No self-intersection | Line segment intersection test |
| Path Setter | All functional objects visited | Object reference check |
| Path Setter | Length within target range | Path length sum |
| Connector | All flagged transitions have connectors | Coverage check |
| Connector | Connectors fit available footprint | Footprint vs bounds |
| Connector | No connector-object intersections | Bounding box test |
| Connector | Complete path is continuous | Gap detection |
| Placement | Domino count within budget | Count check |
| Physics | Chain completion ratio | Smoke test output |
| Physics | No anomalies (clipping, explosions) | VLM on key frames |
| Camera | Visibility ratio > 90% | Occlusion ray test |
| Camera | Smooth motion | Velocity analysis |

**Failure attribution:**

When validation fails, the validator produces:

```yaml
validation_result:
  step: connector_agent
  status: FAIL
  failures:
    - check: connector_fits_footprint
      details: "Connector conn_01 (spiral, radius 0.35m) exceeds available 
               footprint at seg_02. Spiral footprint: 0.7m × 0.7m, 
               available: 0.6m × 0.6m"
      severity: blocking
      suggested_fix: "Reduce spiral radius to 0.28m (4.5 turns) or 
                      request path setter to allocate more space"
      attributed_to: connector_agent  # Could also be path_setter if footprint was wrong
  
  cascade_required: false  # Only re-run from connector agent
```

---

### 3.7 Final Reviewer Agent

**Domain:** Holistic end-to-end review of the complete rendered run.  
**Model:** Qwen3-VL-8B (or 30B for escalation)

The final reviewer watches the complete rendered video and evaluates it as a whole. It catches issues that per-step validation misses — things that are only visible when everything comes together.

**Inputs:**
- Rendered video (or key frame sequence)
- Scene manifest, path data, connector data (for context)
- Full pipeline history (what each agent did, any retries)

**What it checks:**
- Does the full domino chain complete without breaks?
- Is the camera work smooth and the action always visible?
- Does the scene look visually cohesive (theme consistency)?
- Are there any physics anomalies visible in the render?
- Is the pacing good (not too fast, not too slow)?
- Are the transitions between segments smooth?

**Failure attribution:**

The final reviewer has access to the full agent roster and must attribute failures:

```yaml
final_review:
  status: FAIL
  issues:
    - description: "Chain breaks at transition from table to spiral descent. 
                    Last 3 dominos on table edge don't have enough momentum 
                    to reach first domino on spiral."
      attributed_to: connector_agent
      reason: "Spiral entry point is 4cm too far from table edge"
      suggested_fix: "Move spiral start position 4cm closer to table edge, 
                      or add a small ramp transition piece"
      cascade_from: connector_agent  # Re-run from agent 3 onwards
    
    - description: "Camera is briefly occluded by table leg during spiral 
                    descent at frames 340-380"
      attributed_to: camera_router
      reason: "Occlusion check interval (24 frames) was too slow to catch 
               the table leg passing through frame"
      suggested_fix: "Increase occlusion check frequency to every 12 frames 
                      during spiral segments"
      cascade_from: camera_router  # Re-run from agent 5 onwards
```

**Cascade logic:**

When the final reviewer attributes a failure:
- If attributed to Set Designer → re-run ALL agents (1-5 + camera + render)
- If attributed to Path Setter → re-run from Agent 2 onwards
- If attributed to Connector Agent → re-run from Agent 3 onwards
- If attributed to Placement Agent → re-run from Agent 4 onwards
- If attributed to Camera Router → re-run from Agent 5 onwards only

---

## 4. Iteration History & Agent Memory

### Session State

Every pipeline run maintains a full session history:

```
sessions/{session_id}/
├── manifest.yaml                 # Creative brief + content type
├── agents/
│   ├── set_designer/
│   │   ├── attempt_1/
│   │   │   ├── output.yaml       # Scene manifest
│   │   │   ├── validation.yaml   # Per-step validation result
│   │   │   └── summary.md        # Human-readable summary of what was tried
│   │   └── attempt_2/            # If retry was needed
│   │       └── ...
│   ├── path_setter/
│   │   └── ...
│   ├── connector/
│   │   └── ...
│   ├── placement/
│   │   └── ...
│   └── camera/
│       └── ...
├── physics/
│   ├── bake_result.json          # Physics simulation output
│   └── smoke_test.json           # Quick validation results
├── render/
│   ├── preview/                  # Low-quality preview frames
│   └── final/                    # Production render
├── final_review.yaml             # Final reviewer assessment
└── result.yaml                   # Overall status: success/failed/escalated
```

### Feedback Format

When the validator or final reviewer sends feedback to an agent, it includes:

```yaml
feedback:
  to_agent: connector_agent
  attempt: 2
  
  # What went wrong
  failure: "Spiral connector at seg_02 — entry point too far from table edge"
  
  # What's been tried before (human-readable summary to save tokens)
  history: |
    Attempt 1: Placed spiral at radius 0.35m, center [1.8, 0.3, 0.0].
    Validation failed: spiral footprint exceeded available space.
    
    Attempt 2: Reduced radius to 0.28m, 4.5 turns.
    Validation passed but final review failed: entry point 4cm too far
    from table edge, chain breaks at transition.
  
  # Suggested fix
  suggestion: "Move spiral center 4cm closer to table edge, 
               or add a small ramp transition piece between table and spiral"
  
  # What the agent should focus on
  constraint: "Keep spiral radius ≤ 0.30m to fit footprint. 
               Ensure first spiral waypoint is within 2cm of table edge."
```

### Iteration Limits

- **Per agent:** 10 attempts maximum
- **Pipeline total:** 30 attempts across all agents combined
- **On exhaustion:** Log failure with each agent's summary of what went wrong

```yaml
failure_report:
  status: EXHAUSTED
  message: "Pipeline failed after 30 total attempts"
  agent_summaries:
    set_designer: 
      attempts: 2
      summary: "Placed dinner table scene. Second attempt moved chair 
               closer after path setter couldn't reach it."
    path_setter:
      attempts: 4
      summary: "Struggled with gradient from table to floor. 62° direct 
               descent too steep. Tried routing via book stack but added 
               too much total length. Final attempt allocated more horizontal 
               space for spiral."
    connector:
      attempts: 8
      summary: "Could not find a spiral/staircase configuration that both 
               fits the available footprint AND connects smoothly to the 
               table edge. Core issue: table height (2.25m) requires too 
               many spiral turns for the available space (1.0m × 1.0m). 
               Recommendation: reduce table scale or move adjacent objects 
               to create more floor space."
```

---

## 5. Calibration Learning System

### Architecture

```
┌────────────────────────────────┐
│         SANDBOX                 │
│  Calibration loop for new       │
│  scenario types                 │
│  Max 10 iterations              │
│  Full pipeline render per iter  │
└────────────┬───────────────────┘
             │ Passes quality gate?
             ▼
┌────────────────────────────────┐
│       QUALITY GATE              │
│  Chain completion: 100%         │
│  Zero physics anomalies         │
│  Confidence scoring             │
│  Human review for first 20      │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│    KNOWLEDGE BASE (ChromaDB)    │
│  HttpClient → localhost:8000    │
│  Embeddings: nomic-embed-text   │
│    via Ollama                   │
│  Stores correction factors      │
│    (not absolute values)        │
│  Tagged with Blender version    │
└────────────────────────────────┘
```

### What Gets Calibrated

Calibration stores **correction factors** relative to formula-derived defaults:

```yaml
calibration_entry:
  scenario_descriptor: "spiral descent, 2.25m height, radius 0.3m, 3 turns"
  segment_type: spiral_descent
  
  corrections:
    spacing_correction: 0.92       # 8% tighter than formula
    friction_correction: 1.08      # 8% more friction than default
    trigger_force_correction: 1.15 # 15% more initial force needed
  
  metadata:
    confidence: 0.88
    iteration_count: 4
    blender_version: "4.1"
    date_calibrated: "2026-04-06"
    content_type: domino
```

### Lookup Flow

```python
# ChromaDB setup
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection(
    name="calibrations",
    metadata={"hnsw:space": "cosine"}
)

# Embeddings via Ollama
def embed(text):
    response = requests.post("http://localhost:11434/api/embed", json={
        "model": "nomic-embed-text",
        "input": text
    })
    return response.json()["embeddings"][0]
```

### Bootstrap Plan

**Phase 1 (current Blender capabilities):**
1. Straight line, 30 dominos
2. Straight line, 300 dominos
3. Gentle S-curve
4. Tight S-curve
5. Spiral, 2 turns
6. Spiral, 4 turns
7. Cascade, 150 dominos
8. Cascade, 400 dominos
9. Branching, 3 branches
10. Branching, 5 branches

**Phase 2 (after skill library expansion):**
11-15. Inclines/declines at various angles
16-18. Size profiles (increasing, decreasing, alternating)
19-20. Combinations (curve + incline, spiral + size change)

---

## 6. Skill Library

### Structure

```
skills/
├── shared/                       # Content-agnostic
│   ├── paths/
│   │   ├── straight_line.py
│   │   ├── arc.py
│   │   ├── spiral.py
│   │   ├── s_curve.py
│   │   └── staircase.py
│   ├── connectors/
│   │   ├── ramp.py
│   │   ├── spiral_ramp.py
│   │   ├── staircase.py
│   │   ├── platform.py
│   │   └── plank_bridge.py
│   ├── surfaces/
│   │   ├── flat.py
│   │   └── ramp_surface.py
│   └── environment/
│       ├── ground_textures.py    # Wood, concrete, grass, etc.
│       ├── lighting_presets.py   # Warm indoor, cool outdoor, dramatic
│       └── skybox.py             # HDRI for outdoor scenes
│
├── domino/                       # Domino-specific
│   ├── place_domino.py
│   ├── apply_rigid_body.py
│   ├── setup_trigger.py
│   ├── spacing_calculator.py
│   └── size_gradient.py
│
└── marble_run/                   # Marble run-specific (future)
    ├── pieces/
    │   ├── straight_track.py
    │   ├── curved_track.py
    │   ├── funnel.py
    │   └── loop.py
    ├── place_ball.py
    └── connector_validator.py
```

### Primitive Contract

Every skill library primitive follows a standard contract:

```yaml
primitive:
  name: create_spiral_ramp
  category: connector
  compatible_with: [domino, marble_run]
  
  inputs:
    - name: center
      type: Vector3
      description: "Center point of the spiral"
    - name: radius
      type: float
      range: [0.1, 2.0]
    - name: turns
      type: float
      range: [0.5, 10.0]
    - name: start_height
      type: float
    - name: end_height
      type: float
    - name: direction
      type: enum [clockwise, counterclockwise]
  
  outputs:
    - name: waypoints
      type: list[Vector3]
      description: "Dense waypoints along the spiral path"
    - name: mesh
      type: bpy.types.Object
      description: "The ramp mesh added to the Blender scene"
  
  test_criteria:
    - gradient_within_limits: true
    - mesh_no_self_intersection: true
    - waypoints_continuous: true
```

### Creating New Primitives

New primitives can be developed by the AI agent in the sandbox:

1. Human or agent defines the contract (inputs, outputs, test criteria)
2. Agent writes the implementation
3. Sandbox runs the primitive with test parameters
4. VLM validates the output against test criteria
5. Human reviews the first version
6. After initial review, future iterations of the same primitive can be auto-validated
7. Validated primitive is added to the skill library

---

## 7. Asset Catalogue

### Structure

Each asset in the catalogue has:

```yaml
asset:
  id: dining_table_01
  name: "Rustic Dining Table"
  source: polyhaven  # or custom, blenderkit, etc.
  file: assets/furniture/dining_table_01.blend
  license: CC0
  
  # Theme tags for set designer filtering
  themes: [dinner, kitchen, rustic, indoor]
  category: furniture
  
  # Physical properties
  dimensions: [1.2, 0.6, 0.75]  # L × W × H in metres (at scale 1.0)
  
  # Navigable surfaces (where dominos/marbles can travel)
  surfaces:
    - name: tabletop
      type: flat
      local_height: 0.75        # Height of surface relative to object origin
      bounds: [[0, 0], [1.2, 0.6]]  # 2D boundary polygon on the surface
      # No fixed entry/exit — pathfinder decides routing
  
  # Collision properties for physics
  collision_type: mesh  # or convex_hull, box
  is_static: true
  
  # Visual notes for the set designer
  description: "Weathered oak dining table, seats 6. Works well with 
               chair_rustic_01 and plate_ceramic_01."
```

### Asset Sources

- **Poly Haven:** CC0 licensed, high quality, downloadable via API. Primary source for realistic props.
- **Custom built:** Simple geometric pieces (ramps, platforms, planks) generated procedurally. Used for connectors.
- **Future:** BlenderKit free tier, Sketchfab CC-licensed models.

### Scale Reference

The set designer receives scale context relative to domino size:

```
Reference: Standard domino is 0.04m (4cm) tall.

At scale 1.0:
- A dining table (0.75m) is ~19 dominos tall
- A book (0.03m thick) is ~0.75 dominos tall
- A chair seat (0.45m) is ~11 dominos tall

At scale 3.0 (gigantified):
- A dining table (2.25m) is ~56 dominos tall
- A book (0.09m thick) is ~2.25 dominos tall
- A chair seat (1.35m) is ~34 dominos tall
```

---

## 8. Environment System

### Ground Textures

Predefined ground texture options mapped to themes:

```yaml
ground_textures:
  wooden_floorboards:
    themes: [indoor, dinner, bedroom, rustic]
    material: wood_floor_001
  
  concrete:
    themes: [outdoor, industrial, cityscape]
    material: concrete_001
  
  grass:
    themes: [outdoor, garden, park]
    material: grass_001
  
  tile:
    themes: [indoor, kitchen, bathroom]
    material: tile_white_001
  
  carpet:
    themes: [indoor, bedroom, living_room]
    material: carpet_001
```

### Lighting Presets

```yaml
lighting_presets:
  warm_indoor:
    type: area_light
    color: [1.0, 0.9, 0.8]
    intensity: 500
    skybox: none  # No skybox for indoor
    ambient: warm
    themes: [dinner, bedroom, rustic]
  
  cool_indoor:
    type: area_light
    color: [0.9, 0.95, 1.0]
    intensity: 600
    skybox: none
    ambient: neutral
    themes: [kitchen, bathroom, modern]
  
  daylight_outdoor:
    type: sun_light
    color: [1.0, 1.0, 0.95]
    intensity: 3
    skybox: outdoor_hdri_001  # HDRI for outdoor scenes
    themes: [outdoor, garden, cityscape]
  
  golden_hour:
    type: sun_light
    color: [1.0, 0.85, 0.6]
    intensity: 2
    skybox: golden_hour_hdri_001
    themes: [outdoor, dramatic]
```

### Rule: Indoor vs Outdoor

- Indoor themes → lighting preset (area/point lights), NO skybox, solid or textured ceiling optional
- Outdoor themes → sun light + HDRI skybox, no ceiling

The set designer specifies indoor/outdoor in the scene manifest. The environment system applies the appropriate setup.

---

## 9. Scene Bounds

Fixed bounding box per content type:

```yaml
scene_bounds:
  domino:
    size: [10, 10, 5]         # 10m × 10m × 5m
    course_length_target: 10  # metres (rough guide for set designer)
    
  marble_run:
    size: [20, 20, 10]        # Larger — marble runs need more elevation
    course_length_target: 15
```

The set designer receives these bounds and the approximate course length target. The pathfinder calculates the actual path length based on what the set designer places.

---

## 10. Tech Stack

| Component | Technology | Notes |
|---|---|---|
| Orchestrator | Kairos agent harness | Existing architecture |
| LLM (agents) | Local models via Ollama | All agent prompts run locally |
| VLM (validation) | Qwen3-VL-8B via Ollama | Lightweight per-step checks |
| VLM (final review) | Qwen3-VL-8B/30B via Ollama | Comprehensive end-to-end review |
| Embeddings | nomic-embed-text via Ollama | For ChromaDB similarity search |
| Knowledge base | ChromaDB | HttpClient → localhost:8000 (existing Docker service) |
| Blender execution | Headless CLI | `blender --background --python script.py` |
| Asset storage | Local filesystem | `assets/` directory with catalogue YAML |
| Session storage | Local filesystem | `sessions/{id}/` with full history |
| Skill library | Python modules | `skills/` directory with contracts |
| Config | Existing Kairos config | Feature flag: `calibration_enabled` |

---

## 11. Existing Codebase & Reuse Strategy

This is a refactor of the existing Kairos-agent codebase, not a greenfield build. The goal is to take what works from the existing code and adapt it to the new design. Anything no longer relevant should be removed.

### Code to Preserve and Adapt

| Existing Component | Reuse Strategy |
|---|---|
| LangGraph orchestration | Adapt as the multi-agent orchestrator. New agent nodes, same framework. |
| LiteLLM / Ollama integration | Keep as-is. All agents use this for LLM calls. |
| Blender headless execution (`asyncio.create_subprocess_exec`) | Keep as-is. This is the execution layer for all Blender work. |
| `generate_domino_course.py` | Refactor into skill library primitives. Extract path generation logic into separate modules. |
| `smoke_test_domino.py` | Extend for calibration validation. Keep existing mode, add calibration mode. |
| `video_review_agent.py` (Qwen3-VL) | Adapt for per-step validation and final review. Add calibration-specific prompts. |
| FFmpeg composition pipeline | Keep as-is for final render output. |
| `DominoCourseConfig` | Extend with calibration fields. Create ScenarioDescriptor for ChromaDB search. |
| Three-stream logging (`events.jsonl`, `console.jsonl`, `decisions.jsonl`) | Keep as-is. Extend to capture per-agent decision chains. |
| Docker setup (ChromaDB, Ollama, etc.) | Keep as-is. ChromaDB already running on port 8000. |
| ACE-Step / BEATs / audio pipeline | Out of scope for this refactor. Keep but don't modify — post-production is a separate pipeline. |

### Code to Remove

| Component | Reason |
|---|---|
| Pymunk engine and all related code (`engines/pymunk/`) | Replaced by Blender-only architecture. No longer needed. |
| Hard-coded physics overrides in `idea_agent.py` | Replaced by calibration system (after Phase 1 is proven). |
| `adjust_parameters()` in `simulation_agent.py` | Replaced by calibration-informed iteration. |
| Static `domino_archetypes.json` physics_defaults | Supplanted by ChromaDB calibrations. |
| Any pipeline code specific to Pymunk execution | Dead code after Pymunk removal. |

### Important: Feature Flag Transition

The existing locked-physics pipeline MUST continue working throughout the refactor. New functionality is feature-flagged (`calibration_enabled: bool = False`). Only after Phase 1 acceptance criteria are met should the flag be flipped. Only after Phase 3 acceptance criteria are met should the old pipeline code be removed.

---

## 12. Asset Sourcing

### Who Downloads Assets

**Kyle downloads assets manually.** The implementing agent cannot access external asset libraries (Poly Haven, BlenderKit, etc.) due to network restrictions. 

**Kyle's task:** Download 15-20 starter assets from Poly Haven (CC0 licensed) and place them in the `assets/models/` directory. Focus on:

- Furniture: 2-3 tables (dining, coffee, desk), 2-3 chairs, bookshelf
- Props: books, plates, cups, bottles, boxes
- Architectural: simple wall/block shapes, planks
- Decorative: vases, lamps, picture frames, rugs

**File format:** .blend files preferred (native Blender). .glb/.gltf also acceptable.

**The agent's task:** Once files are in `assets/models/`, the agent:
1. Scans the directory and inventories all available models
2. Opens each in Blender headless to extract dimensions and bounding boxes
3. Generates the catalogue YAML with metadata (dimensions, surfaces, theme tags)
4. Human reviews and adjusts theme tags / surface annotations as needed

### Asset Catalogue Bootstrap

The agent generates initial catalogue entries programmatically, but surface annotations (where dominos can run on each object) require human review for the first batch. After the first 15-20 assets are annotated, the agent can use those as examples to auto-annotate future assets with human spot-checks.

---

## 13. Implementation Phases

### Branch Strategy

Each phase is implemented on a dedicated branch, built sequentially off the previous phase's branch. Branches are committed but NOT pull-requested until all acceptance criteria are met and Kyle has reviewed.

```
main
  └── phase/0-skill-library
       └── phase/1-calibration
            └── phase/2-creative-pipeline
                 └── phase/3-physics-camera
                      └── phase/4-cleanup
                           └── phase/5-marble-run (future)
```

Each phase branch must pass its acceptance criteria before the next phase branch is created from it.

---

### Phase 0: Skill Library Foundation

**Branch:** `phase/0-skill-library`  
**Built from:** `main`

**Work:**
- Define the primitive contract schema (inputs, outputs, test criteria)
- Implement core path primitives: `straight_line`, `arc`, `spiral`, `s_curve`
- Implement core connector primitives: `ramp`, `spiral_ramp`, `staircase`, `platform`, `plank_bridge`
- Set up asset catalogue schema and write the catalogue generator script
- Set up environment system (ground textures, lighting presets, skybox logic)
- Set up the `skills/` directory structure (shared, domino, marble_run namespaces)
- Write primitive test harness (can run each primitive in isolation in Blender headless)

**Acceptance Criteria:**
- [ ] Each primitive can be instantiated in Blender headless with test parameters and produces valid geometry (no crashes, no degenerate meshes)
- [ ] `create_spiral_ramp(center=[0,0,0], radius=0.3, turns=3, start_height=2.0, end_height=0.0)` produces a visible spiral ramp in a .blend file
- [ ] `create_staircase(start=[0,0,0], end=[1,0,1], step_count=10)` produces visible stairs
- [ ] Asset catalogue generator scans `assets/models/`, opens each .blend, extracts dimensions, and writes catalogue YAML
- [ ] Environment system applies correct ground texture and lighting preset for a given theme keyword
- [ ] **Completed run:** A 30-domino straight-line course generated entirely from skill library primitives (not the old `generate_domino_course.py`), physics baked, and rendered to MP4

**Human dependencies:**
- Kyle: Download 15-20 assets to `assets/models/` before this phase starts
- Kyle: Review generated asset catalogue YAML (especially surface annotations)

---

### Phase 1: Calibration System

**Branch:** `phase/1-calibration`  
**Built from:** `phase/0-skill-library`

**Work:**
- Wire up ChromaDB via `HttpClient(host="localhost", port=8000)`
- Implement embedding via Ollama's `nomic-embed-text` (`/api/embed` endpoint)
- Implement ScenarioDescriptor schema (the ChromaDB search key)
- Implement calibration sandbox (iteration loop, parameter adjustment, feedback routing)
- Implement quality gate (chain completion, anomaly detection, confidence scoring)
- Implement ChromaDB storage and lookup (store corrections, query by similarity)
- Implement composite parameter blending (combine corrections from partial matches)
- Extend `smoke_test_domino.py` with calibration mode (richer output, more frames)
- Add `calibration_enabled` feature flag to Settings
- Run the 10 Phase 1 bootstrap scenarios

**Acceptance Criteria:**
- [ ] ChromaDB connection works — can store and retrieve a test calibration entry
- [ ] Embedding pipeline works — scenario descriptor embeds via Ollama and returns a vector
- [ ] Calibration sandbox runs a straight-line scenario, iterates if chain breaks, stores successful calibration
- [ ] Quality gate correctly blocks a deliberately bad calibration (e.g., 50% chain completion)
- [ ] Composite lookup works — querying a "gentle s-curve" returns the stored s-curve calibration as the top match
- [ ] **Completed run:** All 10 Phase 1 bootstrap scenarios calibrated and stored in ChromaDB. At least 8 of 10 must pass quality gate within 5 iterations. The remaining 2 may take up to 10 iterations.
- [ ] **Composite test:** Run a new scenario (e.g., tight spiral) that wasn't in the bootstrap. Verify it retrieves relevant calibrations and uses them as starting parameters. It should converge faster than a scenario with no prior data.

**Human dependencies:**
- Kyle: ChromaDB Docker service running on port 8000
- Kyle: Ollama running with `nomic-embed-text` model pulled
- Kyle: Review first 10 calibration results for quality (human spot-check)

---

### Phase 2: Creative Pipeline Agents

**Branch:** `phase/2-creative-pipeline`  
**Built from:** `phase/1-calibration`

**Work:**
- Implement Set Designer agent (prompt, asset catalogue access, theme selection, object placement)
- Implement Path Setter agent (surface metadata reasoning, spline routing, gradient validation)
- Implement Connector Agent (skill library access, ChromaDB calibration lookup, connector selection and placement)
- Implement per-step validation agent (checks after each of the three agents)
- Wire up the sequential agent pipeline with handoffs (Agent 1 output → Agent 2 input → Agent 3 input)
- Implement cascade re-run logic (failure attribution → re-run from responsible agent)
- Implement iteration history and feedback format (human-readable summaries per attempt)
- Implement session storage structure (`sessions/{id}/agents/...`)

**Acceptance Criteria:**
- [ ] Set Designer produces a valid scene manifest given an asset catalogue and theme
- [ ] Path Setter routes a spline through the placed objects, flags transitions needing connectors
- [ ] Connector Agent fills flagged transitions with appropriate primitives from the skill library
- [ ] Per-step validator catches a deliberately introduced error (e.g., overlapping objects) and attributes it correctly
- [ ] Cascade logic works — a connector failure triggers re-run from Agent 3 only, not Agent 1
- [ ] Feedback to agents includes history of previous attempts in human-readable summary format
- [ ] **Completed run:** Full creative pipeline produces a scene manifest → routed path → connected path → placed dominos → physics bake → validated chain completion. Minimum one successful end-to-end run with at least 3 functional objects at different elevations and at least 2 connector pieces.
- [ ] **Failure recovery test:** Introduce a scenario that fails on first attempt (e.g., too-steep gradient). Verify the system retries with adjusted parameters and eventually succeeds within 10 total pipeline iterations.

**Human dependencies:**
- Kyle: Review Set Designer agent's creative output quality (is it generating interesting, varied scenes?)
- Kyle: Review at least 3 full pipeline runs for overall quality

---

### Phase 3: Physics + Camera Pipeline

**Branch:** `phase/3-physics-camera`  
**Built from:** `phase/2-creative-pipeline`

**Work:**
- Integrate placement agent with calibration system (lookup before placement, store corrections after)
- Implement camera router (third-person follow, wavefront tracking, speed matching)
- Implement occlusion detection (ray casting from camera to wavefront)
- Implement smooth camera transitions (bezier interpolation during repositioning)
- Implement camera validator (VLM checks visibility, smooth motion, composition)
- Implement final reviewer agent (watches full rendered run, attributes failures across all agents)
- Wire up final reviewer's cascade logic (failure → route to responsible agent → re-run downstream)

**Acceptance Criteria:**
- [ ] Camera follows domino wavefront at matching speed along the path
- [ ] Camera repositions smoothly when occluded (no hard cuts, lerp over 30-60 frames)
- [ ] Camera validator flags a deliberately occluded sequence (e.g., camera behind a wall)
- [ ] Final reviewer watches a complete rendered run and produces a pass/fail with attribution
- [ ] Final reviewer correctly identifies a physics failure vs a camera failure and routes feedback to the right agent
- [ ] **Completed run:** Full pipeline from Set Designer through to final rendered MP4, including camera follow, occlusion avoidance, and final review pass. The rendered video must show dominos visible throughout, smooth camera motion, and complete chain propagation. Minimum two successful end-to-end runs with different themes.
- [ ] **Attribution test:** Deliberately introduce a chain break (physics failure) AND a camera occlusion in the same run. Verify the final reviewer identifies both issues and attributes them to different agents.

**Human dependencies:**
- Kyle: Review rendered videos for camera quality (is the third-person follow working well?)
- Kyle: Final sign-off that the pipeline produces upload-ready content

---

### Phase 4: Cleanup & Consolidation

**Branch:** `phase/4-cleanup`  
**Built from:** `phase/3-physics-camera`

**Work:**
- Remove Pymunk engine and ALL related code (`engines/pymunk/`, pymunk pipeline, pymunk configs)
- Remove hard-coded physics overrides from `idea_agent.py` (now replaced by calibration)
- Remove `adjust_parameters()` from `simulation_agent.py` (now replaced by calibration)
- Remove static `domino_archetypes.json` physics_defaults (now in ChromaDB)
- Remove any dead imports, unused utilities, orphaned config entries
- Update all documentation (README, CLAUDE.md, inline comments) to reflect new architecture
- Verify no existing tests break after removal
- Ensure `calibration_enabled` flag can be safely set to `True` as default
- Clean up any temporary scaffolding or compatibility shims from earlier phases

**Acceptance Criteria:**
- [ ] No references to Pymunk anywhere in the codebase (`grep -r "pymunk"` returns zero results)
- [ ] No references to removed override code (`grep -r "adjust_parameters"` returns zero)
- [ ] All existing tests pass
- [ ] `calibration_enabled` defaults to `True`
- [ ] **Completed run:** Full pipeline end-to-end with calibration enabled by default, producing a rendered MP4 identical in quality to Phase 3 runs. This verifies nothing broke during cleanup.
- [ ] Codebase passes a basic lint/import check — no broken imports from removed modules

**Human dependencies:**
- Kyle: Final review of removed code list before deletion (agent should present the list and wait for approval)

---

### Phase 5: Marble Run Content Type (Future)

**Branch:** `phase/5-marble-run`  
**Built from:** `phase/4-cleanup`

**Work:**
- Design marble run skill library (track pieces with connector system)
- Implement connector validator (diameter matching, direction alignment, momentum checks)
- Implement marble-specific placement agent
- Implement marble-specific calibration entries in ChromaDB
- Adapt camera system for smaller objects (closer follow distance, faster tracking)
- Bootstrap marble run calibrations (10 scenarios)

**Acceptance Criteria:**
- [ ] Marble track pieces connect via the connector system (no gaps, no misaligned joints)
- [ ] Ball completes a 10-piece track without flying off
- [ ] Camera follows the ball at appropriate distance
- [ ] **Completed run:** Full pipeline producing a marble run video with at least 5 different track piece types, elevation changes, and smooth camera follow. Rendered to MP4.

**Human dependencies:**
- Kyle: Design approval on marble track piece library (what pieces to build)
- Kyle: Review first marble run renders for quality

---

### Phase 6: Expansion (Future)

**Branch:** `phase/6-expansion`  
**Built from:** `phase/5-marble-run`

**Work:**
- Grow asset catalogue to 50+ assets
- Grow skill library with new connector types and path types
- Implement AI-assisted primitive development (agent writes new skills in sandbox)
- Mixed content scenes (dominos triggering marble runs — future goal)

**Acceptance criteria:** To be defined when this phase is scoped.

---

## 14. Human Dependencies Summary

Things the implementing agent cannot do and will need from Kyle:

| Dependency | When Needed | Details |
|---|---|---|
| Download 15-20 3D assets | Before Phase 0 | From Poly Haven (CC0). Place in `assets/models/`. Tables, chairs, books, props, architectural pieces. |
| Review asset catalogue YAML | During Phase 0 | Agent generates metadata. Kyle reviews surface annotations and theme tags. |
| ChromaDB running on port 8000 | Before Phase 1 | Existing Docker service. Verify it's up. |
| Ollama with required models | Before Phase 1 | `nomic-embed-text` for embeddings. Coding model and VLM already in use. |
| Review calibration results | During Phase 1 | Spot-check first 10 calibrated scenarios for quality. |
| Review creative agent output | During Phase 2 | Is the Set Designer producing interesting, varied scenes? |
| Review rendered videos | During Phase 3 | Camera quality, overall visual quality, upload-readiness. |
| Approve code deletion list | During Phase 4 | Agent presents list of files/code to remove. Kyle approves before deletion. |

---

## 15. Notes for Implementing Agent

### Codebase Context

This is a refactor of the existing **kairos-agent** repository. Before starting any phase:
1. Read the existing codebase thoroughly — understand current file structure, imports, and data flow
2. Identify which existing modules map to new components (see Section 11)
3. Preserve working functionality throughout — the old pipeline must work until the new one is proven
4. Use feature flags, not big-bang replacements

### Key Architectural Constraints

- **Local-first:** All LLM inference via Ollama. No external API calls for production runs.
- **Sequential model loading:** Kyle's RTX 3090 (24GB VRAM) loads models one at a time. Don't assume concurrent GPU model loading.
- **Blender 5.0.1:** EEVEE renderer, Bullet physics engine. Verify Blender Python API compatibility.
- **LangGraph:** Existing orchestration framework. New agents should be LangGraph nodes.
- **LiteLLM:** Existing LLM routing layer. New agent LLM calls should go through LiteLLM.

### What NOT to Do

- Don't create a parallel schema when you can extend an existing one
- Don't add new dependencies when existing tools in the stack can handle it
- Don't remove working code until the replacement is proven (feature flag first)
- Don't assume network access to external services beyond the allowlisted domains
- Don't hard-code model names — use the existing LiteLLM routing configuration
- Don't create separate repos or services — everything lives in kairos-agent

---

## 16. Open Questions (For Future Iterations)

1. **Post-production integration:** Separate agent pipeline consuming theme manifest for sound/music. To be specced independently.
2. **Mixed content scenes:** Dominos triggering marble runs in the same scene. Requires cross-content-type connector primitives.
3. **Procedural asset generation:** AI generating simple geometric assets (custom furniture shapes, abstract structures) rather than relying on pre-made models.
4. **Multi-camera:** Multiple camera angles with cuts rather than single continuous shot. May suit certain content styles.
5. **Community asset sharing:** If the system works well, sharing calibration data and asset catalogues between instances.
