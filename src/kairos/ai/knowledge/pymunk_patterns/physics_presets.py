# ============================================================================
# Kairos — Physics Presets per Scenario Category
# ============================================================================
# Codified parameter dictionaries from tested simulations.
# These are injected into prompts so the LLM uses proven values.
# ============================================================================

BALL_PIT = {
    "name": "ball_pit",
    "description": "Satisfying bouncing — balls filling containers, cascading, colliding",
    "space": {
        "gravity": (0, 900),
        "damping": 0.99,
        "iterations": 10,
    },
    "shape": {
        "radius_range": (15, 30),
        "mass_range": (1.0, 3.0),
        "elasticity": 0.7,           # Effective bounce: 0.7 * wall_elasticity
        "friction": 0.4,
    },
    "walls": {
        "elasticity": 0.5,           # Effective bounce with ball: 0.7 * 0.5 = 0.35
        "friction": 0.6,
    },
    "limits": {
        "max_bodies": 200,
        "body_count_initial": "1-5",
        "body_count_max": "100-200",
        "spawn_rate_calm": 1,        # bodies/sec in intro phase
        "spawn_rate_build": 4,       # bodies/sec in build phase
        "spawn_rate_peak": 6,        # bodies/sec in climax phase
    },
    "notes": [
        "Lower gravity (600) gives more airtime for visual appeal",
        "Elasticity is MULTIPLIED: ball(0.7) × wall(0.5) = effective 0.35 bounce",
        "Use pymunk.moment_for_circle(mass, 0, radius) — NEVER guess moment",
    ],
}

DOMINO_CHAIN = {
    "name": "domino_chain",
    "description": "Chain reactions — dominoes toppling in sequence",
    "space": {
        "gravity": (0, 981),
        "damping": 0.99,
        "iterations": 20,            # Higher iterations prevent mushy stacks
    },
    "shape": {
        "domino_width": 20,
        "domino_height": 60,
        "spacing_center_to_center": "35-45",  # 0.6× height rule
        "mass": 1.0,
        "elasticity": 0.1,           # Low bounce = clean toppling
        "friction": 0.7,
    },
    "walls": {
        "elasticity": 0.2,
        "friction": 0.8,
    },
    "trigger": {
        "method": "apply_impulse_at_local_point((300, 0), (0, -25))",
        "description": "Push first domino at top edge, horizontal force",
        "delay_sec": 3,
    },
    "limits": {
        "max_bodies": 150,
        "body_count_initial": "all (placed at start)",
        "layout": "single ground plane, straight/S-curve/gentle arc",
    },
    "notes": [
        "Spacing MUST be > domino width (20px) so they can fall into gaps",
        "Optimal spacing: 0.6× domino height = 36px center-to-center",
        "Use 2 substeps per frame: space.step(1/60) × 2 for 30fps",
        "Elasticity 0.2-0.3 gives clean toppling without excessive bouncing",
        "Do NOT use spirals or multi-level — keep it simple on one ground plane",
    ],
}

DESTRUCTION = {
    "name": "destruction",
    "description": "Structures built up then dramatically destroyed",
    "space": {
        "gravity": (0, 900),
        "damping": 0.99,
        "iterations": 25,            # High iterations for stable stacking
    },
    "shape": {
        "block_width": "60-100",
        "block_height": "30-50",
        "mass": 2.0,
        "elasticity": 0.1,           # Low bounce for stable stocking
        "friction": 0.8,
    },
    "walls": {
        "elasticity": 0.3,
        "friction": 0.8,
    },
    "wrecking_ball": {
        "mass": 80,
        "radius": 50,
        "velocity": "(±1500, 0)",
        "elasticity": 0.3,
    },
    "pre_settle": {
        "method": "Run space.step(1/60) for 300+ iterations before rendering",
        "damping_override": 0.8,
        "sleep_threshold": 0.5,
    },
    "limits": {
        "max_bodies": 200,
        "structure": "10-15 layers high, 3-5 blocks wide",
        "no_joints_needed": True,
    },
    "notes": [
        "NO joints needed — just stack blocks and let gravity + friction hold them",
        "Pre-settle is MANDATORY — structure must be stable before recording",
        "Temporarily set damping=0.8 during settle phase for faster stabilisation",
        "Dust particles: tiny circles (radius 2-5, mass 0.05) at impact point",
        "Wrecking ball enters from screen edge at climax time",
    ],
}

MARBLE_FUNNEL = {
    "name": "marble_funnel",
    "description": "Marbles rolling through funnels, ramps, spiral tracks",
    "space": {
        "gravity": (0, 900),
        "damping": 0.99,
        "iterations": 15,
    },
    "shape": {
        "radius_range": (10, 20),
        "mass_range": (1.0, 2.0),
        "elasticity": 0.5,
        "friction": 0.6,             # Both ball AND surface need >= 0.5 for rolling
    },
    "walls": {
        "elasticity": 0.3,
        "friction": 0.7,
        "ramp_construction": "pymunk.Segment(static_body, (x1,y1), (x2,y2), 5)",
    },
    "limits": {
        "max_bodies": 200,
        "funnel_stages": "3-5 angled segments converging",
        "spawn_from": "top of screen with slight horizontal variation",
    },
    "notes": [
        "Both marble AND surface need friction >= 0.5 for satisfying rolling",
        "Ramps built from pymunk.Segment between two endpoints",
        "Funnels: two angled segments converging toward a center point",
        "Remove marbles that exit below screen bottom",
        "Marble elasticity 0.5 gives satisfying bounces without too much chaos",
    ],
}

# All presets indexed by category name
ALL_PRESETS = {
    "ball_pit": BALL_PIT,
    "domino_chain": DOMINO_CHAIN,
    "destruction": DESTRUCTION,
    "marble_funnel": MARBLE_FUNNEL,
}


def format_preset_for_prompt(category: str) -> str:
    """Format a physics preset as human-readable text for prompt injection."""
    preset = ALL_PRESETS.get(category)
    if not preset:
        return ""

    lines = [f"## Physics Preset: {preset['description']}", ""]

    # Space settings
    s = preset["space"]
    lines.append(f"- space.gravity = {s['gravity']}")
    lines.append(f"- space.damping = {s['damping']}")
    lines.append(f"- space.iterations = {s['iterations']}")
    lines.append("")

    # Shape settings
    sh = preset["shape"]
    for k, v in sh.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Wall settings
    w = preset["walls"]
    for k, v in w.items():
        lines.append(f"- wall {k}: {v}")
    lines.append("")

    # Notes
    lines.append("### Important Notes:")
    for note in preset.get("notes", []):
        lines.append(f"- {note}")

    return "\n".join(lines)
