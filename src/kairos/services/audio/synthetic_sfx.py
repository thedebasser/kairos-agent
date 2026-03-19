"""Synthetic domino collision SFX generator.

Produces short percussive "click/clack/tap" sounds using pure synthesis
when no Freesound API key is available. This ensures every video has
collision audio regardless of external API availability.

Each theme maps to a different synthesis recipe (wood, metal, plastic, etc).
Uses layered sine harmonics with shaped envelopes — NO random noise —
to produce clean, realistic impact sounds.
"""

from __future__ import annotations

import logging
import math
import random
import struct
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
SYNTH_SFX_DIR = _PROJECT_ROOT / "assets" / "sfx" / "synthetic"

SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def _write_wav(path: Path, samples: list[float], sample_rate: int = SAMPLE_RATE) -> None:
    """Write normalised float samples (-1..1) to a 16-bit WAV file."""
    peak = max(abs(s) for s in samples) if samples else 1.0
    if peak == 0:
        peak = 1.0
    scale = 32767 / peak * 0.9  # leave a bit of headroom

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        raw = struct.pack(
            f"<{len(samples)}h",
            *(int(s * scale) for s in samples),
        )
        wf.writeframes(raw)


def _generate_impact(
    duration_ms: int = 60,
    fundamental_hz: float = 2000.0,
    attack_ms: float = 0.5,
    body_decay: float = 60.0,
    tail_decay: float = 15.0,
    tail_mix: float = 0.3,
    harmonics: list[tuple[float, float]] | None = None,
    body_freq_hz: float = 0.0,
) -> list[float]:
    """Generate a clean impact / tap / click sound using layered harmonics.

    Uses a two-stage envelope (sharp attack body + softer resonant tail)
    with harmonic overtones for tonal colour. No random noise.

    Args:
        duration_ms: Total length in milliseconds.
        fundamental_hz: Base frequency of the impact body.
        attack_ms: Attack transient duration (very short = sharp click).
        body_decay: Exponential decay for the body (higher = faster fade).
        tail_decay: Slower decay for the resonant tail / ring.
        tail_mix: Mix of tail relative to body (0-1).
        harmonics: List of (freq_multiplier, amplitude) for overtones.
            Default: [(1.0, 1.0), (2.0, 0.5), (3.0, 0.25), (5.0, 0.1)]
        body_freq_hz: Optional separate frequency for the body thump.
            If 0, uses fundamental_hz.
    """
    if harmonics is None:
        harmonics = [(1.0, 1.0), (2.0, 0.5), (3.0, 0.25), (5.0, 0.1)]

    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    attack_samples = max(1, int(SAMPLE_RATE * attack_ms / 1000))
    body_f = body_freq_hz if body_freq_hz > 0 else fundamental_hz
    samples: list[float] = []

    for i in range(n_samples):
        t = i / SAMPLE_RATE

        # Attack envelope: instant rise then fast decay
        if i < attack_samples:
            attack_env = i / attack_samples
        else:
            attack_env = 1.0

        # Body: sharp exponential decay (the initial "click/tap")
        body_env = attack_env * math.exp(-body_decay * t)
        body = math.sin(2.0 * math.pi * body_f * t)
        # Add a bit of 2nd harmonic for body richness
        body += 0.3 * math.sin(2.0 * math.pi * body_f * 2.0 * t)

        # Tail: slower resonant ring with harmonics
        tail_env = attack_env * math.exp(-tail_decay * t)
        tail = 0.0
        for mult, amp in harmonics:
            tail += amp * math.sin(2.0 * math.pi * fundamental_hz * mult * t)

        sample = body_env * body + tail_mix * tail_env * tail
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# Theme-specific synthesis recipes
# ---------------------------------------------------------------------------

_RECIPES: dict[str, dict] = {
    "wood": {
        "duration_ms": 55,
        "fundamental_hz": 1600.0,
        "attack_ms": 0.3,
        "body_decay": 80.0,
        "tail_decay": 25.0,
        "tail_mix": 0.2,
        "harmonics": [(1.0, 1.0), (2.3, 0.4), (4.1, 0.15)],
        "body_freq_hz": 800.0,
    },
    "metal": {
        "duration_ms": 80,
        "fundamental_hz": 1800.0,
        "attack_ms": 0.4,
        "body_decay": 50.0,
        "tail_decay": 12.0,
        "tail_mix": 0.3,
        "harmonics": [(1.0, 1.0), (2.0, 0.35), (3.5, 0.12)],
        "body_freq_hz": 900.0,
    },
    "plastic": {
        "duration_ms": 50,
        "fundamental_hz": 1600.0,
        "attack_ms": 0.3,
        "body_decay": 80.0,
        "tail_decay": 25.0,
        "tail_mix": 0.15,
        "harmonics": [(1.0, 1.0), (2.0, 0.25), (3.0, 0.08)],
        "body_freq_hz": 800.0,
    },
    "stone": {
        "duration_ms": 50,
        "fundamental_hz": 2000.0,
        "attack_ms": 0.3,
        "body_decay": 90.0,
        "tail_decay": 20.0,
        "tail_mix": 0.25,
        "harmonics": [(1.0, 1.0), (1.5, 0.5), (3.0, 0.2), (4.5, 0.1)],
        "body_freq_hz": 600.0,
    },
    "heavy": {
        "duration_ms": 90,
        "fundamental_hz": 600.0,
        "attack_ms": 0.5,
        "body_decay": 40.0,
        "tail_decay": 10.0,
        "tail_mix": 0.35,
        "harmonics": [(1.0, 1.0), (2.0, 0.6), (3.0, 0.3)],
        "body_freq_hz": 300.0,
    },
    "sci-fi": {
        "duration_ms": 80,
        "fundamental_hz": 5000.0,
        "attack_ms": 0.1,
        "body_decay": 50.0,
        "tail_decay": 12.0,
        "tail_mix": 0.5,
        "harmonics": [(1.0, 1.0), (1.618, 0.7), (2.618, 0.4), (4.236, 0.2)],
        "body_freq_hz": 3000.0,
    },
}

# Map theme names → recipe key
_THEME_RECIPE_MAP: dict[str, str] = {
    "deep_space": "sci-fi",
    "enchanted_forest": "wood",
    "golden_hour": "stone",
    "arctic_lab": "plastic",
    "neon_city": "metal",
    "candy_land": "plastic",
    "lava_world": "heavy",
}


def generate_synthetic_pool(
    theme_name: str,
    count: int = 8,
) -> list[Path]:
    """Generate a pool of varied synthetic collision sounds for a theme.

    Each sound is a slight variation of the theme's base recipe,
    using harmonic synthesis (no noise) for clean, realistic impacts.

    Args:
        theme_name: Theme name (maps to a synthesis recipe).
        count: Number of variations to generate.

    Returns:
        List of paths to generated WAV files.
    """
    SYNTH_SFX_DIR.mkdir(parents=True, exist_ok=True)

    recipe_key = _THEME_RECIPE_MAP.get(theme_name, "wood")
    base = _RECIPES.get(recipe_key, _RECIPES["wood"])

    paths: list[Path] = []
    for i in range(count):
        # Apply per-variation jitter to create natural variety
        duration = base["duration_ms"] + random.randint(-8, 12)
        fundamental = base["fundamental_hz"] * random.uniform(0.88, 1.15)
        body_decay = base["body_decay"] * random.uniform(0.85, 1.2)
        tail_decay = base["tail_decay"] * random.uniform(0.8, 1.25)
        tail_mix = max(0.0, min(1.0, base["tail_mix"] + random.uniform(-0.05, 0.05)))
        body_freq = base.get("body_freq_hz", 0) * random.uniform(0.9, 1.12)

        # Jitter harmonics slightly for each variation
        base_harmonics = base.get("harmonics", [(1.0, 1.0), (2.0, 0.5), (3.0, 0.25)])
        harmonics = [
            (mult * random.uniform(0.97, 1.03), amp * random.uniform(0.85, 1.15))
            for mult, amp in base_harmonics
        ]

        samples = _generate_impact(
            duration_ms=max(25, duration),
            fundamental_hz=fundamental,
            attack_ms=base.get("attack_ms", 0.3),
            body_decay=body_decay,
            tail_decay=tail_decay,
            tail_mix=tail_mix,
            harmonics=harmonics,
            body_freq_hz=body_freq,
        )

        out = SYNTH_SFX_DIR / f"synth_{theme_name}_{i:03d}.wav"
        _write_wav(out, samples)
        paths.append(out)

    logger.info(
        "[synth_sfx] Generated %d synthetic SFX for theme '%s' (recipe=%s)",
        len(paths), theme_name, recipe_key,
    )
    return paths
