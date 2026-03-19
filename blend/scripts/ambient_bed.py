"""Theme-specific ambient bed generator for domino videos.

Generates subtle ambient backgrounds that prevent the "dead digital void"
between collision hits, matching the oddly-satisfying / ASMR-adjacent style.

Each theme gets a unique ambient texture:
- Room tone (very subtle frequency hum)
- Environmental motif (rain, air, distant hum)
- Low-passed to sit behind the collision clicks (psychoacoustic depth)

Design principle: ambient sits "behind" the clicks. Rolling off highs
pushes ambience back perceptually, leaving clicks forward without
raising click loudness excessively.
"""

from __future__ import annotations

import math
import os
import random
import struct
import wave
from pathlib import Path
from typing import Any


OUTPUT_SAMPLE_RATE = 48000

# ── Ambient presets per theme ───────────────────────────────────────────────
# Each preset generates a unique ambient texture.
# amplitude values are intentionally very low — ambience should be felt, not heard.

AMBIENT_PRESETS: dict[str, dict[str, Any]] = {
    "wood": {
        # Cozy workshop: warm low hum (no noise texture — avoids white-noise)
        "room_tone_hz": 80,
        "room_tone_amplitude": 0.015,
        "texture": "none",
        "texture_amplitude": 0.0,
        "texture_cutoff_hz": 800,
        "drift_rate": 0.3,
        "total_amplitude": 0.018,
    },
    "plastic": {
        # Clean studio: barely audible 60 Hz hum (no noise)
        "room_tone_hz": 60,
        "room_tone_amplitude": 0.010,
        "texture": "none",
        "texture_amplitude": 0.0,
        "texture_cutoff_hz": 600,
        "drift_rate": 0.15,
        "total_amplitude": 0.012,
    },
    "ceramic": {
        # Zen space: soft tonal bed
        "room_tone_hz": 100,
        "room_tone_amplitude": 0.012,
        "texture": "none",
        "texture_amplitude": 0.0,
        "texture_cutoff_hz": 800,
        "drift_rate": 0.4,
        "total_amplitude": 0.015,
    },
    "metal": {
        # Sci-fi environment: deep electronic hum
        "room_tone_hz": 50,
        "room_tone_amplitude": 0.018,
        "texture": "none",
        "texture_amplitude": 0.0,
        "texture_cutoff_hz": 600,
        "drift_rate": 0.2,
        "total_amplitude": 0.020,
    },
    "heavy": {
        # Lava world: rumbling low-end undertone
        "room_tone_hz": 40,
        "room_tone_amplitude": 0.020,
        "texture": "none",
        "texture_amplitude": 0.0,
        "texture_cutoff_hz": 500,
        "drift_rate": 0.5,
        "total_amplitude": 0.025,
    },
}

# Map themes to ambient preset
THEME_AMBIENT_MAP: dict[str, str] = {
    "deep_space": "metal",
    "enchanted_forest": "wood",
    "golden_hour": "ceramic",
    "arctic_lab": "plastic",
    "neon_city": "metal",
    "candy_land": "plastic",
    "lava_world": "heavy",
}


# ── Texture generators ──────────────────────────────────────────────────────

def _generate_filtered_noise(
    n_samples: int,
    cutoff_hz: float,
    sample_rate: int,
    rng: random.Random,
) -> list[float]:
    """Generate low-passed noise using a simple 1-pole filter.

    This creates the ambient "texture" bed — rain, air, wind, etc.
    The low-pass ensures it sits behind the collision transients.
    """
    # Simple 1-pole low-pass: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    samples = []
    prev = 0.0
    for _ in range(n_samples):
        x = rng.gauss(0, 1)
        prev = alpha * x + (1 - alpha) * prev
        samples.append(prev)

    return samples


def _generate_room_tone(
    n_samples: int,
    freq_hz: float,
    amplitude: float,
    sample_rate: int,
) -> list[float]:
    """Generate a subtle room tone (low-frequency hum).

    Uses layered low sines to create a believable room presence.
    """
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        # Primary hum
        val = math.sin(2 * math.pi * freq_hz * t) * amplitude
        # Subtle 2nd harmonic
        val += math.sin(2 * math.pi * freq_hz * 2 * t) * amplitude * 0.3
        # Very subtle 3rd
        val += math.sin(2 * math.pi * freq_hz * 3 * t) * amplitude * 0.1
        samples.append(val)
    return samples


def _apply_drift(
    samples: list[float],
    drift_rate: float,
    sample_rate: int,
    rng: random.Random,
) -> list[float]:
    """Apply slow amplitude drift to prevent static feel.

    Modulates volume with a very low-frequency envelope so the
    ambience breathes naturally.
    """
    # LFO: very slow random walk
    n = len(samples)
    envelope = [1.0] * n
    current = 1.0
    step_size = drift_rate / sample_rate

    for i in range(n):
        current += rng.gauss(0, step_size)
        current = max(0.5, min(1.5, current))
        envelope[i] = current

    return [s * e for s, e in zip(samples, envelope)]


# ── Main generator ──────────────────────────────────────────────────────────

def generate_ambient_bed(
    output_wav: str,
    duration_sec: float,
    theme_name: str | None = None,
    material_name: str | None = None,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    fade_in_sec: float = 2.0,
    fade_out_sec: float = 3.0,
) -> dict[str, Any]:
    """Generate a theme-appropriate ambient bed WAV file.

    Args:
        output_wav: Output WAV path.
        duration_sec: Duration in seconds.
        theme_name: Theme name to select ambient preset.
        material_name: Material name (fallback if theme not found).
        sample_rate: Output sample rate.
        fade_in_sec: Fade-in duration at start.
        fade_out_sec: Fade-out duration at end.

    Returns:
        Dict with status, path, duration, peak level.
    """
    # Resolve preset
    preset_key = "wood"  # default
    if theme_name:
        preset_key = THEME_AMBIENT_MAP.get(theme_name, "wood")
    elif material_name:
        preset_key = material_name if material_name in AMBIENT_PRESETS else "wood"

    preset = AMBIENT_PRESETS[preset_key]
    rng = random.Random(123)

    n_samples = int(duration_sec * sample_rate)

    # Generate room tone
    room_tone = _generate_room_tone(
        n_samples,
        preset["room_tone_hz"],
        preset["room_tone_amplitude"],
        sample_rate,
    )

    # Generate texture layer (filtered noise) — only if preset requests it
    tex_amp = preset["texture_amplitude"]
    if tex_amp > 0 and preset.get("texture") != "none":
        texture = _generate_filtered_noise(
            n_samples,
            preset["texture_cutoff_hz"],
            sample_rate,
            rng,
        )
        texture = [s * tex_amp for s in texture]
        mixed = [(r + t) for r, t in zip(room_tone, texture)]
    else:
        # Room-tone only — no noise texture (avoids audible white noise)
        mixed = list(room_tone)

    total_amp = preset["total_amplitude"]

    # Apply drift
    mixed = _apply_drift(mixed, preset["drift_rate"], sample_rate, rng)

    # Normalize to target amplitude
    peak = max((abs(s) for s in mixed), default=1.0)
    if peak > 0:
        scale = total_amp / peak
        mixed = [s * scale for s in mixed]

    # Apply fade in/out
    fade_in_samples = int(fade_in_sec * sample_rate)
    fade_out_samples = int(fade_out_sec * sample_rate)
    for i in range(min(fade_in_samples, n_samples)):
        mixed[i] *= i / fade_in_samples
    for i in range(min(fade_out_samples, n_samples)):
        idx = n_samples - 1 - i
        if idx >= 0:
            mixed[idx] *= i / fade_out_samples

    # Write stereo WAV (same content on both channels)
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_wav):
        os.remove(output_wav)

    final_peak = max((abs(s) for s in mixed), default=0.0)

    int_samples = [max(-32767, min(32767, int(s * 32767))) for s in mixed]
    with wave.open(output_wav, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Interleave stereo
        stereo = []
        for s in int_samples:
            stereo.append(s)
            stereo.append(s)
        wf.writeframes(struct.pack(f"<{len(stereo)}h", *stereo))

    wav_size = os.path.getsize(output_wav)
    print(f"[ambient_bed] Generated {preset_key} ambient: {output_wav} "
          f"({wav_size:,} bytes, {duration_sec:.1f}s, peak={final_peak:.4f})")

    return {
        "status": "ok",
        "wav_path": output_wav,
        "wav_size_bytes": wav_size,
        "duration_sec": duration_sec,
        "peak_amplitude": round(final_peak, 4),
        "preset": preset_key,
        "texture": preset["texture"],
    }
