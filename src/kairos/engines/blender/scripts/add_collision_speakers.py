"""Material-driven collision audio mixer for domino videos.

Implements the sound design research: ASMR-adjacent foley with
material-specific layered synthesis (transient + body + tail),
full anti-machine-gun variation stack, and energy mapping.

Bypasses Blender's audio engine entirely — bpy.ops.sound.mixdown()
only processes VSE sequencer strips, NOT 3D Speaker objects, so we do
the mixing in pure Python for guaranteed audible output.

Sound design references:
- Transient/body/tail percussion model (signalflux.org)
- Anti-machine-gun: shuffle-bag, gain/pitch/start-offset jitter (SOS, musicradar)
- Energy mapping: tip-gap → intensity proxy
- ASMR-adjacent: near-field, darker timbre, controlled brightness
"""

from __future__ import annotations

import array
import json
import math
import os
import random
import struct
import wave
from pathlib import Path
from typing import Any

# bpy is available when running inside Blender (for scene fps / frame range)
try:
    import bpy  # type: ignore[import-untyped]
except ImportError:
    bpy = None  # type: ignore[assignment]

# ── Constants ───────────────────────────────────────────────────────────────

OUTPUT_SAMPLE_RATE = 48000  # match video audio standard


# ── Material presets ────────────────────────────────────────────────────────
# Each material = transient + body + tail layers per the research report.
# Designed for cozy/oddly-satisfying feel: near-field, controlled brightness.

MATERIAL_PRESETS: dict[str, dict[str, Any]] = {
    "wood": {
        # Cozy workshop wood: crisp but not harsh, "desk toy" intimacy
        "transient": {
            "freq_hz": 3200,        # bright click
            "duration_ms": 8,       # very short
            "amplitude": 0.6,
            "decay_rate": 200.0,    # instant
        },
        "body": {
            "freq_hz": 800,         # warm knock
            "freq2_hz": 1600,       # 2nd harmonic
            "freq2_mix": 0.3,
            "duration_ms": 55,
            "amplitude": 0.5,
            "decay_rate": 65.0,
        },
        "tail": {
            "freq_hz": 400,         # board resonance
            "duration_ms": 90,
            "amplitude": 0.12,
            "decay_rate": 25.0,
        },
        # Anti-machine-gun jitter ranges
        "gain_jitter_db": (-2.5, 1.5),
        "pitch_jitter_semitones": (-2.0, 2.0),
        "start_offset_ms": (0, 4),      # ms of sample start jitter
        "eq_tilt_db": (-1.5, 1.5),      # subtle brightness variation
        "num_variations": 24,            # pool size
    },
    "plastic": {
        # Clean pastel plastic: smoother transient, pleasing low-mid "thock"
        "transient": {
            "freq_hz": 2800,        # softer click than wood
            "duration_ms": 6,
            "amplitude": 0.45,
            "decay_rate": 220.0,
        },
        "body": {
            "freq_hz": 600,         # creamy thock
            "freq2_hz": 1200,
            "freq2_mix": 0.2,
            "duration_ms": 50,
            "amplitude": 0.55,
            "decay_rate": 70.0,
        },
        "tail": {
            "freq_hz": 300,         # subtle resonance
            "duration_ms": 70,
            "amplitude": 0.08,
            "decay_rate": 30.0,
        },
        "gain_jitter_db": (-2.0, 1.0),
        "pitch_jitter_semitones": (-1.5, 1.5),  # tighter than wood
        "start_offset_ms": (0, 3),
        "eq_tilt_db": (-1.0, 1.0),
        "num_variations": 24,
    },
    "ceramic": {
        # Zen stone/ceramic: bright clack, longer ring, more "detail"
        "transient": {
            "freq_hz": 4000,        # bright clack
            "duration_ms": 6,
            "amplitude": 0.55,
            "decay_rate": 180.0,
        },
        "body": {
            "freq_hz": 1200,        # clean mid
            "freq2_hz": 2400,
            "freq2_mix": 0.35,
            "duration_ms": 60,
            "amplitude": 0.45,
            "decay_rate": 50.0,
        },
        "tail": {
            "freq_hz": 600,         # ceramic ring
            "duration_ms": 140,     # longer tail = more "detail"
            "amplitude": 0.18,
            "decay_rate": 18.0,
        },
        "gain_jitter_db": (-2.0, 2.0),
        "pitch_jitter_semitones": (-2.5, 2.5),
        "start_offset_ms": (0, 5),
        "eq_tilt_db": (-2.0, 2.0),
        "num_variations": 24,
    },
    "metal": {
        # Neon/sci-fi: resonant ping, metallic ring
        "transient": {
            "freq_hz": 5000,
            "duration_ms": 5,
            "amplitude": 0.5,
            "decay_rate": 250.0,
        },
        "body": {
            "freq_hz": 1800,
            "freq2_hz": 3600,
            "freq2_mix": 0.4,
            "duration_ms": 70,
            "amplitude": 0.4,
            "decay_rate": 40.0,
        },
        "tail": {
            "freq_hz": 900,         # metallic ring
            "duration_ms": 180,
            "amplitude": 0.2,
            "decay_rate": 12.0,
        },
        "gain_jitter_db": (-3.0, 2.0),
        "pitch_jitter_semitones": (-3.0, 3.0),
        "start_offset_ms": (0, 3),
        "eq_tilt_db": (-2.0, 2.0),
        "num_variations": 24,
    },
    "heavy": {
        # Lava world: deep thud, bass impact
        "transient": {
            "freq_hz": 2000,
            "duration_ms": 10,
            "amplitude": 0.55,
            "decay_rate": 150.0,
        },
        "body": {
            "freq_hz": 300,
            "freq2_hz": 600,
            "freq2_mix": 0.5,
            "duration_ms": 80,
            "amplitude": 0.6,
            "decay_rate": 35.0,
        },
        "tail": {
            "freq_hz": 150,
            "duration_ms": 120,
            "amplitude": 0.15,
            "decay_rate": 15.0,
        },
        "gain_jitter_db": (-2.0, 3.0),
        "pitch_jitter_semitones": (-4.0, 1.0),
        "start_offset_ms": (0, 6),
        "eq_tilt_db": (-2.0, 1.5),
        "num_variations": 24,
    },
}

# Map theme names → material key
THEME_MATERIAL_MAP: dict[str, str] = {
    "deep_space": "metal",
    "enchanted_forest": "wood",
    "golden_hour": "ceramic",
    "arctic_lab": "plastic",
    "neon_city": "metal",
    "candy_land": "plastic",
    "lava_world": "heavy",
    "default": "wood",
}


# ── Layered synthesis ───────────────────────────────────────────────────────

def _synthesize_layer(
    freq_hz: float,
    duration_ms: int,
    amplitude: float,
    decay_rate: float,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    freq2_hz: float = 0.0,
    freq2_mix: float = 0.0,
    noise_mix: float = 0.0,
    rng: random.Random | None = None,
) -> list[float]:
    """Synthesize one layer (transient, body, or tail).

    Uses sine harmonics + tiny noise with shaped envelope.
    """
    if rng is None:
        rng = random.Random()

    n_samples = int(sample_rate * duration_ms / 1000)
    attack_samples = max(1, int(sample_rate * 0.001))  # 1ms attack
    samples: list[float] = []

    for i in range(n_samples):
        t = i / sample_rate

        # Envelope: instant attack → exponential decay
        if i < attack_samples:
            env = i / attack_samples
        else:
            env = math.exp(-decay_rate * t)

        # Primary sine
        sig = math.sin(2 * math.pi * freq_hz * t)

        # Optional 2nd harmonic
        if freq2_hz > 0 and freq2_mix > 0:
            sig = sig * (1.0 - freq2_mix) + math.sin(2 * math.pi * freq2_hz * t) * freq2_mix

        # Tiny noise for realism (ASMR-adjacent texture)
        sig += rng.uniform(-1, 1) * noise_mix

        samples.append(sig * env * amplitude)

    return samples


def _apply_eq_tilt(samples: list[float], tilt_db: float, sample_rate: int = OUTPUT_SAMPLE_RATE) -> list[float]:
    """Apply a simple high-shelf tilt using a 1-pole filter.

    Positive tilt_db = brighter; negative = darker.
    This gives subtle per-hit timbral variation without heavy DSP.
    """
    if abs(tilt_db) < 0.1:
        return samples

    # Simple 1-pole: y[n] = x[n] + coeff * (x[n] - y[n-1])
    coeff = tilt_db * 0.02  # very gentle
    coeff = max(-0.15, min(0.15, coeff))

    out = []
    prev = 0.0
    for s in samples:
        filtered = s + coeff * (s - prev)
        prev = filtered
        out.append(filtered)
    return out


def _generate_collision_sample(
    material: dict[str, Any],
    rng: random.Random,
    pitch_shift_semitones: float = 0.0,
    gain_db: float = 0.0,
    eq_tilt_db: float = 0.0,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
) -> list[float]:
    """Generate a single collision sample with 3-layer synthesis.

    Layers: transient (click) + body (thock/knock) + tail (resonance).
    """
    pitch_ratio = 2.0 ** (pitch_shift_semitones / 12.0)
    gain_linear = 10.0 ** (gain_db / 20.0)

    # Determine total length from longest layer
    max_ms = max(
        material["transient"]["duration_ms"],
        material["body"]["duration_ms"],
        material["tail"]["duration_ms"],
    )
    total_samples = int(sample_rate * max_ms / 1000)

    # Synthesize each layer
    trans_cfg = material["transient"]
    transient = _synthesize_layer(
        freq_hz=trans_cfg["freq_hz"] * pitch_ratio,
        duration_ms=trans_cfg["duration_ms"],
        amplitude=trans_cfg["amplitude"],
        decay_rate=trans_cfg["decay_rate"],
        sample_rate=sample_rate,
        rng=rng,
    )

    body_cfg = material["body"]
    body = _synthesize_layer(
        freq_hz=body_cfg["freq_hz"] * pitch_ratio,
        duration_ms=body_cfg["duration_ms"],
        amplitude=body_cfg["amplitude"],
        decay_rate=body_cfg["decay_rate"],
        sample_rate=sample_rate,
        freq2_hz=body_cfg.get("freq2_hz", 0) * pitch_ratio,
        freq2_mix=body_cfg.get("freq2_mix", 0),
        rng=rng,
    )

    tail_cfg = material["tail"]
    tail = _synthesize_layer(
        freq_hz=tail_cfg["freq_hz"] * pitch_ratio,
        duration_ms=tail_cfg["duration_ms"],
        amplitude=tail_cfg["amplitude"],
        decay_rate=tail_cfg["decay_rate"],
        sample_rate=sample_rate,
        rng=rng,
    )

    # Mix layers together (all start at sample 0)
    mixed: list[float] = [0.0] * total_samples
    for i in range(total_samples):
        val = 0.0
        if i < len(transient):
            val += transient[i]
        if i < len(body):
            val += body[i]
        if i < len(tail):
            val += tail[i]
        mixed[i] = val * gain_linear

    # Apply EQ tilt for timbral micro-variation
    mixed = _apply_eq_tilt(mixed, eq_tilt_db, sample_rate)

    return mixed


# ── Variation pool generation ───────────────────────────────────────────────

def generate_variation_pool(
    material_name: str,
    output_dir: str,
    num_variations: int | None = None,
    theme_name: str | None = None,
) -> list[str]:
    """Generate a pool of collision sound variations for a material.

    Implements the "never the same hit twice" strategy:
    - Each variation has unique pitch, gain, EQ tilt, and synthesis seed
    - Pool is large enough for shuffle-bag selection

    Args:
        material_name: Key into MATERIAL_PRESETS (wood, plastic, ceramic, metal, heavy).
        output_dir: Directory to write WAV files.
        num_variations: Number of variations to generate (default from preset).
        theme_name: Optional theme name to resolve material.

    Returns:
        List of WAV file paths.
    """
    # Resolve material
    if material_name not in MATERIAL_PRESETS:
        if theme_name:
            material_name = THEME_MATERIAL_MAP.get(theme_name, "wood")
        else:
            material_name = "wood"

    material = MATERIAL_PRESETS[material_name]
    if num_variations is None:
        num_variations = material.get("num_variations", 24)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pitch_range = material["pitch_jitter_semitones"]
    gain_range = material["gain_jitter_db"]
    eq_range = material["eq_tilt_db"]

    paths: list[str] = []
    for i in range(num_variations):
        rng = random.Random(42 + i)

        # Per-variation randomisation
        pitch = rng.uniform(*pitch_range)
        gain = rng.uniform(*gain_range)
        eq_tilt = rng.uniform(*eq_range)

        samples = _generate_collision_sample(
            material, rng,
            pitch_shift_semitones=pitch,
            gain_db=gain,
            eq_tilt_db=eq_tilt,
        )

        # Write WAV (mono, 16-bit, 48kHz)
        filepath = str(out / f"collision_{material_name}_{i:02d}.wav")
        _write_wav_mono(filepath, samples)
        paths.append(filepath)

    print(f"[collision_audio] Generated {len(paths)} {material_name} variations in {output_dir}")
    return paths


def _write_wav_mono(filepath: str, samples: list[float], sample_rate: int = OUTPUT_SAMPLE_RATE) -> None:
    """Write float samples to a mono 16-bit WAV."""
    peak = max((abs(s) for s in samples), default=1.0)
    if peak == 0:
        peak = 1.0
    # Normalize to ~-3dB headroom
    scale = 0.7 / peak

    int_samples = [max(-32767, min(32767, int(s * scale * 32767))) for s in samples]
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(int_samples)}h", *int_samples))


# ── Shuffle-bag selection ───────────────────────────────────────────────────

class ShuffleBag:
    """No-repeat sample selection: cycles through all items before repeating.

    Prevents the "machine-gun effect" of hearing the same sample twice
    in a row. After exhausting the bag, reshuffles for the next cycle.
    """

    def __init__(self, items: list[Any], rng: random.Random | None = None):
        self._items = list(items)
        self._rng = rng or random.Random()
        self._bag: list[Any] = []

    def next(self) -> Any:
        if not self._bag:
            self._bag = list(self._items)
            self._rng.shuffle(self._bag)
        return self._bag.pop()


# ── Energy mapping from tip gaps ────────────────────────────────────────────

def _compute_energy_map(
    tip_frames: dict[str, int],
    fps: int,
) -> dict[str, float]:
    """Compute per-domino energy proxy from inter-tip gap.

    Shorter gaps between successive tips → higher energy (faster cascade).
    Returns gain multiplier per domino (0.7 – 1.3 range).
    """
    sorted_items = sorted(tip_frames.items(), key=lambda x: x[1])
    if len(sorted_items) < 2:
        return {name: 1.0 for name, _ in sorted_items}

    # Compute gaps
    gaps: list[float] = []
    for i in range(1, len(sorted_items)):
        gap_frames = sorted_items[i][1] - sorted_items[i - 1][1]
        gap_sec = max(0.001, gap_frames / fps)
        gaps.append(gap_sec)

    # First domino has no predecessor — use median gap
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.5
    gaps.insert(0, median_gap)

    # Map gap to energy: shorter gap = more energy
    energy_map: dict[str, float] = {}
    for i, (name, _frame) in enumerate(sorted_items):
        gap = gaps[i]
        # Sigmoid-ish mapping: rapid gaps → 1.2-1.3, slow gaps → 0.7-0.8
        energy = 1.0 + 0.3 * (1.0 / (1.0 + math.exp(5.0 * (gap - 0.15))) - 0.5)
        energy = max(0.7, min(1.3, energy))
        energy_map[name] = energy

    return energy_map


# ── Direct Python audio mixing ──────────────────────────────────────────────

def _load_wav_mono(filepath: str) -> list[float]:
    """Load a mono WAV file and return float samples in [-1, 1]."""
    with wave.open(filepath, "r") as wf:
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        raw = wf.readframes(n_frames)
        n_ints = n_frames * n_channels
        int_samples = struct.unpack(f"<{n_ints}h", raw)
        # If stereo, take just the left channel
        if n_channels == 2:
            int_samples = int_samples[::2]
        return [s / 32767.0 for s in int_samples]


def _mix_collisions(
    tip_frames: dict[str, int],
    tap_buffers: list[list[float]],
    fps: int,
    end_frame: int,
    energy_map: dict[str, float],
    material: dict[str, Any],
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    volume: float = 0.85,
) -> tuple[array.array, float]:
    """Place collision sounds at tip-frame timestamps with full variation.

    Implements:
    - Shuffle-bag selection (no-repeat)
    - Per-hit gain jitter × energy mapping
    - Per-hit start-offset jitter (anti-machine-gun)

    Returns (interleaved_int16_stereo_array, peak_amplitude).
    """
    # Total mono samples (add 1 second safety margin)
    total_samples = int((end_frame / fps) * sample_rate) + sample_rate

    # Float mixing buffers
    mix_l = array.array("f", bytes(total_samples * 4))
    mix_r = array.array("f", bytes(total_samples * 4))

    # Shuffle-bag for no-repeat selection
    rng = random.Random(42)
    bag = ShuffleBag(list(range(len(tap_buffers))), rng)

    # Per-hit jitter ranges from material preset
    start_offset_range_ms = material.get("start_offset_ms", (0, 3))
    gain_jitter_range = material.get("gain_jitter_db", (-2.0, 1.5))

    placed = 0
    sorted_tips = sorted(tip_frames.items(), key=lambda x: x[1])

    for domino_name, tip_frame in sorted_tips:
        # Shuffle-bag selection (no consecutive repeats)
        tap_idx = bag.next()
        tap = tap_buffers[tap_idx]

        # Start-offset jitter (ms)
        offset_ms = rng.uniform(*start_offset_range_ms)
        offset_samples = int(offset_ms * sample_rate / 1000)

        # Energy-mapped gain
        energy = energy_map.get(domino_name, 1.0)
        hit_gain_db = rng.uniform(*gain_jitter_range)
        hit_gain = (10.0 ** (hit_gain_db / 20.0)) * energy * volume

        # Place in mix buffer
        sample_offset = int((tip_frame / fps) * sample_rate) + offset_samples
        for i, s in enumerate(tap):
            idx = sample_offset + i
            if 0 <= idx < total_samples:
                val = s * hit_gain
                mix_l[idx] += val
                mix_r[idx] += val

        placed += 1

    # Peak detection
    peak = 0.0
    for i in range(total_samples):
        peak = max(peak, abs(mix_l[i]), abs(mix_r[i]))

    # Soft-limit if clipping
    if peak > 0.95:
        scale = 0.95 / peak
        for i in range(total_samples):
            mix_l[i] *= scale
            mix_r[i] *= scale
        peak = 0.95

    # Convert to interleaved stereo int16
    out = array.array("h", bytes(total_samples * 2 * 2))
    idx = 0
    for i in range(total_samples):
        out[idx] = max(-32767, min(32767, int(mix_l[i] * 32767)))
        out[idx + 1] = max(-32767, min(32767, int(mix_r[i] * 32767)))
        idx += 2

    print(f"[collision_audio] Placed {placed} hits, peak={peak:.4f}")
    return out, peak


def mix_collision_audio(
    tip_frames: dict[str, int],
    wav_paths: list[str],
    output_wav: str,
    fps: int = 30,
    end_frame: int | None = None,
    material_name: str = "wood",
    theme_name: str | None = None,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
) -> dict[str, Any]:
    """Mix collision sounds into a single stereo WAV — pure Python, no Blender audio.

    Args:
        tip_frames: Mapping of domino name → frame number when it tips.
        wav_paths: Paths to the generated collision WAV files.
        output_wav: Destination path for the mixed WAV.
        fps: Scene frames per second.
        end_frame: Last frame of the animation (for buffer sizing).
        material_name: Collision material (wood, plastic, ceramic, metal, heavy).
        theme_name: Optional theme name to resolve material.
        sample_rate: Output sample rate (default 48 kHz).

    Returns:
        Dict with status, wav_path, wav_size_bytes, peak_amplitude, speakers_placed.
    """
    if not tip_frames or not wav_paths:
        print("[collision_audio] No tip frames or WAV paths — skipping mix")
        return {"status": "skipped", "speakers_placed": 0}

    # Resolve material preset
    mat_key = material_name
    if mat_key not in MATERIAL_PRESETS:
        mat_key = THEME_MATERIAL_MAP.get(theme_name or "", "wood")
    material = MATERIAL_PRESETS.get(mat_key, MATERIAL_PRESETS["wood"])

    # Load all variations into memory
    tap_buffers: list[list[float]] = []
    for wp in wav_paths:
        tap_buffers.append(_load_wav_mono(wp))
    print(f"[collision_audio] Loaded {len(tap_buffers)} {mat_key} variations")

    # Compute energy map from tip gaps
    energy_map = _compute_energy_map(tip_frames, fps)
    print(f"[collision_audio] Energy map: min={min(energy_map.values()):.2f}, "
          f"max={max(energy_map.values()):.2f}")

    # Determine end frame
    if end_frame is None:
        end_frame = max(tip_frames.values()) + 30

    # Ensure output directory exists
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_wav):
        os.remove(output_wav)

    # Mix with full variation stack
    audio_data, peak = _mix_collisions(
        tip_frames, tap_buffers, fps, end_frame,
        energy_map, material, sample_rate,
    )

    # Write WAV
    with wave.open(output_wav, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    wav_size = os.path.getsize(output_wav)
    print(f"[collision_audio] Output: {output_wav} ({wav_size:,} bytes)")
    print(f"[collision_audio] Peak amplitude: {peak:.4f}")
    print(f"[collision_audio] Material: {mat_key}")

    return {
        "status": "ok",
        "wav_path": output_wav,
        "wav_size_bytes": wav_size,
        "peak_amplitude": round(peak, 4),
        "speakers_placed": len(tip_frames),
        "material": mat_key,
    }


# ── Public entry point ──────────────────────────────────────────────────────

def add_collision_audio(
    tip_frames_path: str,
    sfx_dir: str,
    output_wav: str,
    num_variations: int = 24,
    theme_name: str | None = None,
) -> dict[str, Any]:
    """Full pipeline: generate material-matched collision pool → mix → write WAV.

    Args:
        tip_frames_path: Path to tip_frames.json (domino_name → frame_number).
        sfx_dir: Directory to store generated collision WAVs.
        output_wav: Path for the final mixed stereo WAV.
        num_variations: Number of variations to generate.
        theme_name: Theme name to pick material preset.

    Returns:
        Dict with status, speakers_placed, wav_path, peak_amplitude, material.
    """
    # Load tip frames
    tip_frames: dict[str, int] = json.loads(Path(tip_frames_path).read_text())
    print(f"[collision_audio] Loaded {len(tip_frames)} tip frames")

    # Resolve material from theme
    material_name = THEME_MATERIAL_MAP.get(theme_name or "", "wood")

    # Try to read theme_name from config if not provided
    if theme_name is None:
        config_dir = Path(tip_frames_path).parent
        theme_config = config_dir / "theme_config.json"
        if theme_config.exists():
            try:
                tc = json.loads(theme_config.read_text())
                theme_name = tc.get("theme_name", "")
                material_name = THEME_MATERIAL_MAP.get(theme_name, "wood")
                print(f"[collision_audio] Theme from config: {theme_name} -> material: {material_name}")
            except Exception:
                pass

    print(f"[collision_audio] Material: {material_name}")

    # Generate variation pool (material-driven synthesis)
    wav_paths = generate_variation_pool(
        material_name, sfx_dir,
        num_variations=num_variations,
        theme_name=theme_name,
    )

    # Get FPS and end frame from Blender scene if available
    fps = 30
    end_frame = None
    if bpy is not None:
        try:
            scene = bpy.context.scene
            fps = scene.render.fps
            end_frame = scene.frame_end
            print(f"[collision_audio] Scene: {fps} fps, frame_end={end_frame}")
        except Exception:
            pass

    # Mix all collisions into a single stereo WAV
    result = mix_collision_audio(
        tip_frames, wav_paths, output_wav,
        fps=fps, end_frame=end_frame,
        material_name=material_name,
        theme_name=theme_name,
    )

    return result
