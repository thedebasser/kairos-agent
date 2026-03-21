"""Mix collision sound effects into a video at frame-accurate timestamps.

Reads an SFX manifest (frame → sfx_wav_path) produced by the environment
pipeline, and uses FFmpeg to overlay each collision sound at the correct
time. Optionally mixes with an ambient background track.

Run after bake_and_render.py has produced the rendered video:

    python mix_audio.py \
        --video renders/render.mp4 \
        --manifest sfx_manifest.json \
        --output renders/render_with_sfx.mp4 \
        [--ambient ambient_loop.mp3]
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from kairos.config import get_settings
from kairos.services.async_subprocess import run_async

logger = logging.getLogger(__name__)


def _get_ffmpeg_path() -> str:
    """Get the resolved FFmpeg path from centralised config."""
    return get_settings().ffmpeg_path

# Audio mixing constants
SFX_VOLUME_DB = -22       # Collision SFX base volume (soft, calming)
AMBIENT_VOLUME_DB = -18   # Background ambient volume
MAX_CONCURRENT_SFX = 35   # FFmpeg adelay limit — fewer inputs = less noise floor
SFX_GROUP_SIZE = 50       # Process in batches for very long chains


def build_sfx_manifest(
    tip_frames: dict[str, int],
    sfx_paths: list[str],
    fps: int = 30,
) -> dict[int, str]:
    """Build an SFX manifest from physics tip_frames data.

    Args:
        tip_frames: Dict of domino_name → frame when it tipped.
        sfx_paths: List of varied SFX file paths to cycle through.
        fps: Video frame rate.

    Returns: Dict of frame_number → sfx_file_path.
    """
    if not sfx_paths:
        return {}

    manifest: dict[int, str] = {}
    sorted_tips = sorted(tip_frames.items(), key=lambda x: x[1])

    for i, (domino_name, frame) in enumerate(sorted_tips):
        sfx = sfx_paths[i % len(sfx_paths)]
        manifest[frame] = sfx

    return manifest


async def mix_sfx_into_video(
    video_path: str,
    manifest: dict[int, str] | dict[str, str],
    output_path: str,
    fps: int = 30,
    ambient_path: str | None = None,
    timeout_sec: int = 300,
) -> bool:
    """Mix collision SFX into a video using FFmpeg.

    Places each SFX at its frame-accurate timestamp using adelay filters.
    For large manifests, downsamples to MAX_CONCURRENT_SFX to avoid
    FFmpeg filter graph limits.

    Phase 4: converted from blocking ``subprocess.run`` to async.

    Args:
        video_path: Path to input video.
        manifest: Dict of frame_number → sfx_file_path.
        output_path: Path for output video with SFX.
        fps: Video frame rate.
        ambient_path: Optional ambient background audio loop.
        timeout_sec: FFmpeg timeout.

    Returns True on success.
    """
    if not manifest:
        logger.info("[mix_audio] Empty manifest — copying video as-is")
        shutil.copy2(video_path, output_path)
        return True

    # Normalise keys to int
    norm_manifest: dict[int, str] = {}
    for k, v in manifest.items():
        norm_manifest[int(k)] = str(v)

    # Sort by frame
    sorted_entries = sorted(norm_manifest.items())

    # Downsample if too many collision sounds
    if len(sorted_entries) > MAX_CONCURRENT_SFX:
        step = len(sorted_entries) // MAX_CONCURRENT_SFX
        sorted_entries = sorted_entries[::step][:MAX_CONCURRENT_SFX]
        logger.info("[mix_audio] Downsampled to %d SFX entries", len(sorted_entries))

    # Filter out missing files
    valid_entries = [
        (frame, sfx) for frame, sfx in sorted_entries
        if Path(sfx).exists()
    ]
    if not valid_entries:
        logger.warning("[mix_audio] No valid SFX files — copying video as-is")
        shutil.copy2(video_path, output_path)
        return True

    # Build FFmpeg command
    cmd = [_get_ffmpeg_path(), "-y", "-i", video_path]

    # Add SFX inputs
    for _, sfx_path in valid_entries:
        cmd.extend(["-i", sfx_path])

    # Add ambient input if provided
    ambient_idx = None
    if ambient_path and Path(ambient_path).exists():
        cmd.extend(["-stream_loop", "-1", "-i", ambient_path])
        ambient_idx = len(valid_entries) + 1

    # Build filter_complex
    fc_parts: list[str] = []

    # Apply adelay to each SFX
    for idx, (frame, _) in enumerate(valid_entries):
        input_idx = idx + 1  # 0 is the video
        delay_ms = int(frame / fps * 1000)
        fc_parts.append(
            f"[{input_idx}:a]adelay={delay_ms}|{delay_ms},"
            f"volume={SFX_VOLUME_DB}dB[sfx{idx}]"
        )

    # Mix all SFX streams together
    n_sfx = len(valid_entries)
    sfx_inputs = "".join(f"[sfx{i}]" for i in range(n_sfx))

    # Noise-gate after amix to kill static from many near-silent tracks
    gate_filter = "agate=threshold=0.002:attack=0.5:release=50,highpass=f=80"

    if ambient_idx:
        # Mix SFX together, then mix with ambient
        fc_parts.append(
            f"{sfx_inputs}amix=inputs={n_sfx}:duration=longest:normalize=0,"
            f"{gate_filter}[sfxmix]"
        )
        fc_parts.append(
            f"[{ambient_idx}:a]volume={AMBIENT_VOLUME_DB}dB[ambvol]"
        )
        fc_parts.append(
            "[sfxmix][ambvol]amix=inputs=2:duration=longest:normalize=0,apad[aout]"
        )
    else:
        fc_parts.append(
            f"{sfx_inputs}amix=inputs={n_sfx}:duration=longest:normalize=0,"
            f"{gate_filter},apad[aout]"
        )

    filter_complex = ";".join(fc_parts)

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",  # Don't re-encode video
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",  # Clip audio pad to video length
        output_path,
    ])

    logger.info("[mix_audio] Mixing %d SFX into video...", n_sfx)

    try:
        rc, _, stderr = await run_async(cmd, timeout=timeout_sec, text=True)
        if rc != 0:
            logger.error("[mix_audio] FFmpeg failed: %s", stderr[-500:])
            # Fall back to video without SFX
            shutil.copy2(video_path, output_path)
            return False

        logger.info("[mix_audio] SFX mixed successfully: %s", output_path)
        return True

    except (RuntimeError, TimeoutError):
        logger.error("[mix_audio] FFmpeg timed out after %ds", timeout_sec)
        shutil.copy2(video_path, output_path)
        return False
    except Exception as exc:
        logger.error("[mix_audio] Mix failed: %s", exc)
        shutil.copy2(video_path, output_path)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description="Mix collision SFX into video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--manifest", required=True, help="SFX manifest JSON path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--ambient", default=None, help="Ambient audio loop path")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ok = mix_sfx_into_video(
        args.video, manifest, args.output,
        fps=args.fps, ambient_path=args.ambient,
    )
    sys.exit(0 if ok else 1)
