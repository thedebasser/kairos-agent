"""Kairos Agent — FFmpeg Compositor Service.

Builds and executes FFmpeg commands to assemble final video:
- Raw simulation video
- Music track at -18dB with fade-out
- Caption overlays (Inter Bold, white + black stroke, lower third)
- Channel watermark
- 9:16 portrait output, correct codec

All FFmpeg operations are async (run via asyncio subprocess).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from kairos.models.contracts import (
    Caption,
    CaptionSet,
    MusicTrackMetadata,
    SimulationStats,
)
from kairos.services.caption import (
    CAPTION_FADE_IN_SEC,
    CAPTION_FADE_OUT_SEC,
    CAPTION_FONT,
    CAPTION_FONT_SIZE,
    CAPTION_POSITION_Y_PCT,
    CAPTION_STROKE_COLOUR,
    CAPTION_STROKE_WIDTH,
    build_ffmpeg_caption_filter,
)

logger = logging.getLogger(__name__)

# Audio constants
MUSIC_VOLUME_DB = -18
MUSIC_FADE_OUT_SEC = 3.0

# Output constants
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
OUTPUT_FPS = 30
OUTPUT_CODEC = "libx264"
OUTPUT_AUDIO_CODEC = "aac"
OUTPUT_AUDIO_BITRATE = "192k"
OUTPUT_CRF = 23  # Quality factor (lower = better, 18-28 typical)
OUTPUT_PRESET = "medium"
OUTPUT_PIXEL_FORMAT = "yuv420p"


def build_caption_filters(captions: CaptionSet) -> list[str]:
    """Build FFmpeg drawtext filter strings for all captions.

    Args:
        captions: CaptionSet containing all captions.

    Returns:
        List of FFmpeg drawtext filter strings.
    """
    filters = []
    for caption in captions.captions:
        filters.append(
            build_ffmpeg_caption_filter(
                caption,
                video_width=OUTPUT_WIDTH,
                video_height=OUTPUT_HEIGHT,
            )
        )
    return filters


def build_watermark_filter(
    watermark_path: str,
    *,
    position: str = "bottom_right",
    opacity: float = 0.5,
    margin: int = 20,
    size: int = 60,
) -> str:
    """Build FFmpeg overlay filter for channel watermark.

    Args:
        watermark_path: Path to watermark image (PNG with transparency).
        position: Corner position ('top_left', 'top_right', 'bottom_left', 'bottom_right').
        opacity: Watermark opacity (0.0-1.0).
        margin: Pixel margin from edge.
        size: Target height for watermark in pixels.

    Returns:
        FFmpeg filter string for watermark overlay.
    """
    positions = {
        "top_left": f"x={margin}:y={margin}",
        "top_right": f"x=W-w-{margin}:y={margin}",
        "bottom_left": f"x={margin}:y=H-h-{margin}",
        "bottom_right": f"x=W-w-{margin}:y=H-h-{margin}",
    }
    pos = positions.get(position, positions["bottom_right"])
    return (
        f"[wm];movie={watermark_path},"
        f"scale=-1:{size},"
        f"format=rgba,"
        f"colorchannelmixer=aa={opacity}"
        f"[wmscaled];[wm][wmscaled]overlay={pos}"
    )


def build_audio_filter(
    duration_sec: float,
    *,
    volume_db: int = MUSIC_VOLUME_DB,
    fade_out_sec: float = MUSIC_FADE_OUT_SEC,
) -> str:
    """Build FFmpeg audio filter for music track.

    Applies volume reduction and fade-out at end.

    Args:
        duration_sec: Total video duration in seconds.
        volume_db: Volume in dB (negative = quieter).
        fade_out_sec: Duration of fade-out at end.

    Returns:
        FFmpeg audio filter string.
    """
    fade_start = max(0, duration_sec - fade_out_sec)
    return (
        f"volume={volume_db}dB,"
        f"afade=t=out:st={fade_start:.1f}:d={fade_out_sec:.1f}"
    )


def build_ffmpeg_command(
    raw_video_path: str,
    music_path: str,
    output_path: str,
    captions: CaptionSet,
    duration_sec: float,
    *,
    watermark_path: str | None = None,
) -> list[str]:
    """Build the complete FFmpeg command for video assembly.

    Assembly order:
    1. Input raw simulation video
    2. Input music track
    3. Scale to 9:16 portrait (1080x1920)
    4. Overlay captions (drawtext filters)
    5. Overlay watermark (if provided)
    6. Mix audio at -18dB with fade-out
    7. Encode to H.264 + AAC

    Args:
        raw_video_path: Path to raw simulation video.
        music_path: Path to music track file.
        output_path: Path for final output video.
        captions: CaptionSet to overlay.
        duration_sec: Target duration in seconds.
        watermark_path: Optional path to watermark PNG.

    Returns:
        FFmpeg command as list of strings.
    """
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", raw_video_path,
        "-i", music_path,
    ]

    # Build video filter chain
    video_filters: list[str] = []

    # Scale to output resolution (9:16 portrait)
    video_filters.append(
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    # Caption overlays
    caption_filters = build_caption_filters(captions)
    video_filters.extend(caption_filters)

    # Combine video filters
    vf = ",".join(video_filters)

    # Watermark overlay (if provided)
    if watermark_path:
        watermark_filter = build_watermark_filter(watermark_path)
        # The watermark filter needs a separate chain:
        # [0:v]scale+captions[wm];movie=...[wmscaled];[wm][wmscaled]overlay[vout]
        vf_with_wm = f"[0:v]{vf}{watermark_filter}[vout]"
    else:
        vf_with_wm = f"[0:v]{vf}[vout]"

    # Audio filter
    af = build_audio_filter(duration_sec)

    cmd.extend([
        "-filter_complex",
        f"{vf_with_wm};[1:a]{af}[aout]",
        "-map", "[vout]",
        "-map", "[aout]",
    ])

    # Output settings
    cmd.extend([
        "-c:v", OUTPUT_CODEC,
        "-preset", OUTPUT_PRESET,
        "-crf", str(OUTPUT_CRF),
        "-pix_fmt", OUTPUT_PIXEL_FORMAT,
        "-c:a", OUTPUT_AUDIO_CODEC,
        "-b:a", OUTPUT_AUDIO_BITRATE,
        "-r", str(OUTPUT_FPS),
        "-t", f"{duration_sec:.1f}",
        "-shortest",
        output_path,
    ])

    return cmd


async def run_ffmpeg(cmd: list[str], *, timeout_sec: int = 300) -> tuple[int, str, str]:
    """Run an FFmpeg command asynchronously.

    Args:
        cmd: FFmpeg command as list of strings.
        timeout_sec: Timeout in seconds.

    Returns:
        Tuple of (returncode, stdout, stderr).
    """
    logger.info("Running FFmpeg: %s", " ".join(cmd[:6]) + " ...")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        logger.error("FFmpeg timed out after %ds", timeout_sec)
        return -1, "", f"FFmpeg timed out after {timeout_sec}s"

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    if process.returncode != 0:
        logger.error("FFmpeg failed (rc=%d): %s", process.returncode, stderr[-500:])
    else:
        logger.info("FFmpeg completed successfully")

    return process.returncode or 0, stdout, stderr


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system PATH."""
    return shutil.which("ffmpeg") is not None
