"""Kairos Agent — Caption Framework.

POC: Hook caption only (0-2s, ≤6 words).
Framework supports future expansion to rule, tension, payoff captions.
"""

from __future__ import annotations

import logging

from kairos.models.contracts import Caption, CaptionSet, CaptionType

logger = logging.getLogger(__name__)

# Caption rendering constants (for FFmpeg compositor)
CAPTION_FONT = "Inter-Bold"
CAPTION_FONT_FALLBACK = "Montserrat-Bold"
CAPTION_FONT_SIZE = 72  # Minimum, scaled to ~5% of frame height
CAPTION_COLOUR = "#FFFFFF"
CAPTION_STROKE_COLOUR = "#000000"
CAPTION_STROKE_WIDTH = 3
CAPTION_SHADOW_OFFSET = 2
CAPTION_SHADOW_OPACITY = 0.5
CAPTION_POSITION_Y_PCT = 0.75  # Lower third
CAPTION_FADE_IN_SEC = 0.3
CAPTION_FADE_OUT_SEC = 0.3
CAPTION_MAX_WORDS = 6
CAPTION_MAX_DURATION_SEC = 3.0


def validate_caption_text(text: str) -> bool:
    """Validate that caption text meets constraints.

    - ≤6 words
    - Not empty
    - ≤50 characters
    """
    if not text or not text.strip():
        return False
    words = text.strip().split()
    return len(words) <= CAPTION_MAX_WORDS and len(text) <= 50


def create_hook_caption(
    text: str,
    *,
    start_sec: float = 0.0,
    duration_sec: float = 2.5,
) -> Caption:
    """Create a hook caption (shown at video start).

    Args:
        text: Hook text (≤6 words).
        start_sec: When to show (default 0s).
        duration_sec: How long to show (default 2.5s).

    Returns:
        Caption object.

    Raises:
        ValueError: If text exceeds constraints.
    """
    if not validate_caption_text(text):
        msg = f"Hook caption must be ≤6 words and ≤50 chars, got: '{text}'"
        raise ValueError(msg)

    return Caption(
        caption_type=CaptionType.HOOK,
        text=text.strip(),
        start_sec=start_sec,
        end_sec=start_sec + duration_sec,
    )


def build_caption_set(hook_text: str) -> CaptionSet:
    """Build a CaptionSet with hook caption only (POC).

    Args:
        hook_text: The hook caption text.

    Returns:
        CaptionSet containing the hook caption.
    """
    hook = create_hook_caption(hook_text)
    return CaptionSet(captions=[hook])


def build_ffmpeg_caption_filter(
    caption: Caption,
    *,
    video_width: int = 1080,
    video_height: int = 1920,
) -> str:
    """Generate FFmpeg drawtext filter string for a caption.

    Args:
        caption: The caption to render.
        video_width: Video width in pixels.
        video_height: Video height in pixels.

    Returns:
        FFmpeg drawtext filter string.
    """
    font_size = max(CAPTION_FONT_SIZE, int(video_height * 0.05))
    y_pos = int(video_height * CAPTION_POSITION_Y_PCT)

    # Escape special characters for FFmpeg
    escaped_text = caption.text.replace("'", "'\\''").replace(":", "\\:")

    return (
        f"drawtext=text='{escaped_text}'"
        f":fontfile={CAPTION_FONT}"
        f":fontsize={font_size}"
        f":fontcolor={CAPTION_COLOUR}"
        f":borderw={CAPTION_STROKE_WIDTH}"
        f":bordercolor={CAPTION_STROKE_COLOUR}"
        f":shadowcolor=black@{CAPTION_SHADOW_OPACITY}"
        f":shadowx={CAPTION_SHADOW_OFFSET}"
        f":shadowy={CAPTION_SHADOW_OFFSET}"
        f":x=(w-text_w)/2"
        f":y={y_pos}"
        f":enable='between(t,{caption.start_sec},{caption.end_sec})'"
        f":alpha='if(lt(t,{caption.start_sec + CAPTION_FADE_IN_SEC}),"
        f"(t-{caption.start_sec})/{CAPTION_FADE_IN_SEC},"
        f"if(gt(t,{caption.end_sec - CAPTION_FADE_OUT_SEC}),"
        f"({caption.end_sec}-t)/{CAPTION_FADE_OUT_SEC},1))'"
    )
