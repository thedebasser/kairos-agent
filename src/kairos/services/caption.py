"""Kairos Agent — Caption Framework.

POC: Hook caption only (0-2s, ≤6 words).
Framework supports future expansion to rule, tension, payoff captions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kairos.models.contracts import Caption, CaptionSet, CaptionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FFmpeg drawtext text-escaping (Finding 7.3)
# ---------------------------------------------------------------------------

def ffmpeg_escape_text(text: str) -> str:
    """Escape *text* for use inside an FFmpeg drawtext ``text='...'`` value.

    FFmpeg drawtext requires escaping of:
        \\  →  \\\\
        '   →  '\\''   (end-quote, literal quote, re-open quote)
        :   →  \\:
        ;   →  \\;
        [   →  \\[
        ]   →  \\]
        =   →  \\=
        %   →  %%      (percent is the expansion char in drawtext)
        newlines are stripped (drawtext doesn't support literal newlines).

    Order matters: backslash must be escaped first.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "'\\''")
    text = text.replace(":", "\\:")
    text = text.replace(";", "\\;")
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace("=", "\\=")
    text = text.replace("%", "%%")
    # Strip newlines — drawtext can't render them directly
    text = text.replace("\n", " ").replace("\r", "")
    return text


def ffmpeg_escape_path(path: str) -> str:
    """Escape a file *path* for use inside an FFmpeg drawtext ``fontfile='...'``.

    Converts backslashes to forward slashes (Windows paths) and escapes
    colons (e.g. ``C:`` drive letters).
    """
    return path.replace("\\", "/").replace(":", "\\:")

# Font resolution — bundled Inter-Bold with fallback to Windows Arial Bold
_FONTS_DIR = Path(__file__).resolve().parent / "fonts"
_INTER_BOLD = _FONTS_DIR / "Inter-Bold.ttf"
_ARIAL_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")

def _resolve_font_path() -> str:
    """Resolve the best available font file path."""
    if _INTER_BOLD.exists():
        return str(_INTER_BOLD)
    if _ARIAL_BOLD.exists():
        return str(_ARIAL_BOLD)
    logger.warning(
        "No bundled font found at %s or %s — captions may not render",
        _INTER_BOLD,
        _ARIAL_BOLD,
    )
    return "Arial"  # Last resort — FFmpeg system font lookup


# Caption rendering constants (for FFmpeg compositor)
CAPTION_FONT = _resolve_font_path()
CAPTION_FONT_FALLBACK = str(_ARIAL_BOLD) if _ARIAL_BOLD.exists() else "Arial"
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
CAPTION_H_MARGIN_PX = 40  # Horizontal margin on each side


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
    caption_colour: str = "",
    caption_stroke_colour: str = "",
) -> str:
    """Generate FFmpeg drawtext filter string for a caption.

    Args:
        caption: The caption to render.
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        caption_colour: Override text colour (hex, e.g. "#FF69B4").
        caption_stroke_colour: Override stroke colour (hex).

    Returns:
        FFmpeg drawtext filter string.
    """
    text_colour = caption_colour or CAPTION_COLOUR
    stroke_colour = caption_stroke_colour or CAPTION_STROKE_COLOUR

    # Calculate font size that fits within video width with margins.
    # Average character width ≈ 0.55× font size for Inter Bold.
    max_text_width = video_width - 2 * CAPTION_H_MARGIN_PX
    char_count = max(len(caption.text), 1)
    avg_char_width_ratio = 0.55
    # Font size that would fit the full text in one line
    fit_font_size = int(max_text_width / (char_count * avg_char_width_ratio))
    # Clamp between a readable minimum and the original calculated size
    base_font_size = max(CAPTION_FONT_SIZE, int(video_height * 0.05))
    font_size = max(40, min(base_font_size, fit_font_size))
    y_pos = int(video_height * CAPTION_POSITION_Y_PCT)

    # Escape special characters for FFmpeg drawtext (Finding 7.3)
    escaped_text = ffmpeg_escape_text(caption.text)

    # Escape font path for FFmpeg (colons in Windows paths need escaping)
    escaped_font = ffmpeg_escape_path(CAPTION_FONT)

    return (
        f"drawtext=text='{escaped_text}'"
        f":fontfile='{escaped_font}'"
        f":fontsize={font_size}"
        f":fontcolor={text_colour}"
        f":borderw={CAPTION_STROKE_WIDTH}"
        f":bordercolor={stroke_colour}"
        f":shadowcolor=black@{CAPTION_SHADOW_OPACITY}"
        f":shadowx={CAPTION_SHADOW_OFFSET}"
        f":shadowy={CAPTION_SHADOW_OFFSET}"
        f":x=max({CAPTION_H_MARGIN_PX}\\,(w-text_w)/2)"
        f":y={y_pos}"
        f":enable='between(t,{caption.start_sec},{caption.end_sec})'"
        f":alpha='if(lt(t,{caption.start_sec + CAPTION_FADE_IN_SEC}),"
        f"(t-{caption.start_sec})/{CAPTION_FADE_IN_SEC},"
        f"if(gt(t,{caption.end_sec - CAPTION_FADE_OUT_SEC}),"
        f"({caption.end_sec}-t)/{CAPTION_FADE_OUT_SEC},1))'"
    )
