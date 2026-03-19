"""Unit tests for caption framework.

Tests caption validation, creation, and FFmpeg filter generation.
"""

import pytest

from kairos.schemas.contracts import CaptionType
from kairos.services.caption import (
    CAPTION_MAX_WORDS,
    build_caption_set,
    build_ffmpeg_caption_filter,
    create_hook_caption,
    validate_caption_text,
)

pytestmark = pytest.mark.unit


class TestValidateCaptionText:
    """Tests for caption text validation."""

    def test_valid_short_text(self):
        assert validate_caption_text("What happens next?") is True

    def test_valid_six_words(self):
        assert validate_caption_text("one two three four five six") is True

    def test_too_many_words(self):
        assert validate_caption_text("one two three four five six seven") is False

    def test_empty_text(self):
        assert validate_caption_text("") is False

    def test_whitespace_only(self):
        assert validate_caption_text("   ") is False

    def test_max_chars(self):
        # 50 chars is the limit; 1 word of 50 chars is still ≤6 words → valid
        text = "A" * 50
        assert validate_caption_text(text) is True
        assert validate_caption_text("Short") is True

    def test_exactly_50_chars_one_word(self):
        # 50 chars, 1 word — valid length-wise
        assert validate_caption_text("A" * 50) is True

    def test_over_50_chars(self):
        assert validate_caption_text("A" * 51) is False


class TestCreateHookCaption:
    """Tests for hook caption creation."""

    def test_valid_hook(self):
        caption = create_hook_caption("What happens next?")
        assert caption.caption_type == CaptionType.HOOK
        assert caption.start_sec == 0.0
        assert caption.end_sec == 2.5

    def test_custom_timing(self):
        caption = create_hook_caption("Watch this!", start_sec=0.5, duration_sec=2.0)
        assert caption.start_sec == 0.5
        assert caption.end_sec == 2.5

    def test_invalid_text_raises(self):
        with pytest.raises(ValueError, match="≤6 words"):
            create_hook_caption("this has way too many words for a hook caption")

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            create_hook_caption("")


class TestBuildCaptionSet:
    """Tests for building a complete caption set."""

    def test_hook_only_set(self):
        cs = build_caption_set("What happens next?")
        assert len(cs.captions) == 1
        assert cs.hook is not None
        assert cs.hook.caption_type == CaptionType.HOOK

    def test_invalid_hook_raises(self):
        with pytest.raises(ValueError):
            build_caption_set("way too many words in this hook caption text")


class TestFFmpegCaptionFilter:
    """Tests for FFmpeg drawtext filter generation."""

    def test_filter_contains_text(self):
        caption = create_hook_caption("Watch this!")
        filter_str = build_ffmpeg_caption_filter(caption)
        assert "Watch this!" in filter_str

    def test_filter_contains_timing(self):
        caption = create_hook_caption("Test", start_sec=0.0, duration_sec=2.0)
        filter_str = build_ffmpeg_caption_filter(caption)
        assert "between(t,0.0,2.0)" in filter_str

    def test_filter_contains_font_settings(self):
        caption = create_hook_caption("Test")
        filter_str = build_ffmpeg_caption_filter(caption)
        assert "fontsize=" in filter_str
        assert "fontcolor=#FFFFFF" in filter_str
        assert "borderw=3" in filter_str
