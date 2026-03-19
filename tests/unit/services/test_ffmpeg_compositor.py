"""Unit tests for FFmpeg compositor — build_ffmpeg_command audio mixing.

Finding 7.2: Four entirely different FFmpeg filter graphs are constructed
for the 2×2 matrix of (has_video_audio × has_tts). These tests verify the
generated command strings are syntactically valid and contain the expected
stream mappings without executing FFmpeg.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kairos.schemas.contracts import Caption, CaptionSet, CaptionType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_captions() -> CaptionSet:
    """Minimal caption set for testing."""
    return CaptionSet(
        captions=[
            Caption(
                text="Test caption",
                start_sec=0.0,
                end_sec=2.0,
                caption_type=CaptionType.HOOK,
            ),
        ]
    )


@pytest.fixture()
def base_kwargs(tmp_path, sample_captions: CaptionSet) -> dict:
    """Base keyword arguments for build_ffmpeg_command."""
    raw_video = tmp_path / "raw.mp4"
    raw_video.write_bytes(b"\x00")
    music = tmp_path / "music.mp3"
    music.write_bytes(b"\x00")
    output = tmp_path / "output.mp4"
    return {
        "raw_video_path": str(raw_video),
        "music_path": str(music),
        "output_path": str(output),
        "captions": sample_captions,
        "duration_sec": 65.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_filter_complex(cmd: list[str]) -> str:
    """Return the -filter_complex value from a command list."""
    for i, tok in enumerate(cmd):
        if tok == "-filter_complex":
            return cmd[i + 1]
    raise ValueError("-filter_complex not found in command")


def _has_output_maps(cmd: list[str]) -> bool:
    """Return True if cmd maps both [vout] and [aout]."""
    return "[vout]" in cmd and "[aout]" in cmd


# ---------------------------------------------------------------------------
# Branch 1: has_video_audio=True, has_tts=True
# ---------------------------------------------------------------------------

class TestAudioMixBranch_SFX_TTS:
    """Branch: video has SFX audio AND TTS is provided → 2-way amix (sfx+tts, no music)."""

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=True)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_two_way_amix(self, _mock_ffmpeg, _mock_probe, base_kwargs, tmp_path):
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        tts_file = tmp_path / "tts.mp3"
        tts_file.write_bytes(b"\x00")
        base_kwargs["tts_path"] = str(tts_file)

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)

        # Must reference two audio inputs (SFX + TTS) and mix them
        assert "amix=inputs=2" in fc
        assert "[sfx]" in fc
        assert "[music]" not in fc
        assert "[tts]" in fc
        assert "[aout]" in fc
        assert _has_output_maps(cmd)

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=True)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_sfx_stream_formatting(self, _mock_ffmpeg, _mock_probe, base_kwargs, tmp_path):
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        tts_file = tmp_path / "tts.mp3"
        tts_file.write_bytes(b"\x00")
        base_kwargs["tts_path"] = str(tts_file)

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)

        # SFX comes from stream 0:a
        assert "[0:a]" in fc
        # TTS from stream 1:a (music input removed)
        assert "[1:a]" in fc

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=True)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_normalize_off(self, _mock_ffmpeg, _mock_probe, base_kwargs, tmp_path):
        """amix should use normalize=0 to prevent auto-levelling."""
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        tts_file = tmp_path / "tts.mp3"
        tts_file.write_bytes(b"\x00")
        base_kwargs["tts_path"] = str(tts_file)

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)
        assert "normalize=0" in fc


# ---------------------------------------------------------------------------
# Branch 2: has_video_audio=False, has_tts=True
# ---------------------------------------------------------------------------

class TestAudioMixBranch_NoSFX_TTS:
    """Branch: no SFX, TTS provided → TTS-only stream (no amix)."""

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=False)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_tts_only(self, _mock_ffmpeg, _mock_probe, base_kwargs, tmp_path):
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        tts_file = tmp_path / "tts.mp3"
        tts_file.write_bytes(b"\x00")
        base_kwargs["tts_path"] = str(tts_file)

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)

        # Single TTS stream, no amix needed
        assert "amix" not in fc
        assert "[music]" not in fc
        assert "[1:a]" in fc
        assert "volume=2dB" in fc
        assert "[aout]" in fc
        # No SFX stream should be referenced
        assert "[sfx]" not in fc
        assert _has_output_maps(cmd)

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=False)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_no_0a_reference(self, _mock_ffmpeg, _mock_probe, base_kwargs, tmp_path):
        """Without SFX, stream 0:a should NOT appear."""
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        tts_file = tmp_path / "tts.mp3"
        tts_file.write_bytes(b"\x00")
        base_kwargs["tts_path"] = str(tts_file)

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)
        assert "[0:a]" not in fc


# ---------------------------------------------------------------------------
# Branch 3: has_video_audio=True, has_tts=False
# ---------------------------------------------------------------------------

class TestAudioMixBranch_SFX_NoTTS:
    """Branch: video has SFX but no TTS → SFX-only stream (no amix)."""

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=True)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_sfx_only(self, _mock_ffmpeg, _mock_probe, base_kwargs):
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)

        # Single SFX stream, no amix needed
        assert "amix" not in fc
        assert "[0:a]" in fc
        assert "volume=6dB" in fc
        assert "[music]" not in fc
        assert "[aout]" in fc
        # No TTS
        assert "[tts]" not in fc
        assert _has_output_maps(cmd)

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=True)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_sfx_padded(self, _mock_ffmpeg, _mock_probe, base_kwargs):
        """SFX track should be padded to full duration."""
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)
        assert "apad=whole_dur=65" in fc


# ---------------------------------------------------------------------------
# Branch 4: has_video_audio=False, has_tts=False
# ---------------------------------------------------------------------------

class TestAudioMixBranch_NoSFX_NoTTS:
    """Branch: no SFX, no TTS → anullsrc silence generator."""

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=False)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_silence_generator(self, _mock_ffmpeg, _mock_probe, base_kwargs):
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)

        # Silence via anullsrc, no amix
        assert "amix" not in fc
        assert "anullsrc" in fc
        assert "atrim=duration=65" in fc
        assert "[aout]" in fc
        assert _has_output_maps(cmd)

    @patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=False)
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_no_volume_no_fade(self, _mock_ffmpeg, _mock_probe, base_kwargs):
        """Silence generator should have no volume or fade processing."""
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        cmd = build_ffmpeg_command(**base_kwargs)
        fc = _extract_filter_complex(cmd)
        assert "volume" not in fc
        assert "afade" not in fc


# ---------------------------------------------------------------------------
# Cross-branch invariants
# ---------------------------------------------------------------------------

class TestAudioMixInvariants:
    """Properties that must hold across ALL branches."""

    @pytest.mark.parametrize(
        "has_audio,tts_exists",
        [(True, True), (True, False), (False, True), (False, False)],
        ids=["sfx+tts", "sfx_no_tts", "no_sfx+tts", "music_only"],
    )
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_output_codec_settings(self, _mock_ffmpeg, has_audio, tts_exists, base_kwargs, tmp_path):
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        with patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=has_audio):
            if tts_exists:
                tts_file = tmp_path / "tts.mp3"
                tts_file.write_bytes(b"\x00")
                base_kwargs["tts_path"] = str(tts_file)

            cmd = build_ffmpeg_command(**base_kwargs)

        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-pix_fmt" in cmd
        assert "yuv420p" in cmd

    @pytest.mark.parametrize(
        "has_audio,tts_exists",
        [(True, True), (True, False), (False, True), (False, False)],
        ids=["sfx+tts", "sfx_no_tts", "no_sfx+tts", "music_only"],
    )
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_maps_video_and_audio(self, _mock_ffmpeg, has_audio, tts_exists, base_kwargs, tmp_path):
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        with patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=has_audio):
            if tts_exists:
                tts_file = tmp_path / "tts.mp3"
                tts_file.write_bytes(b"\x00")
                base_kwargs["tts_path"] = str(tts_file)

            cmd = build_ffmpeg_command(**base_kwargs)

        assert _has_output_maps(cmd), f"Missing [vout]/[aout] maps in: {cmd}"

    @pytest.mark.parametrize(
        "has_audio,tts_exists",
        [(True, True), (True, False), (False, True), (False, False)],
        ids=["sfx+tts", "sfx_no_tts", "no_sfx+tts", "music_only"],
    )
    @patch("kairos.services.ffmpeg_compositor._get_ffmpeg_path", return_value="ffmpeg")
    def test_duration_flag(self, _mock_ffmpeg, has_audio, tts_exists, base_kwargs, tmp_path):
        """All branches should cap output to duration_sec."""
        from kairos.services.ffmpeg_compositor import build_ffmpeg_command

        with patch("kairos.services.ffmpeg_compositor._probe_has_audio", return_value=has_audio):
            if tts_exists:
                tts_file = tmp_path / "tts.mp3"
                tts_file.write_bytes(b"\x00")
                base_kwargs["tts_path"] = str(tts_file)

            cmd = build_ffmpeg_command(**base_kwargs)

        assert "-t" in cmd
        t_idx = cmd.index("-t")
        assert cmd[t_idx + 1] == "65.0"


# ---------------------------------------------------------------------------
# build_audio_filter
# ---------------------------------------------------------------------------

class TestBuildAudioFilter:
    """Tests for the build_audio_filter helper."""

    def test_default_volume(self):
        from kairos.services.ffmpeg_compositor import build_audio_filter
        result = build_audio_filter(65.0)
        assert "volume=-18dB" in result

    def test_custom_volume(self):
        from kairos.services.ffmpeg_compositor import build_audio_filter
        result = build_audio_filter(65.0, volume_db=-12)
        assert "volume=-12dB" in result

    def test_fade_out_timing(self):
        from kairos.services.ffmpeg_compositor import build_audio_filter
        result = build_audio_filter(65.0, fade_out_sec=3.0)
        assert "afade=t=out:st=62.0:d=3.0" in result

    def test_short_duration(self):
        """Fade start should clamp to 0 if duration < fade_out."""
        from kairos.services.ffmpeg_compositor import build_audio_filter
        result = build_audio_filter(2.0, fade_out_sec=5.0)
        assert "st=0.0" in result


# ---------------------------------------------------------------------------
# Caption text escaping (Finding 7.3)
# ---------------------------------------------------------------------------

class TestFFmpegEscapeText:
    """Tests for the comprehensive drawtext escaping utility."""

    def test_backslash_escaped(self):
        from kairos.services.caption import ffmpeg_escape_text
        assert "\\\\" in ffmpeg_escape_text("back\\slash")

    def test_colon_escaped(self):
        from kairos.services.caption import ffmpeg_escape_text
        assert "\\:" in ffmpeg_escape_text("time: 3pm")

    def test_semicolon_escaped(self):
        from kairos.services.caption import ffmpeg_escape_text
        assert "\\;" in ffmpeg_escape_text("a;b")

    def test_brackets_escaped(self):
        from kairos.services.caption import ffmpeg_escape_text
        result = ffmpeg_escape_text("[test]")
        assert "\\[" in result
        assert "\\]" in result

    def test_equals_escaped(self):
        from kairos.services.caption import ffmpeg_escape_text
        assert "\\=" in ffmpeg_escape_text("a=b")

    def test_percent_escaped(self):
        from kairos.services.caption import ffmpeg_escape_text
        assert "%%" in ffmpeg_escape_text("100%")

    def test_single_quote_escaped(self):
        from kairos.services.caption import ffmpeg_escape_text
        assert "'\\''" in ffmpeg_escape_text("it's")

    def test_newlines_stripped(self):
        from kairos.services.caption import ffmpeg_escape_text
        result = ffmpeg_escape_text("line1\nline2\r\nline3")
        assert "\n" not in result
        assert "\r" not in result

    def test_plain_text_unchanged(self):
        from kairos.services.caption import ffmpeg_escape_text
        text = "Hello World"
        assert ffmpeg_escape_text(text) == text

    def test_combined_special_chars(self):
        from kairos.services.caption import ffmpeg_escape_text
        result = ffmpeg_escape_text("100% of [users]: it's a=b;c")
        # Should contain all escapes without crashing
        assert "%%" in result
        assert "\\[" in result
        assert "\\:" in result
        assert "\\;" in result
        assert "\\=" in result
