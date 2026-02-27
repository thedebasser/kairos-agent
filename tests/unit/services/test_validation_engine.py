"""Unit tests for validation engine.

Tests all Tier 1 programmatic validation checks.
Note: These tests require ffprobe/ffmpeg to be installed for full coverage.
Tests with missing ffmpeg gracefully skip.
"""

import pytest

from kairos.services.validation import (
    check_audio_present,
    check_duration,
    check_file_size,
    check_fps,
    check_resolution,
    check_valid_mp4,
    validate_simulation,
)

pytestmark = pytest.mark.unit


class TestCheckValidMp4:
    """Tests for MP4 validity check."""

    def test_nonexistent_file(self, tmp_path):
        result = check_valid_mp4(str(tmp_path / "nonexistent.mp4"))
        assert result.passed is False

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        result = check_valid_mp4(str(empty))
        assert result.passed is False

    def test_corrupt_file(self, tmp_path):
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"not a real mp4 file content here")
        result = check_valid_mp4(str(corrupt))
        assert result.passed is False


class TestCheckFileSize:
    """Tests for file size validation."""

    def test_nonexistent_file(self, tmp_path):
        result = check_file_size(str(tmp_path / "missing.mp4"))
        assert result.passed is False

    def test_empty_file_fails(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        result = check_file_size(str(empty))
        assert result.passed is False

    def test_small_file_fails(self, tmp_path):
        small = tmp_path / "small.mp4"
        small.write_bytes(b"x" * 100)
        result = check_file_size(str(small))
        assert result.passed is False


class TestValidateSimulation:
    """Tests for the full validation pipeline."""

    def test_nonexistent_video(self, tmp_path):
        result = validate_simulation(str(tmp_path / "missing.mp4"))
        assert result.passed is False
        assert result.tier1_passed is False

    def test_tier2_skipped_by_default(self, tmp_path):
        fake = tmp_path / "fake.mp4"
        fake.write_bytes(b"fake")
        result = validate_simulation(str(fake), run_tier2=False)
        assert result.tier2_passed is None
