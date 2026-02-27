"""Unit tests for publish queue service."""

import pytest

from kairos.services.publish_queue import (
    MAX_RETRY_ATTEMPTS,
    generate_platform_metadata,
    should_retry,
)

pytestmark = pytest.mark.unit


class TestShouldRetry:
    """Tests for retry logic."""

    def test_failed_with_attempts_remaining(self):
        assert should_retry("failed", 1) is True

    def test_failed_at_max_attempts(self):
        assert should_retry("failed", MAX_RETRY_ATTEMPTS) is False

    def test_published_no_retry(self):
        assert should_retry("published", 0) is False

    def test_queued_no_retry(self):
        assert should_retry("queued", 0) is False


class TestGeneratePlatformMetadata:
    """Tests for platform-specific metadata generation."""

    def test_youtube_metadata(self):
        meta = generate_platform_metadata(
            base_title="Test Title",
            platform="youtube_shorts",
            category="ball_pit",
        )
        assert meta["title"] == "Test Title"
        assert "#ballpit" in meta["description"]

    def test_tiktok_has_hashtags_in_title(self):
        meta = generate_platform_metadata(
            base_title="Test Title",
            platform="tiktok",
            category="ball_pit",
        )
        assert "#" in meta["title"]

    def test_unknown_platform_fallback(self):
        meta = generate_platform_metadata(
            base_title="Test",
            platform="unknown_platform",
            category="test",
        )
        assert meta["title"] == "Test"
