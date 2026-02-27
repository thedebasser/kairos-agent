"""Tests for kairos.services.uploader — Upload & Publishing Service."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kairos.services.uploader import (
    TIKTOK_MIN_DURATION_SEC,
    PublishAttemptResult,
    TikTokUploader,
    UploadResult,
    YouTubeShortsUploader,
    calculate_padding_duration,
    get_all_uploaders,
    get_uploader,
    needs_duration_padding,
    process_publish_queue,
    publish_video,
    register_uploader,
)


# ── Duration Padding ─────────────────────────────────────────────────

class TestDurationPadding:
    def test_tiktok_needs_padding_short_video(self):
        assert needs_duration_padding(55.0, "tiktok") is True

    def test_tiktok_no_padding_long_video(self):
        assert needs_duration_padding(65.0, "tiktok") is False

    def test_tiktok_exact_boundary(self):
        assert needs_duration_padding(62.0, "tiktok") is False

    def test_youtube_never_needs_padding(self):
        assert needs_duration_padding(30.0, "youtube_shorts") is False

    def test_calculate_padding_tiktok(self):
        assert calculate_padding_duration(50.0, "tiktok") == 12.0

    def test_calculate_padding_no_deficit(self):
        assert calculate_padding_duration(70.0, "tiktok") == 0.0

    def test_calculate_padding_other_platform(self):
        assert calculate_padding_duration(30.0, "youtube_shorts") == 0.0


# ── Platform Uploaders ───────────────────────────────────────────────

class TestYouTubeShortsUploader:
    @pytest.fixture()
    def uploader(self):
        return YouTubeShortsUploader()

    def test_platform_name(self, uploader):
        assert uploader.platform_name == "youtube_shorts"

    async def test_upload_file_not_found(self, uploader):
        result = await uploader.upload(
            video_path="/nonexistent/video.mp4",
            title="Test",
            description="Test desc",
        )
        assert result.success is False
        assert "not found" in result.error

    async def test_upload_success(self, uploader, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake_video_data")

        result = await uploader.upload(
            video_path=str(video),
            title="Test Video",
            description="Test description",
        )
        assert result.success is True
        assert result.platform_video_id.startswith("yt_")


class TestTikTokUploader:
    @pytest.fixture()
    def uploader(self):
        return TikTokUploader()

    def test_platform_name(self, uploader):
        assert uploader.platform_name == "tiktok"

    async def test_upload_file_not_found(self, uploader):
        result = await uploader.upload(
            video_path="/nonexistent/video.mp4",
            title="Test",
            description="Test desc",
        )
        assert result.success is False

    async def test_upload_success(self, uploader, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake_video_data")

        result = await uploader.upload(
            video_path=str(video),
            title="Test",
            description="Desc",
        )
        assert result.success is True
        assert result.platform_video_id.startswith("tt_")


# ── Uploader Registry ────────────────────────────────────────────────

class TestUploaderRegistry:
    def test_get_youtube_uploader(self):
        u = get_uploader("youtube_shorts")
        assert u is not None
        assert u.platform_name == "youtube_shorts"

    def test_get_tiktok_uploader(self):
        u = get_uploader("tiktok")
        assert u is not None
        assert u.platform_name == "tiktok"

    def test_get_unknown_returns_none(self):
        assert get_uploader("unknown_platform") is None

    def test_get_all_uploaders(self):
        all_u = get_all_uploaders()
        assert "youtube_shorts" in all_u
        assert "tiktok" in all_u

    def test_register_custom_uploader(self):
        from kairos.services.uploader import PlatformUploader

        class CustomUploader(PlatformUploader):
            @property
            def platform_name(self):
                return "custom_test"

            async def upload(self, **kwargs):
                return UploadResult(success=True, platform_video_id="custom_123")

        register_uploader(CustomUploader())
        assert get_uploader("custom_test") is not None


# ── publish_video with retries ────────────────────────────────────────

class TestPublishVideo:
    async def test_publish_success_first_attempt(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")

        result = await publish_video(
            queue_id=uuid.uuid4(),
            output_id=uuid.uuid4(),
            platform="youtube_shorts",
            video_path=str(video),
            title="Test",
            description="Desc",
        )
        assert result.success is True
        assert result.attempts == 1
        assert result.platform_video_id is not None

    async def test_publish_unknown_platform(self):
        result = await publish_video(
            queue_id=uuid.uuid4(),
            output_id=uuid.uuid4(),
            platform="unknown_platform",
            video_path="/fake/path.mp4",
            title="test",
            description="desc",
        )
        assert result.success is False
        assert "No uploader" in result.error

    async def test_publish_retry_on_failure(self):
        """Upload fails with file-not-found, retries, all fail."""
        result = await publish_video(
            queue_id=uuid.uuid4(),
            output_id=uuid.uuid4(),
            platform="youtube_shorts",
            video_path="/nonexistent/file.mp4",
            title="Test",
            description="Desc",
            max_attempts=2,
        )
        assert result.success is False
        assert result.attempts == 2

    async def test_publish_sends_alert_on_exhaust(self):
        """When all attempts exhausted, Slack alert is sent."""
        with patch("kairos.services.uploader._send_publish_failure_alert") as mock_alert:
            mock_alert.return_value = None
            result = await publish_video(
                queue_id=uuid.uuid4(),
                output_id=uuid.uuid4(),
                platform="youtube_shorts",
                video_path="/nonexistent/file.mp4",
                title="Test",
                description="Desc",
                max_attempts=1,
            )
            assert result.success is False
            mock_alert.assert_called_once()


# ── process_publish_queue ─────────────────────────────────────────────

class TestProcessPublishQueue:
    async def test_process_batch(self, tmp_path):
        v1 = tmp_path / "v1.mp4"
        v2 = tmp_path / "v2.mp4"
        v1.write_bytes(b"video1")
        v2.write_bytes(b"video2")

        items = [
            {
                "queue_id": uuid.uuid4(),
                "output_id": uuid.uuid4(),
                "platform": "youtube_shorts",
                "video_path": str(v1),
                "title": "Video 1",
                "description": "Desc 1",
            },
            {
                "queue_id": uuid.uuid4(),
                "output_id": uuid.uuid4(),
                "platform": "tiktok",
                "video_path": str(v2),
                "title": "Video 2",
                "description": "Desc 2",
            },
        ]

        results = await process_publish_queue(items)
        assert len(results) == 2
        assert all(r.success for r in results)

    async def test_process_empty_queue(self):
        results = await process_publish_queue([])
        assert results == []

    async def test_process_mixed_results(self, tmp_path):
        video = tmp_path / "v1.mp4"
        video.write_bytes(b"data")

        items = [
            {
                "queue_id": uuid.uuid4(),
                "output_id": uuid.uuid4(),
                "platform": "youtube_shorts",
                "video_path": str(video),
                "title": "Good",
                "description": "",
            },
            {
                "queue_id": uuid.uuid4(),
                "output_id": uuid.uuid4(),
                "platform": "youtube_shorts",
                "video_path": "/missing/file.mp4",
                "title": "Bad",
                "description": "",
            },
        ]

        results = await process_publish_queue(items)
        assert results[0].success is True
        assert results[1].success is False


# ── Upload result ─────────────────────────────────────────────────────

class TestUploadResult:
    def test_success_result(self):
        r = UploadResult(success=True, platform_video_id="abc123")
        assert r.success is True
        assert r.platform_video_id == "abc123"

    def test_failure_result(self):
        r = UploadResult(success=False, error="API error")
        assert r.success is False
        assert r.error == "API error"

    def test_default_metadata(self):
        r = UploadResult(success=True)
        assert r.metadata == {}


# ── PublishAttemptResult ──────────────────────────────────────────────

class TestPublishAttemptResult:
    def test_fields(self):
        qid = uuid.uuid4()
        r = PublishAttemptResult(
            queue_id=qid,
            platform="tiktok",
            success=True,
            platform_video_id="tt_abc",
            attempts=1,
        )
        assert r.queue_id == qid
        assert r.platform == "tiktok"
        assert r.success is True
        assert r.attempts == 1


# ── Slack alert ───────────────────────────────────────────────────────

class TestSlackAlert:
    async def test_alert_no_webhook(self):
        """No error when webhook URL is not configured."""
        from kairos.services.uploader import _send_publish_failure_alert

        with patch("kairos.services.uploader.get_settings") as mock_settings:
            mock_settings.return_value.slack_webhook_url = ""
            # Should not raise
            await _send_publish_failure_alert(
                platform="tiktok",
                queue_id=uuid.uuid4(),
                attempts=3,
                error="API timeout",
            )
