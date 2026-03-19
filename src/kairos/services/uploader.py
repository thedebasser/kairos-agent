"""Kairos Agent — Upload & Publishing Service.

Handles uploading videos to platforms (YouTube Shorts, TikTok, etc.)
via pluggable uploader backends. Includes retry logic with exponential
backoff and Discord alerting on repeated failures.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from kairos.config import get_settings
from kairos.schemas.contracts import PublishStatus
from kairos.ai.tracing.sinks.langfuse_sink import record_metric, trace_pipeline_step
from kairos.services.publish_queue import (
    MAX_RETRY_ATTEMPTS,
    generate_platform_metadata,
    should_retry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Duration Padding
# =============================================================================

TIKTOK_MIN_DURATION_SEC = 62  # TikTok monetisation floor


def needs_duration_padding(duration_sec: float, platform: str) -> bool:
    """Check if a video needs padding for platform monetisation rules.

    TikTok requires ≥62s for mid-roll ad eligibility.
    """
    if platform == "tiktok" and duration_sec < TIKTOK_MIN_DURATION_SEC:
        return True
    return False


def calculate_padding_duration(duration_sec: float, platform: str) -> float:
    """Calculate how many seconds of padding are needed."""
    if platform == "tiktok":
        deficit = TIKTOK_MIN_DURATION_SEC - duration_sec
        return max(0.0, deficit)
    return 0.0


async def apply_duration_padding(
    video_path: str,
    padding_sec: float,
    output_path: str | None = None,
) -> str:
    """Pad a video to meet minimum duration via FFmpeg.

    Adds a freeze-frame (tpad) at the end of the video, followed by a
    fade-to-black over the padded section.

    Args:
        video_path: Path to the source video.
        padding_sec: Seconds of padding to add.
        output_path: Optional output path. Defaults to video_path with '_padded' suffix.

    Returns:
        Path to the padded video.
    """
    import subprocess

    if output_path is None:
        p = Path(video_path)
        output_path = str(p.with_stem(f"{p.stem}_padded"))

    pad_ms = int(padding_sec * 1000)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"tpad=stop_mode=clone:stop_duration={padding_sec:.1f},"
               f"fade=t=out:st={padding_sec - 1:.1f}:d=1",
        "-af", f"apad=pad_dur={padding_sec:.1f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    logger.info("Padding video by %.1fs: %s -> %s", padding_sec, video_path, output_path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error("FFmpeg padding failed: %s", result.stderr[-500:])
        return video_path  # Fallback to original
    return output_path


# =============================================================================
# Platform Uploader ABC + implementations
# =============================================================================


@dataclass
class UploadResult:
    """Result of a platform upload attempt."""

    success: bool
    platform_video_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PlatformUploader(ABC):
    """Abstract base class for platform upload adapters."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Platform identifier (e.g., 'youtube_shorts')."""

    @abstractmethod
    async def upload(
        self,
        *,
        video_path: str,
        title: str,
        description: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UploadResult:
        """Upload a video to the platform.

        Args:
            video_path: Path to the video file.
            title: Video title.
            description: Video description.
            tags: Optional list of tags.
            metadata: Additional platform-specific metadata.

        Returns:
            UploadResult with success/failure and platform_video_id.
        """


class YouTubeShortsUploader(PlatformUploader):
    """YouTube Shorts uploader via YouTube Data API v3.

    Requires YOUTUBE_API_KEY and YOUTUBE_REFRESH_TOKEN in environment.
    This is a scaffold — real OAuth2 flow and API calls are TODO.
    """

    @property
    def platform_name(self) -> str:
        return "youtube_shorts"

    async def upload(
        self,
        *,
        video_path: str,
        title: str,
        description: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UploadResult:
        """Upload a video to YouTube Shorts.

        TODO: Implement real YouTube Data API v3 upload:
        1. OAuth2 token refresh
        2. Resumable upload initiation
        3. Video chunk upload
        4. Set snippet (title, description, tags, categoryId=22)
        5. Set status (privacyStatus=public, selfDeclaredMadeForKids=false)
        6. Set #Shorts in title/description
        """
        logger.info("YouTube Shorts upload: %s — %s", title, video_path)

        if not Path(video_path).exists():
            return UploadResult(
                success=False,
                error=f"Video file not found: {video_path}",
            )

        # Placeholder: real implementation would call YouTube API
        platform_video_id = f"yt_{uuid.uuid4().hex[:12]}"
        logger.info("YouTube upload complete: %s", platform_video_id)

        return UploadResult(
            success=True,
            platform_video_id=platform_video_id,
            metadata={"platform": "youtube_shorts", "shorts": True},
        )


class TikTokUploader(PlatformUploader):
    """TikTok uploader via Upload-Post API.

    Requires TIKTOK_ACCESS_TOKEN in environment.
    This is a scaffold — real API calls are TODO.
    """

    @property
    def platform_name(self) -> str:
        return "tiktok"

    async def upload(
        self,
        *,
        video_path: str,
        title: str,
        description: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UploadResult:
        """Upload a video to TikTok.

        TODO: Implement real TikTok Upload-Post API:
        1. Initiate upload (POST /v2/post/publish/video/init/)
        2. Upload video file to upload_url
        3. Poll for processing status
        4. Set title/description/hashtags
        """
        logger.info("TikTok upload: %s — %s", title, video_path)

        if not Path(video_path).exists():
            return UploadResult(
                success=False,
                error=f"Video file not found: {video_path}",
            )

        platform_video_id = f"tt_{uuid.uuid4().hex[:12]}"
        logger.info("TikTok upload complete: %s", platform_video_id)

        return UploadResult(
            success=True,
            platform_video_id=platform_video_id,
            metadata={"platform": "tiktok"},
        )


class InstagramReelsUploader(PlatformUploader):
    """Instagram Reels uploader via Graph API.

    Requires business account and INSTAGRAM_ACCESS_TOKEN.
    Scaffold — real API calls are TODO.
    """

    @property
    def platform_name(self) -> str:
        return "instagram_reels"

    async def upload(
        self,
        *,
        video_path: str,
        title: str,
        description: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UploadResult:
        logger.info("Instagram Reels upload: %s — %s", title, video_path)
        if not Path(video_path).exists():
            return UploadResult(success=False, error=f"Video file not found: {video_path}")
        platform_video_id = f"ig_{uuid.uuid4().hex[:12]}"
        return UploadResult(
            success=True,
            platform_video_id=platform_video_id,
            metadata={"platform": "instagram_reels"},
        )


class FacebookReelsUploader(PlatformUploader):
    """Facebook Reels uploader via Graph API.

    Scaffold — real API calls are TODO.
    """

    @property
    def platform_name(self) -> str:
        return "facebook_reels"

    async def upload(
        self,
        *,
        video_path: str,
        title: str,
        description: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UploadResult:
        logger.info("Facebook Reels upload: %s — %s", title, video_path)
        if not Path(video_path).exists():
            return UploadResult(success=False, error=f"Video file not found: {video_path}")
        platform_video_id = f"fb_{uuid.uuid4().hex[:12]}"
        return UploadResult(
            success=True,
            platform_video_id=platform_video_id,
            metadata={"platform": "facebook_reels"},
        )


class SnapchatSpotlightUploader(PlatformUploader):
    """Snapchat Spotlight uploader.

    Scaffold — real API calls are TODO.
    """

    @property
    def platform_name(self) -> str:
        return "snapchat_spotlight"

    async def upload(
        self,
        *,
        video_path: str,
        title: str,
        description: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UploadResult:
        logger.info("Snapchat Spotlight upload: %s — %s", title, video_path)
        if not Path(video_path).exists():
            return UploadResult(success=False, error=f"Video file not found: {video_path}")
        platform_video_id = f"sc_{uuid.uuid4().hex[:12]}"
        return UploadResult(
            success=True,
            platform_video_id=platform_video_id,
            metadata={"platform": "snapchat_spotlight"},
        )


# Registry of platform uploaders
_uploaders: dict[str, PlatformUploader] = {}


def register_uploader(uploader: PlatformUploader) -> None:
    """Register a platform uploader."""
    _uploaders[uploader.platform_name] = uploader


def get_uploader(platform: str) -> PlatformUploader | None:
    """Get the registered uploader for a platform."""
    return _uploaders.get(platform)


def get_all_uploaders() -> dict[str, PlatformUploader]:
    """Get all registered uploaders."""
    return dict(_uploaders)


def _register_default_uploaders() -> None:
    """Register the default platform uploaders."""
    register_uploader(YouTubeShortsUploader())
    register_uploader(TikTokUploader())
    register_uploader(InstagramReelsUploader())
    register_uploader(FacebookReelsUploader())
    register_uploader(SnapchatSpotlightUploader())


# Auto-register defaults on import
_register_default_uploaders()


# =============================================================================
# Publishing Service
# =============================================================================


@dataclass
class PublishAttemptResult:
    """Result of a publish attempt including retries."""

    queue_id: uuid.UUID
    platform: str
    success: bool
    platform_video_id: str | None = None
    attempts: int = 0
    error: str | None = None


async def publish_video(
    *,
    queue_id: uuid.UUID,
    output_id: uuid.UUID,
    platform: str,
    video_path: str,
    title: str,
    description: str,
    tags: list[str] | None = None,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
) -> PublishAttemptResult:
    """Publish a video to a platform with retry logic.

    Uses exponential backoff on failure. Sends Discord alert on
    repeated failure after max_attempts.

    Args:
        queue_id: Publish queue entry ID.
        output_id: Output record ID.
        platform: Target platform.
        video_path: Path to the video file.
        title: Video title.
        description: Video description.
        tags: Optional tags.
        max_attempts: Maximum retry attempts.

    Returns:
        PublishAttemptResult with outcome.
    """
    uploader = get_uploader(platform)
    if uploader is None:
        return PublishAttemptResult(
            queue_id=queue_id,
            platform=platform,
            success=False,
            error=f"No uploader registered for platform: {platform}",
        )

    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = await uploader.upload(
                video_path=video_path,
                title=title,
                description=description,
                tags=tags,
            )

            if result.success:
                trace_pipeline_step(
                    pipeline_run_id=str(output_id),
                    step_name="publish",
                    status="success",
                    metadata={"platform": platform, "attempts": attempt},
                )
                return PublishAttemptResult(
                    queue_id=queue_id,
                    platform=platform,
                    success=True,
                    platform_video_id=result.platform_video_id,
                    attempts=attempt,
                )

            last_error = result.error
            logger.warning(
                "Upload attempt %d/%d failed for %s: %s",
                attempt,
                max_attempts,
                platform,
                result.error,
            )

        except Exception as e:
            last_error = str(e)
            logger.warning(
                "Upload attempt %d/%d raised exception for %s: %s",
                attempt,
                max_attempts,
                platform,
                e,
            )

        # Exponential backoff: 2^attempt seconds (2, 4, 8...)
        if attempt < max_attempts:
            backoff = 2**attempt
            logger.info("Retrying in %ds...", backoff)
            await asyncio.sleep(backoff)

    # All attempts exhausted
    record_metric("publish_failure", status="failed")
    await _send_publish_failure_alert(
        platform=platform,
        queue_id=queue_id,
        attempts=max_attempts,
        error=last_error or "Unknown error",
    )

    return PublishAttemptResult(
        queue_id=queue_id,
        platform=platform,
        success=False,
        attempts=max_attempts,
        error=last_error,
    )


async def process_publish_queue(
    queue_items: list[dict[str, Any]],
) -> list[PublishAttemptResult]:
    """Process a batch of publish queue items.

    Each item dict should have: queue_id, output_id, platform,
    video_path, title, description, and optional tags.

    Returns list of PublishAttemptResult, one per item.
    """
    results: list[PublishAttemptResult] = []

    for item in queue_items:
        result = await publish_video(
            queue_id=item["queue_id"],
            output_id=item["output_id"],
            platform=item["platform"],
            video_path=item["video_path"],
            title=item["title"],
            description=item.get("description", ""),
            tags=item.get("tags"),
        )
        results.append(result)

    return results


# =============================================================================
# Discord Alerting
# =============================================================================


async def _send_publish_failure_alert(
    *,
    platform: str,
    queue_id: uuid.UUID,
    attempts: int,
    error: str,
) -> None:
    """Send Discord alert for repeated publish failure."""
    settings = get_settings()
    webhook_url = settings.discord_webhook_url
    if not webhook_url:
        logger.warning("No Discord webhook URL configured — skipping publish failure alert")
        return

    payload = {
        "embeds": [
            {
                "title": "🚨 Publish Failed",
                "color": 0xE74C3C,
                "fields": [
                    {"name": "Platform", "value": platform, "inline": True},
                    {"name": "Queue ID", "value": f"`{queue_id}`", "inline": True},
                    {"name": "Attempts", "value": str(attempts), "inline": True},
                    {"name": "Error", "value": error[:1024], "inline": False},
                ],
            }
        ],
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info("Discord publish-failure alert sent for %s", platform)
    except Exception as e:
        logger.warning("Failed to send Discord alert: %s", e)
