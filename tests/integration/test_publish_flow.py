"""Integration test: approved video → queued → published (mocked)."""

from __future__ import annotations

import uuid

import pytest

from kairos.services.publish_queue import generate_platform_metadata
from kairos.services.uploader import (
    process_publish_queue,
    publish_video,
)


class TestApprovedToPublished:
    """End-to-end flow: approved output → queue → published on platforms."""

    async def test_approved_video_publishes_to_youtube(self, tmp_path):
        """Simulate the full flow: video exists, metadata generated, upload succeeds."""
        video = tmp_path / "final_video.mp4"
        video.write_bytes(b"mp4_header_fake_data" * 100)

        output_id = uuid.uuid4()
        queue_id = uuid.uuid4()

        # Step 1: Generate platform metadata
        metadata = generate_platform_metadata(
            base_title="Chaos Balls",
            platform="youtube_shorts",
            category="physics_chaos",
        )
        assert "title" in metadata
        assert "description" in metadata

        # Step 2: Publish
        result = await publish_video(
            queue_id=queue_id,
            output_id=output_id,
            platform="youtube_shorts",
            video_path=str(video),
            title=metadata["title"],
            description=metadata["description"],
        )

        assert result.success is True
        assert result.platform_video_id is not None
        assert result.attempts == 1

    async def test_approved_video_publishes_to_tiktok(self, tmp_path):
        video = tmp_path / "final_video.mp4"
        video.write_bytes(b"data" * 50)

        metadata = generate_platform_metadata(
            base_title="Bouncing Spheres",
            platform="tiktok",
            category="domino_cascade",
        )

        result = await publish_video(
            queue_id=uuid.uuid4(),
            output_id=uuid.uuid4(),
            platform="tiktok",
            video_path=str(video),
            title=metadata["title"],
            description=metadata["description"],
        )
        assert result.success is True

    async def test_batch_publish_multiple_platforms(self, tmp_path):
        """Publish same video to multiple platforms in one batch."""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"video_data")

        output_id = uuid.uuid4()

        items = []
        for platform in ("youtube_shorts", "tiktok"):
            meta = generate_platform_metadata(
                base_title="Multi-Platform Test",
                platform=platform,
                category="fluid_dynamics",
            )
            items.append({
                "queue_id": uuid.uuid4(),
                "output_id": output_id,
                "platform": platform,
                "video_path": str(video),
                "title": meta["title"],
                "description": meta["description"],
            })

        results = await process_publish_queue(items)
        assert len(results) == 2
        assert all(r.success for r in results)
