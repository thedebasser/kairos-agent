"""Kairos Agent — Analytics Sync Service.

Periodic job that pulls engagement metrics from platform APIs
(YouTube Analytics, TikTok Analytics) and writes them to the
publish_log table. Also updates category_stats based on real
performance data, closing the feedback loop to the Idea Agent.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Engagement Metrics Data
# =============================================================================


@dataclass
class EngagementMetrics:
    """Engagement metrics fetched from a platform API."""

    platform_video_id: str
    platform: str
    views_7d: int = 0
    views_30d: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    avg_view_duration_sec: float = 0.0
    retention_rate: float = 0.0  # 0.0 to 1.0
    revenue_usd: float = 0.0
    fetched_at: datetime = field(default_factory=datetime.now)


@dataclass
class CategoryPerformance:
    """Aggregated performance for a category across all platforms."""

    category: str
    video_count: int = 0
    total_views: int = 0
    avg_retention: float = 0.0
    avg_likes_per_video: float = 0.0
    total_revenue_usd: float = 0.0


# =============================================================================
# Platform Analytics ABC
# =============================================================================


class PlatformAnalyticsFetcher(ABC):
    """Abstract base class for platform analytics adapters."""

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Platform identifier."""

    @abstractmethod
    async def fetch_metrics(
        self,
        platform_video_ids: list[str],
    ) -> list[EngagementMetrics]:
        """Fetch engagement metrics for a list of video IDs.

        Args:
            platform_video_ids: List of platform-specific video IDs.

        Returns:
            List of EngagementMetrics, one per video ID.
        """


class YouTubeAnalyticsFetcher(PlatformAnalyticsFetcher):
    """YouTube Analytics API integration.

    TODO: Implement real YouTube Analytics API calls:
    1. OAuth2 token refresh
    2. GET /v2/reports with video filter
    3. Parse response into EngagementMetrics
    """

    @property
    def platform_name(self) -> str:
        return "youtube_shorts"

    async def fetch_metrics(
        self,
        platform_video_ids: list[str],
    ) -> list[EngagementMetrics]:
        logger.info("Fetching YouTube analytics for %d videos", len(platform_video_ids))

        # Placeholder: would call YouTube Analytics API
        results = []
        for vid_id in platform_video_ids:
            results.append(
                EngagementMetrics(
                    platform_video_id=vid_id,
                    platform="youtube_shorts",
                )
            )
        return results


class TikTokAnalyticsFetcher(PlatformAnalyticsFetcher):
    """TikTok Analytics API integration.

    TODO: Implement real TikTok Research API / Creator API:
    1. Refresh access token
    2. GET /v2/video/query with video_ids
    3. Parse response into EngagementMetrics
    """

    @property
    def platform_name(self) -> str:
        return "tiktok"

    async def fetch_metrics(
        self,
        platform_video_ids: list[str],
    ) -> list[EngagementMetrics]:
        logger.info("Fetching TikTok analytics for %d videos", len(platform_video_ids))

        results = []
        for vid_id in platform_video_ids:
            results.append(
                EngagementMetrics(
                    platform_video_id=vid_id,
                    platform="tiktok",
                )
            )
        return results


# Registry
_fetchers: dict[str, PlatformAnalyticsFetcher] = {}


def register_fetcher(fetcher: PlatformAnalyticsFetcher) -> None:
    """Register an analytics fetcher."""
    _fetchers[fetcher.platform_name] = fetcher


def get_fetcher(platform: str) -> PlatformAnalyticsFetcher | None:
    """Get the registered fetcher for a platform."""
    return _fetchers.get(platform)


def _register_default_fetchers() -> None:
    register_fetcher(YouTubeAnalyticsFetcher())
    register_fetcher(TikTokAnalyticsFetcher())


_register_default_fetchers()


# =============================================================================
# Analytics Sync Job
# =============================================================================


@dataclass
class SyncResult:
    """Result of an analytics sync run."""

    platform: str
    videos_synced: int = 0
    videos_failed: int = 0
    categories_updated: int = 0
    errors: list[str] = field(default_factory=list)


async def sync_platform_analytics(
    *,
    platform: str,
    publish_logs: list[dict[str, Any]],
) -> SyncResult:
    """Sync analytics for a platform from published video logs.

    Args:
        platform: Platform to sync.
        publish_logs: List of dicts with keys:
            - platform_video_id: str
            - output_id: str
            - category: str (optional)

    Returns:
        SyncResult with counts.
    """
    fetcher = get_fetcher(platform)
    if fetcher is None:
        return SyncResult(
            platform=platform,
            errors=[f"No analytics fetcher for platform: {platform}"],
        )

    video_ids = [log["platform_video_id"] for log in publish_logs if log.get("platform_video_id")]

    if not video_ids:
        return SyncResult(platform=platform)

    try:
        metrics_list = await fetcher.fetch_metrics(video_ids)
    except Exception as e:
        logger.error("Failed to fetch analytics for %s: %s", platform, e)
        return SyncResult(platform=platform, errors=[str(e)])

    # Build a lookup: video_id → metrics
    metrics_by_id = {m.platform_video_id: m for m in metrics_list}

    synced = 0
    failed = 0

    for log_entry in publish_logs:
        vid_id = log_entry.get("platform_video_id")
        if not vid_id or vid_id not in metrics_by_id:
            failed += 1
            continue

        # Write metrics back to publish_log in database
        publish_id = log_entry.get("publish_id")
        if publish_id:
            try:
                from kairos.db.session import async_session_factory
                from kairos.db.operations import update_publish_log_metrics
                from uuid import UUID

                metrics = metrics_by_id[vid_id]
                async with async_session_factory() as session:
                    await update_publish_log_metrics(
                        session,
                        UUID(str(publish_id)),
                        views_7d=metrics.views_7d,
                        views_30d=metrics.views_30d,
                        likes=metrics.likes,
                        comments=metrics.comments,
                        shares=metrics.shares,
                        avg_view_duration_sec=metrics.avg_view_duration_sec,
                        retention_rate=metrics.retention_rate,
                        revenue_usd=metrics.revenue_usd,
                    )
                    await session.commit()
            except Exception as e:
                logger.warning("Failed to update publish_log %s: %s", publish_id, e)

        synced += 1

    # Aggregate by category for category_stats updates
    categories_updated = _aggregate_category_performance(publish_logs, metrics_by_id)

    # Write category performance back to database
    if categories_updated:
        try:
            from kairos.db.session import async_session_factory
            from kairos.db.operations import update_category_stat_performance

            async with async_session_factory() as session:
                for cat_name, cat_perf in categories_updated.items():
                    await update_category_stat_performance(
                        session,
                        pipeline=log_entry.get("pipeline", "physics"),
                        category=cat_name,
                        avg_performance=cat_perf.avg_retention * 100,
                        videos_last_30_days=cat_perf.video_count,
                    )
                await session.commit()
                logger.info("Updated %d category stats", len(categories_updated))
        except Exception as e:
            logger.warning("Failed to update category_stats: %s", e)

    return SyncResult(
        platform=platform,
        videos_synced=synced,
        videos_failed=failed,
        categories_updated=len(categories_updated),
    )


def _aggregate_category_performance(
    publish_logs: list[dict[str, Any]],
    metrics_by_id: dict[str, EngagementMetrics],
) -> dict[str, CategoryPerformance]:
    """Aggregate engagement metrics by category.

    Returns dict of category → CategoryPerformance.
    """
    by_category: dict[str, CategoryPerformance] = {}

    for log_entry in publish_logs:
        category = log_entry.get("category", "unknown")
        vid_id = log_entry.get("platform_video_id")

        if vid_id not in metrics_by_id:
            continue

        metrics = metrics_by_id[vid_id]

        if category not in by_category:
            by_category[category] = CategoryPerformance(category=category)

        cp = by_category[category]
        cp.video_count += 1
        cp.total_views += metrics.views_7d
        cp.total_revenue_usd += metrics.revenue_usd

        # Running average for retention
        if cp.video_count > 0:
            cp.avg_retention = (
                (cp.avg_retention * (cp.video_count - 1) + metrics.retention_rate)
                / cp.video_count
            )
            cp.avg_likes_per_video = (
                (cp.avg_likes_per_video * (cp.video_count - 1) + metrics.likes)
                / cp.video_count
            )

    return by_category


async def run_analytics_sync(
    platforms: list[str] | None = None,
    publish_logs_by_platform: dict[str, list[dict[str, Any]]] | None = None,
) -> list[SyncResult]:
    """Run analytics sync across all (or specified) platforms.

    Args:
        platforms: List of platforms to sync. Defaults to all registered.
        publish_logs_by_platform: Dict mapping platform → list of publish_log entries.

    Returns:
        List of SyncResult, one per platform.
    """
    if platforms is None:
        platforms = list(_fetchers.keys())

    if publish_logs_by_platform is None:
        publish_logs_by_platform = {}

    results = []
    for platform in platforms:
        logs = publish_logs_by_platform.get(platform, [])
        result = await sync_platform_analytics(platform=platform, publish_logs=logs)
        results.append(result)
        logger.info(
            "Analytics sync %s: %d synced, %d failed",
            platform,
            result.videos_synced,
            result.videos_failed,
        )

    return results
