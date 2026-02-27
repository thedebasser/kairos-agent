"""Tests for kairos.services.analytics_sync — Analytics Sync Service."""

from __future__ import annotations

import pytest

from kairos.services.analytics_sync import (
    CategoryPerformance,
    EngagementMetrics,
    SyncResult,
    TikTokAnalyticsFetcher,
    YouTubeAnalyticsFetcher,
    _aggregate_category_performance,
    get_fetcher,
    run_analytics_sync,
    sync_platform_analytics,
)


# ── Data classes ──────────────────────────────────────────────────────

class TestEngagementMetrics:
    def test_defaults(self):
        m = EngagementMetrics(platform_video_id="vid1", platform="youtube_shorts")
        assert m.views_7d == 0
        assert m.retention_rate == 0.0
        assert m.revenue_usd == 0.0

    def test_custom_values(self):
        m = EngagementMetrics(
            platform_video_id="vid1",
            platform="tiktok",
            views_7d=5000,
            likes=200,
            retention_rate=0.65,
        )
        assert m.views_7d == 5000
        assert m.likes == 200


class TestCategoryPerformance:
    def test_defaults(self):
        cp = CategoryPerformance(category="physics_chaos")
        assert cp.video_count == 0
        assert cp.total_views == 0

    def test_custom(self):
        cp = CategoryPerformance(
            category="domino_cascade",
            video_count=5,
            total_views=10000,
            avg_retention=0.7,
        )
        assert cp.video_count == 5


class TestSyncResult:
    def test_defaults(self):
        r = SyncResult(platform="youtube_shorts")
        assert r.videos_synced == 0
        assert r.errors == []


# ── Fetchers ──────────────────────────────────────────────────────────

class TestYouTubeFetcher:
    async def test_fetch_returns_metrics(self):
        fetcher = YouTubeAnalyticsFetcher()
        assert fetcher.platform_name == "youtube_shorts"

        results = await fetcher.fetch_metrics(["vid1", "vid2"])
        assert len(results) == 2
        assert all(m.platform == "youtube_shorts" for m in results)

    async def test_fetch_empty(self):
        fetcher = YouTubeAnalyticsFetcher()
        results = await fetcher.fetch_metrics([])
        assert results == []


class TestTikTokFetcher:
    async def test_fetch_returns_metrics(self):
        fetcher = TikTokAnalyticsFetcher()
        assert fetcher.platform_name == "tiktok"

        results = await fetcher.fetch_metrics(["tt1"])
        assert len(results) == 1
        assert results[0].platform == "tiktok"


class TestFetcherRegistry:
    def test_get_youtube_fetcher(self):
        f = get_fetcher("youtube_shorts")
        assert f is not None
        assert f.platform_name == "youtube_shorts"

    def test_get_tiktok_fetcher(self):
        f = get_fetcher("tiktok")
        assert f is not None

    def test_unknown_platform(self):
        assert get_fetcher("snapchat") is None


# ── sync_platform_analytics ──────────────────────────────────────────

class TestSyncPlatformAnalytics:
    async def test_no_fetcher(self):
        result = await sync_platform_analytics(
            platform="unknown_platform",
            publish_logs=[],
        )
        assert result.videos_synced == 0
        assert len(result.errors) == 1

    async def test_empty_logs(self):
        result = await sync_platform_analytics(
            platform="youtube_shorts",
            publish_logs=[],
        )
        assert result.videos_synced == 0
        assert result.videos_failed == 0

    async def test_sync_with_logs(self):
        logs = [
            {"platform_video_id": "yt_abc", "output_id": "out1", "category": "physics"},
            {"platform_video_id": "yt_def", "output_id": "out2", "category": "physics"},
        ]
        result = await sync_platform_analytics(
            platform="youtube_shorts",
            publish_logs=logs,
        )
        assert result.videos_synced == 2
        assert result.videos_failed == 0

    async def test_sync_missing_video_id(self):
        logs = [
            {"output_id": "out1", "category": "physics"},  # No platform_video_id
        ]
        result = await sync_platform_analytics(
            platform="youtube_shorts",
            publish_logs=logs,
        )
        assert result.videos_synced == 0


# ── Category aggregation ─────────────────────────────────────────────

class TestCategoryAggregation:
    def test_aggregate_single_category(self):
        metrics = {
            "vid1": EngagementMetrics(
                platform_video_id="vid1",
                platform="yt",
                views_7d=1000,
                likes=50,
                retention_rate=0.6,
            ),
            "vid2": EngagementMetrics(
                platform_video_id="vid2",
                platform="yt",
                views_7d=2000,
                likes=100,
                retention_rate=0.8,
            ),
        }
        logs = [
            {"platform_video_id": "vid1", "category": "physics"},
            {"platform_video_id": "vid2", "category": "physics"},
        ]

        result = _aggregate_category_performance(logs, metrics)
        assert "physics" in result
        assert result["physics"].video_count == 2
        assert result["physics"].total_views == 3000

    def test_aggregate_multiple_categories(self):
        metrics = {
            "vid1": EngagementMetrics(platform_video_id="vid1", platform="yt", views_7d=500),
            "vid2": EngagementMetrics(platform_video_id="vid2", platform="yt", views_7d=1500),
        }
        logs = [
            {"platform_video_id": "vid1", "category": "domino"},
            {"platform_video_id": "vid2", "category": "fluid"},
        ]

        result = _aggregate_category_performance(logs, metrics)
        assert len(result) == 2
        assert result["domino"].total_views == 500
        assert result["fluid"].total_views == 1500

    def test_aggregate_missing_metrics(self):
        metrics = {}
        logs = [{"platform_video_id": "vid1", "category": "physics"}]
        result = _aggregate_category_performance(logs, metrics)
        assert len(result) == 0


# ── run_analytics_sync ────────────────────────────────────────────────

class TestRunAnalyticsSync:
    async def test_sync_all_platforms(self):
        results = await run_analytics_sync()
        assert len(results) >= 2  # youtube + tiktok
        assert all(isinstance(r, SyncResult) for r in results)

    async def test_sync_specific_platform(self):
        results = await run_analytics_sync(platforms=["youtube_shorts"])
        assert len(results) == 1
        assert results[0].platform == "youtube_shorts"

    async def test_sync_with_data(self):
        logs = {
            "youtube_shorts": [
                {"platform_video_id": "yt_abc", "output_id": "o1", "category": "physics"},
            ],
        }
        results = await run_analytics_sync(
            platforms=["youtube_shorts"],
            publish_logs_by_platform=logs,
        )
        assert results[0].videos_synced == 1
