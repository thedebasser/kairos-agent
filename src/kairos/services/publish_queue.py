"""Kairos Agent — Publish Queue Service.

Manages the publishing queue: scheduling, status transitions, retry logic.
Decouples video production from platform distribution.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from kairos.schemas.contracts import PublishStatus

logger = logging.getLogger(__name__)

# Platform-specific optimal posting times (UTC)
PUBLISH_SCHEDULE: dict[str, dict[str, dict[str, object]]] = {
    "youtube_shorts": {
        "channel_1": {"times": ["09:00", "17:00"], "max_daily": 2},
    },
    "tiktok": {
        "account_1": {"times": ["08:00", "12:00", "18:00"], "max_daily": 3},
    },
    "instagram_reels": {
        "account_1": {"times": ["11:00"], "max_daily": 1},
    },
    "facebook_reels": {
        "account_1": {"times": ["10:00"], "max_daily": 1},
    },
    "snapchat_spotlight": {
        "account_1": {"times": ["09:00", "14:00"], "max_daily": 2},
    },
}

MAX_RETRY_ATTEMPTS = 3


def get_next_publish_time(
    platform: str,
    account: str,
    *,
    after: datetime | None = None,
) -> datetime | None:
    """Calculate the next available publish time for a platform/account.

    Args:
        platform: Platform name (e.g., 'youtube_shorts').
        account: Account identifier.
        after: Earliest allowed time (defaults to now).

    Returns:
        Next scheduled datetime, or None if platform not configured.
    """
    if platform not in PUBLISH_SCHEDULE:
        logger.warning("No schedule configured for platform: %s", platform)
        return None

    accounts = PUBLISH_SCHEDULE[platform]
    if account not in accounts:
        logger.warning("No schedule for account %s on %s", account, platform)
        return None

    config = accounts[account]
    scheduled_times: list[str] = config.get("times", [])  # type: ignore[assignment]
    if not scheduled_times:
        return after or datetime.now(timezone.utc)

    reference = after or datetime.now(timezone.utc)

    # Find the next available time slot from the schedule
    for time_str in sorted(scheduled_times):
        hour, minute = map(int, time_str.split(":"))
        candidate = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate > reference:
            return candidate

    # All slots today are past — use the first slot tomorrow
    from datetime import timedelta
    first_hour, first_minute = map(int, sorted(scheduled_times)[0].split(":"))
    tomorrow = reference + timedelta(days=1)
    return tomorrow.replace(hour=first_hour, minute=first_minute, second=0, microsecond=0)


def should_retry(status: str, attempts: int) -> bool:
    """Check if a failed publish should be retried.

    Args:
        status: Current publish queue status.
        attempts: Number of attempts so far.

    Returns:
        True if the item should be retried.
    """
    return status == PublishStatus.FAILED.value and attempts < MAX_RETRY_ATTEMPTS


def generate_platform_metadata(
    *,
    base_title: str,
    platform: str,
    category: str,
) -> dict[str, str]:
    """Generate platform-specific title and description.

    Args:
        base_title: The base video title.
        platform: Target platform.
        category: Video niche category.

    Returns:
        Dict with 'title' and 'description' keys.
    """
    hashtags = f"#{category.replace('_', '')} #physics #simulation #oddlysatisfying"

    platform_metadata: dict[str, dict[str, str]] = {
        "youtube_shorts": {
            "title": base_title,
            "description": f"{base_title}\n\n{hashtags}",
        },
        "tiktok": {
            "title": f"{base_title} {hashtags}",
            "description": "",
        },
        "instagram_reels": {
            "title": base_title,
            "description": hashtags,
        },
        "facebook_reels": {
            "title": base_title,
            "description": f"{base_title}\n{hashtags}",
        },
        "snapchat_spotlight": {
            "title": base_title,
            "description": "",
        },
    }

    return platform_metadata.get(platform, {"title": base_title, "description": ""})
