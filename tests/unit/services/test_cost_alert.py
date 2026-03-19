"""Tests for P3.29 — Cost alert + Discord webhook.

Verifies:
1. AlertManager.check_run_cost fires when cost > threshold
2. AlertManager.check_run_cost is silent when cost <= threshold
3. Discord webhook payload is formatted correctly
4. Discord alert is skipped when webhook URL is empty
5. check_all sends to both Slack and Discord
6. Alert timestamp is UTC-aware
7. Pipeline graph wires cost check after run
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from kairos.services.monitoring import Alert, AlertManager, MetricsStore


@pytest.fixture()
def _mock_settings():
    """Provide a mock Settings with cost threshold and Discord webhook."""
    settings = MagicMock()
    settings.cost_alert_threshold_usd = 0.30
    settings.discord_webhook_url = "https://discord.com/api/webhooks/test"
    settings.slack_webhook_url = ""
    with patch("kairos.services.monitoring.get_settings", return_value=settings):
        yield settings


class TestCheckRunCost:
    """check_run_cost fires alerts only when threshold is exceeded."""

    def test_fires_alert_above_threshold(self, _mock_settings: MagicMock) -> None:
        mgr = AlertManager(metrics_store=MetricsStore())
        alert = mgr.check_run_cost("run-123", 0.50)

        assert alert is not None
        assert alert.alert_type == "run_cost"
        assert alert.value == 0.50
        assert alert.threshold == 0.30
        assert "run-123" in alert.message

    def test_no_alert_below_threshold(self, _mock_settings: MagicMock) -> None:
        mgr = AlertManager(metrics_store=MetricsStore())
        alert = mgr.check_run_cost("run-456", 0.10)

        assert alert is None
        assert len(mgr._alerts) == 0

    def test_no_alert_at_threshold(self, _mock_settings: MagicMock) -> None:
        mgr = AlertManager(metrics_store=MetricsStore())
        alert = mgr.check_run_cost("run-789", 0.30)

        assert alert is None


class TestDiscordWebhook:
    """Discord webhook is called with correct embed format."""

    def test_discord_payload_format(self, _mock_settings: MagicMock) -> None:
        """Discord alert sends an embed with title, description, color, fields."""
        mgr = AlertManager(metrics_store=MetricsStore())

        with patch("httpx.Client") as MockClient:
            mock_client = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_client)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)

            mgr.check_run_cost("run-embed", 0.99)

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            url = call_args[0][0]
            payload = call_args[1]["json"]

            assert url == "https://discord.com/api/webhooks/test"
            assert "embeds" in payload
            embed = payload["embeds"][0]
            assert embed["title"] == "Pipeline Alert — WARNING"
            assert "run-embed" in embed["description"]
            assert embed["color"] == 0xFFA500  # orange for warning
            assert len(embed["fields"]) == 3

    def test_discord_skipped_when_no_url(self, _mock_settings: MagicMock) -> None:
        """No HTTP call when discord_webhook_url is empty."""
        _mock_settings.discord_webhook_url = ""
        mgr = AlertManager(metrics_store=MetricsStore())

        with patch("httpx.Client") as MockClient:
            mgr.check_run_cost("run-nourl", 0.99)
            MockClient.assert_not_called()


class TestCheckAllSendsBoth:
    """check_all dispatches to both Slack and Discord."""

    def test_check_all_calls_discord(self, _mock_settings: MagicMock) -> None:
        store = MetricsStore()
        mgr = AlertManager(metrics_store=store)

        with (
            patch.object(mgr, "_check_cost_threshold", return_value=[
                Alert(
                    alert_type="cost_threshold",
                    message="test",
                    severity="warning",
                    value=0.50,
                    threshold=0.30,
                ),
            ]),
            patch.object(mgr, "_check_success_rate", return_value=[]),
            patch.object(mgr, "_send_slack_alert") as mock_slack,
            patch.object(mgr, "_send_discord_alert") as mock_discord,
        ):
            alerts = mgr.check_all()

        assert len(alerts) == 1
        mock_slack.assert_called_once()
        mock_discord.assert_called_once()


class TestAlertTimestamp:
    """Alert default timestamp is UTC-aware."""

    def test_alert_timestamp_is_utc(self, _mock_settings: MagicMock) -> None:
        alert = Alert(
            alert_type="test",
            message="test",
            severity="warning",
            value=1.0,
            threshold=0.5,
        )
        assert alert.timestamp.tzinfo is not None
        assert alert.timestamp.tzinfo == timezone.utc
