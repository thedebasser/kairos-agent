"""Tests for the Human Review Dashboard (FastAPI).

Tests:
- Empty queue rendering
- Queue with items rendering
- Submit review (approve/reject)
- API endpoints (health, queue, add)
- Review action validation
- Navigation (previous/next)
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from kairos.schemas.contracts import ReviewAction
from kairos.web.review_app import create_review_app

pytestmark = pytest.mark.unit


@pytest.fixture
def app():
    """Create a fresh app instance for each test."""
    return create_review_app()


@pytest.fixture
async def client(app):
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _sample_output(output_id: str | None = None, **overrides) -> dict:
    """Create a sample output for the review queue."""
    data = {
        "output_id": output_id or str(uuid.uuid4()),
        "status": "pending_review",
        "title": "Test Ball Pit Video",
        "category": "ball_pit",
        "visual_brief": "200 balls cascade into a pit",
        "concept_brief": "Every collision spawns a new ball",
        "duration_sec": 65,
        "simulation_iteration": 3,
        "cost_usd": 0.16,
        "avg_fps": 32,
        "peak_body_count": 247,
        "payoff_timestamp_sec": 48,
        "validation_passed": True,
        "final_video_path": "",  # No real file in tests
        "captions": {
            "captions": [
                {"text": "What happens when...", "start_sec": 0, "end_sec": 2.5, "caption_type": "hook"}
            ]
        },
        "created_at": "2025-01-01T00:00:00",
    }
    data.update(overrides)
    return data


# =============================================================================
# Health & API Endpoints
# =============================================================================


class TestHealthEndpoint:
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["pending_reviews"] == 0


class TestApiQueue:
    async def test_empty_queue(self, client):
        resp = await client.get("/api/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    async def test_queue_after_adding_item(self, client):
        item = _sample_output()
        await client.post("/api/add", json=item)

        resp = await client.get("/api/queue")
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["title"] == "Test Ball Pit Video"


class TestApiAdd:
    async def test_add_item_to_queue(self, client):
        item = _sample_output(output_id="test-123")
        resp = await client.post("/api/add", json=item)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["output_id"] == "test-123"

    async def test_added_item_appears_in_health(self, client):
        item = _sample_output()
        await client.post("/api/add", json=item)
        resp = await client.get("/health")
        assert resp.json()["pending_reviews"] == 1


# =============================================================================
# Review Page Rendering
# =============================================================================


class TestReviewPage:
    async def test_empty_queue_shows_empty_state(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "No videos pending review" in resp.text

    async def test_queue_with_item_shows_video(self, client):
        item = _sample_output(output_id="vid-1")
        await client.post("/api/add", json=item)

        resp = await client.get("/")
        assert resp.status_code == 200
        assert "Test Ball Pit Video" in resp.text
        assert "ball_pit" in resp.text
        assert "1 pending" in resp.text

    async def test_stats_displayed(self, client):
        item = _sample_output()
        await client.post("/api/add", json=item)

        resp = await client.get("/")
        assert "65s" in resp.text  # duration
        assert "$0.16" in resp.text  # cost
        assert "32 avg" in resp.text  # fps
        assert "247 peak" in resp.text  # bodies

    async def test_captions_displayed(self, client):
        item = _sample_output()
        await client.post("/api/add", json=item)

        resp = await client.get("/")
        assert "What happens when..." in resp.text

    async def test_action_buttons_present(self, client):
        item = _sample_output()
        await client.post("/api/add", json=item)

        resp = await client.get("/")
        for action in ReviewAction:
            assert action.value in resp.text


# =============================================================================
# Navigation
# =============================================================================


class TestNavigation:
    async def test_single_item_no_navigation(self, client):
        await client.post("/api/add", json=_sample_output(output_id="a"))

        resp = await client.get("/")
        assert "1 of 1" in resp.text

    async def test_multiple_items_navigation(self, client):
        await client.post("/api/add", json=_sample_output(output_id="a", created_at="2025-01-01"))
        await client.post("/api/add", json=_sample_output(output_id="b", created_at="2025-01-02", title="Second Video"))

        resp = await client.get("/?index=0")
        assert "1 of 2" in resp.text

        resp2 = await client.get("/?index=1")
        assert "2 of 2" in resp2.text
        assert "Second Video" in resp2.text


# =============================================================================
# Review Submission
# =============================================================================


class TestReviewSubmission:
    async def test_approve_redirects(self, client):
        await client.post("/api/add", json=_sample_output(output_id="rev-1"))

        resp = await client.post(
            "/review/rev-1",
            data={"action": "approved", "feedback": ""},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/"

    async def test_approve_updates_status(self, client):
        await client.post("/api/add", json=_sample_output(output_id="rev-2"))

        await client.post(
            "/review/rev-2",
            data={"action": "approved", "feedback": "Looks great!"},
            follow_redirects=False,
        )

        # Item no longer in pending queue
        resp = await client.get("/api/queue")
        assert resp.json()["total"] == 0

    async def test_reject_updates_status(self, client):
        await client.post("/api/add", json=_sample_output(output_id="rev-3"))

        await client.post(
            "/review/rev-3",
            data={"action": "bad_concept", "feedback": "Boring idea"},
            follow_redirects=False,
        )

        resp = await client.get("/api/queue")
        assert resp.json()["total"] == 0

    async def test_invalid_action_rejected(self, client):
        await client.post("/api/add", json=_sample_output(output_id="rev-4"))

        resp = await client.post(
            "/review/rev-4",
            data={"action": "invalid_action", "feedback": ""},
        )
        assert resp.status_code == 400

    async def test_nonexistent_output_404(self, client):
        resp = await client.post(
            "/review/nonexistent",
            data={"action": "approved", "feedback": ""},
        )
        assert resp.status_code == 404

    async def test_all_review_actions_valid(self, client):
        """Every ReviewAction enum value is accepted."""
        for action in ReviewAction:
            oid = f"test-{action.value}"
            await client.post("/api/add", json=_sample_output(output_id=oid))

            resp = await client.post(
                f"/review/{oid}",
                data={"action": action.value, "feedback": ""},
                follow_redirects=False,
            )
            assert resp.status_code == 303, f"Failed for action {action.value}"


# =============================================================================
# Video Serving
# =============================================================================


class TestVideoServing:
    async def test_video_not_found_for_unknown_output(self, client):
        resp = await client.get("/video/unknown-id")
        assert resp.status_code == 404

    async def test_video_not_found_for_missing_file(self, client):
        await client.post("/api/add", json=_sample_output(
            output_id="vid-nofile",
            final_video_path="/nonexistent/path/video.mp4",
        ))
        resp = await client.get("/video/vid-nofile")
        assert resp.status_code == 404
