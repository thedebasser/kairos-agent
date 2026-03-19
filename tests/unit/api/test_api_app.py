"""Tests for the Kairos Agent REST API.

Validates:
- Health endpoint
- Run list (empty, with data, pagination, filters)
- Run detail (found, 404, invalid UUID)
- Run events (found, 404)
- Pipeline start + status

All DB access is mocked via dependency overrides so no real
PostgreSQL connection is needed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_run(
    *,
    pipeline_run_id: uuid.UUID | None = None,
    pipeline: str = "physics",
    status: str = "success",
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    total_cost_usd: float | None = 0.12,
) -> MagicMock:
    """Build a mock PipelineRun row."""
    row = MagicMock()
    row.pipeline_run_id = pipeline_run_id or uuid.uuid4()
    row.pipeline = pipeline
    row.status = status
    row.started_at = started_at or datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    row.completed_at = completed_at or datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc)
    row.total_cost_usd = total_cost_usd
    return row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_runs_dir(tmp_path: Path) -> Path:
    """Temporary runs directory."""
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs


@pytest.fixture()
def app(tmp_runs_dir: Path):
    """Create a fresh API app with overridden dependencies."""
    from kairos.api.app import create_app
    from kairos.api.deps import get_db, get_runs_dir

    application = create_app()

    # Override DB dependency — individual tests patch the mock session
    async def _override_db():
        yield MagicMock()

    application.dependency_overrides[get_db] = _override_db
    application.dependency_overrides[get_runs_dir] = lambda: tmp_runs_dir

    return application


@pytest.fixture()
async def client(app):
    """Async HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    async def test_health_returns_ok(self, client: AsyncClient):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "database" in body
        assert "version" in body


# ---------------------------------------------------------------------------
# Pipeline Status / Start
# ---------------------------------------------------------------------------

class TestPipeline:
    async def test_pipeline_status(self, client: AsyncClient):
        resp = await client.get("/api/pipeline/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "available_pipelines" in body
        assert "physics" in body["available_pipelines"]
        assert "domino" in body["available_pipelines"]

    async def test_pipeline_start(self, client: AsyncClient):
        resp = await client.post(
            "/api/pipeline/start",
            json={"pipeline": "physics"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pipeline"] == "physics"
        assert body["status"] == "started"
        assert "pipeline_run_id" in body

    async def test_pipeline_start_default(self, client: AsyncClient):
        resp = await client.post("/api/pipeline/start", json={})
        assert resp.status_code == 200
        assert resp.json()["pipeline"] == "physics"


# ---------------------------------------------------------------------------
# Run List
# ---------------------------------------------------------------------------

class TestRunList:
    async def test_list_runs_empty(self, app, client: AsyncClient):
        """No rows → empty list with total=0."""
        from kairos.api.deps import get_db

        mock_session = AsyncMock()
        # count query returns 0
        count_result = MagicMock()
        count_result.scalar_one.return_value = 0
        # data query returns empty
        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []

        mock_session.execute = AsyncMock(side_effect=[count_result, data_result])

        async def _db():
            yield mock_session

        app.dependency_overrides[get_db] = _db

        resp = await client.get("/api/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["runs"] == []

    async def test_list_runs_with_data(self, app, client: AsyncClient):
        """Returns formatted run summaries."""
        from kairos.api.deps import get_db

        run1 = _make_pipeline_run(status="success")
        run2 = _make_pipeline_run(pipeline="domino", status="running")

        mock_session = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 2
        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = [run1, run2]
        mock_session.execute = AsyncMock(side_effect=[count_result, data_result])

        async def _db():
            yield mock_session

        app.dependency_overrides[get_db] = _db

        resp = await client.get("/api/runs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["runs"]) == 2
        assert body["runs"][0]["status"] == "success"
        assert body["runs"][1]["pipeline"] == "domino"

    async def test_list_runs_pagination(self, app, client: AsyncClient):
        """Pagination params are echoed back."""
        from kairos.api.deps import get_db

        mock_session = AsyncMock()
        count_result = MagicMock()
        count_result.scalar_one.return_value = 50
        data_result = MagicMock()
        data_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(side_effect=[count_result, data_result])

        async def _db():
            yield mock_session

        app.dependency_overrides[get_db] = _db

        resp = await client.get("/api/runs", params={"limit": 5, "offset": 10})
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 5
        assert body["offset"] == 10
        assert body["total"] == 50


# ---------------------------------------------------------------------------
# Run Detail
# ---------------------------------------------------------------------------

class TestRunDetail:
    async def test_get_run_found(self, app, client: AsyncClient, tmp_runs_dir: Path):
        """Returns run detail enriched with run_summary.json."""
        from kairos.api.deps import get_db

        run_id = uuid.uuid4()
        row = _make_pipeline_run(pipeline_run_id=run_id)

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=row)

        async def _db():
            yield mock_session

        app.dependency_overrides[get_db] = _db

        # Create a run_summary.json
        run_dir = tmp_runs_dir / str(run_id)
        run_dir.mkdir()
        summary = {
            "concept_title": "Chaos Cascade",
            "total_duration_ms": 12345,
            "total_llm_calls": 7,
            "final_video_path": "/output/final.mp4",
            "errors": [],
            "steps": [
                {"step": "concept", "step_number": 1, "attempt": 1, "status": "success", "duration_ms": 2000},
                {"step": "simulate", "step_number": 2, "attempt": 1, "status": "success", "duration_ms": 8000},
            ],
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary))

        resp = await client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pipeline_run_id"] == str(run_id)
        assert body["concept_title"] == "Chaos Cascade"
        assert body["total_llm_calls"] == 7
        assert len(body["steps"]) == 2
        assert body["steps"][0]["step"] == "concept"

    async def test_get_run_not_found(self, app, client: AsyncClient):
        """404 when run doesn't exist."""
        from kairos.api.deps import get_db

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=None)

        async def _db():
            yield mock_session

        app.dependency_overrides[get_db] = _db

        run_id = uuid.uuid4()
        resp = await client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 404

    async def test_get_run_invalid_uuid(self, client: AsyncClient):
        """400 on malformed UUID."""
        resp = await client.get("/api/runs/not-a-uuid")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Run Events
# ---------------------------------------------------------------------------

class TestRunEvents:
    async def test_get_events(self, client: AsyncClient, tmp_runs_dir: Path):
        """Returns events from events.jsonl."""
        run_id = str(uuid.uuid4())
        run_dir = tmp_runs_dir / run_id
        run_dir.mkdir()

        events = [
            {"event_type": "run_started", "event_id": "e1", "run_id": run_id, "timestamp": "2025-01-15T10:00:00Z"},
            {"event_type": "step_completed", "event_id": "e2", "run_id": run_id, "timestamp": "2025-01-15T10:01:00Z"},
        ]
        (run_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events)
        )

        resp = await client.get(f"/api/runs/{run_id}/events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == run_id
        assert body["count"] == 2
        assert body["events"][0]["event_type"] == "run_started"

    async def test_get_events_not_found(self, client: AsyncClient):
        """404 when no events file."""
        run_id = str(uuid.uuid4())
        resp = await client.get(f"/api/runs/{run_id}/events")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    """Validate Pydantic response models."""

    def test_health_response(self):
        from kairos.api.schemas import HealthResponse

        h = HealthResponse(status="ok", database="connected")
        assert h.status == "ok"
        assert h.version == "0.3.0"

    def test_run_summary_response(self):
        from kairos.api.schemas import RunSummaryResponse

        r = RunSummaryResponse(
            pipeline_run_id="abc-123",
            pipeline="physics",
            status="success",
        )
        assert r.pipeline == "physics"
        assert r.total_cost_usd is None

    def test_run_list_response(self):
        from kairos.api.schemas import RunListResponse

        r = RunListResponse(runs=[], total=0, limit=20, offset=0)
        assert r.total == 0
        assert r.runs == []
