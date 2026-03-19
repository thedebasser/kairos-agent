"""Kairos Agent -- API Dependency Injection.

FastAPI dependency functions used across all routes.  All DB access
flows through ``get_db`` so tests can override the session factory.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Root directories (resolved once at import time)
RUNS_DIR: Path = Path(__file__).resolve().parents[3] / "runs"


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session, committing on success."""
    from kairos.db.session import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_runs_dir() -> Path:
    """Return the root ``runs/`` directory."""
    return RUNS_DIR
