"""Kairos Agent — Database Session Management.

Async SQLAlchemy session factory for PostgreSQL.
Engine and session factory are cached as module-level singletons to prevent
connection pool leaks (Finding 4.5 — kairos_architectural_review).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from kairos.config import get_settings

logger = logging.getLogger(__name__)

# Module-level singletons (Finding 4.5)
_engine: AsyncEngine | None = None
_session_factory_instance: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get the async SQLAlchemy engine (cached singleton)."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
        logger.info("Created async SQLAlchemy engine")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the session factory (cached singleton, bound to the engine)."""
    global _session_factory_instance  # noqa: PLW0603
    if _session_factory_instance is None:
        engine = get_engine()
        _session_factory_instance = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory_instance


def async_session_factory() -> AsyncSession:
    """Create a new AsyncSession from the singleton factory.

    Shorthand for ``get_session_factory()()`` — the factory returns an
    ``async_sessionmaker``, which must itself be *called* to produce a
    session.  Callers use: ``async with async_session_factory() as session:``
    """
    return get_session_factory()()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection helper — yields an async session."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Dispose the engine and release all pooled connections.

    Call during clean shutdown.
    """
    global _engine, _session_factory_instance  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory_instance = None
        logger.info("Disposed async SQLAlchemy engine")
