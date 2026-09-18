"""Engine and session lifecycle.

The engine is built from `settings.database_url` and held on `app.state.db`,
mirroring how config is held on `app.state.config`: nothing reaches for a global
connection, and tests point the same code at an in-memory SQLite database.

Sessions are synchronous (psycopg3 sync). Route handlers that touch the database
are declared `def`, so FastAPI runs them in a worker thread and the event loop
is never blocked.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base


def _make_engine(url: str) -> Engine:
    """Build an engine, special-casing in-memory SQLite used by tests.

    A `:memory:` database lives for exactly as long as its connection, so the
    test engine must hand out one shared connection (`StaticPool`) and allow
    cross-thread use (FastAPI's threadpool).
    """
    if url.startswith("sqlite") and ":memory:" in url:
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            future=True,
        )
    return create_engine(url, pool_pre_ping=True, future=True)


class Database:
    """Owns the engine and session factory for one application instance."""

    def __init__(self, url: str) -> None:
        self.engine = _make_engine(url)
        self._sessionmaker = sessionmaker(
            bind=self.engine, autoflush=False, expire_on_commit=False, future=True
        )

    def create_all(self) -> None:
        """Create any missing tables.

        Phase 1 has no migration tool: the schema is small and the dev/CI
        databases are disposable. Alembic arrives when a column has to change
        on a database that holds real data.
        """
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A transactional scope: commit on success, roll back on error."""
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
