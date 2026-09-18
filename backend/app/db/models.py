"""SQLAlchemy models (Design.md §2.2).

Column subset for Phase 1: the query path here is retrieve -> generate -> answer
with *no* verification, so `messages` stores what that path produces (question,
answer JSON, the chunk_ids that were retrieved, latency). The confidence,
`router_decision`, `flagged`, and `fallback_used` columns from Design.md §2.2
land in Phases 2-3 with the code that fills them, rather than as dead nullable
columns now.

JSON columns render as `jsonb` on PostgreSQL (Design.md §2.2) and fall back to
portable `JSON` on SQLite, which backs the test suite.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# `jsonb` on Postgres, `json` everywhere else (SQLite in tests).
JsonColumn = JSON().with_variant(JSONB(), "postgresql")

DocumentStatus = String(16)  # 'processing' | 'ready' | 'failed' (Design.md §1.1)


class Base(DeclarativeBase):
    """Declarative base for every table."""


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(DocumentStatus, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False
    )
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class Message(Base):
    __tablename__ = "messages"

    msg_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # The full structured answer payload (answer text + citations) as returned
    # to the client, so history replays exactly what was shown.
    answer_json: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JsonColumn, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship(back_populates="messages")
