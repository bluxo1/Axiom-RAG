"""SQLAlchemy models (Design.md §2.2).

`messages` stores what one chat turn produces. Phase 3 adds the confidence
routing columns from Design.md §2.2 (`confidence`, `confidence_breakdown`,
`router_decision`, `flagged`, `fallback_used`) alongside the Phase 1 columns,
now that the code that fills them exists — Rule 7: every routing decision is
persisted with its retrieved chunks and scores.

`spend_log` is included now (Rule 8): the LLM and embedding calls that Phase 1
introduced are the first spend, so the budget guard and its counters land with
them — not in Phase 3.

JSON columns render as `jsonb` on PostgreSQL (Design.md §2.2) and fall back to
portable `JSON` on SQLite, which backs the test suite.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
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
    # The full structured answer payload (answer text + citations + confidence)
    # as returned to the client, so history replays exactly what was shown.
    answer_json: Mapped[dict[str, Any]] = mapped_column(JsonColumn, nullable=False)
    # Confidence routing (Design.md §2.2, Architecture.md §3.5), populated from
    # Phase 3. Nullable because a refusal (no grounded answer) has no score.
    confidence: Mapped[float | None] = mapped_column(Numeric(6, 4), nullable=True)
    confidence_breakdown: Mapped[dict[str, Any] | None] = mapped_column(JsonColumn, nullable=True)
    # 'answer' | 'flag' | 'fallback' | 'refusal' (Rule 7: every decision logged).
    router_decision: Mapped[str] = mapped_column(String(16), nullable=False)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JsonColumn, nullable=False)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship(back_populates="messages")


class SpendLog(Base):
    """One row per paid provider call (Design.md §2.2, Prompt.md Rule 8).

    `tokens` is the provider-reported usage when available and a tiktoken
    estimate otherwise; `estimated_cost_usd` uses the config-declared list
    price, deliberately conservative. The daily/monthly caps in the budget
    guard are checked against these rows before a call is made.
    """

    __tablename__ = "spend_log"

    spend_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # llm | embedding | search
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
