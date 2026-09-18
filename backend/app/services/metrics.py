"""Routing metrics (Design.md §1.3, Rule 7).

`GET /metrics` reports two rates over the persisted `messages` rows, alongside
the Rule 8 spend counters: how often an answer was **flagged** (shown with a
low-confidence warning) and how often the **web fallback** was used. Both are
computed from the router-decision columns every turn writes (Rule 7: every
routing decision is persisted), so the demo dashboard reads real history, not a
running in-memory tally.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import ColumnElement, Integer, cast, func, select
from sqlalchemy.sql import ColumnExpressionArgument

from app.db.models import Message
from app.services.runtime import Runtime


@dataclass(frozen=True)
class RoutingMetricsData:
    """Answer-routing rates over all persisted turns."""

    total_messages: int
    flagged: int
    fallback_used: int
    refused: int
    flagged_rate: float
    fallback_rate: float
    refusal_rate: float


def routing_metrics(runtime: Runtime) -> RoutingMetricsData:
    """Aggregate flagged / fallback / refusal counts and rates from `messages`."""
    stmt = select(
        func.count(Message.msg_id),
        func.coalesce(func.sum(_as_int(Message.flagged)), 0),
        func.coalesce(func.sum(_as_int(Message.fallback_used)), 0),
        func.coalesce(func.sum(_as_int(Message.router_decision == "refusal")), 0),
    )
    # `stmt` sums 0/1 casts of each predicate, so one pass yields every count.
    with runtime.db.session() as session:
        total, flagged, fallback_used, refused = session.execute(stmt).one()

    total = int(total)
    flagged = int(flagged)
    fallback_used = int(fallback_used)
    refused = int(refused)
    return RoutingMetricsData(
        total_messages=total,
        flagged=flagged,
        fallback_used=fallback_used,
        refused=refused,
        flagged_rate=_rate(flagged, total),
        fallback_rate=_rate(fallback_used, total),
        refusal_rate=_rate(refused, total),
    )


def _as_int(condition: ColumnExpressionArgument[bool]) -> ColumnElement[int]:
    """Cast a boolean column/predicate to 0/1 so SUM works on every dialect.

    Accepts a mapped boolean column or a comparison predicate — both are boolean
    SQL expressions (`ColumnExpressionArgument`).
    """
    return cast(condition, Integer)


def _rate(count: int, total: int) -> float:
    """A 0-1 rate, rounded; 0.0 over an empty corpus (no division by zero)."""
    return round(count / total, 4) if total else 0.0
