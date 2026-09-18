"""Metrics endpoint (Design.md §1.3).

Two families of numbers for the demo dashboard:

* **routing** — flagged-rate and fallback-rate over every persisted turn
  (Rule 7: every routing decision logged), computed from the `messages`
  router-decision columns.
* **spend** — the Rule 8 caps and today/this-month usage, aggregated from
  `spend_log` by the budget guard.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import RuntimeDep
from app.core.budget import SpendSummaryData
from app.services.metrics import RoutingMetricsData, routing_metrics

router = APIRouter(tags=["meta"])


class RoutingSummary(BaseModel):
    """Answer-routing rates over all persisted turns (Design.md §1.3)."""

    total_messages: int
    flagged: int
    fallback_used: int
    refused: int
    flagged_rate: float
    fallback_rate: float
    refusal_rate: float

    @classmethod
    def from_data(cls, data: RoutingMetricsData) -> RoutingSummary:
        return cls(
            total_messages=data.total_messages,
            flagged=data.flagged,
            fallback_used=data.fallback_used,
            refused=data.refused,
            flagged_rate=data.flagged_rate,
            fallback_rate=data.fallback_rate,
            refusal_rate=data.refusal_rate,
        )


class SpendSummary(BaseModel):
    """Rule 8 caps and usage (Design.md §1.3, Prompt.md Rule 8)."""

    caps: dict[str, float | int]
    today: dict[str, float | int]
    this_month: dict[str, float | int]

    @classmethod
    def from_data(cls, data: SpendSummaryData) -> SpendSummary:
        return cls(caps=data.caps, today=data.today, this_month=data.this_month)


class MetricsResponse(BaseModel):
    routing: RoutingSummary
    spend: SpendSummary


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Routing rates (Rule 7) and spend counters (Rule 8)",
)
def metrics(runtime: RuntimeDep) -> MetricsResponse:
    return MetricsResponse(
        routing=RoutingSummary.from_data(routing_metrics(runtime)),
        spend=SpendSummary.from_data(runtime.budget.spend_summary()),
    )
