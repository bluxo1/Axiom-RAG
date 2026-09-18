"""Metrics endpoint (Design.md §1.3).

Phase 1 slice: the Rule 8 spend counters — caps, today's usage, this month's
usage — all aggregated from `spend_log` by the budget guard. The flagged-rate
and fallback-rate metrics arrive in Phase 3 with the router-decision columns
they are computed from.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.api.deps import RuntimeDep
from app.core.budget import SpendSummaryData

router = APIRouter(tags=["meta"])


class SpendSummary(BaseModel):
    """Rule 8 caps and usage (Design.md §1.3, Prompt.md Rule 8)."""

    caps: dict[str, float | int]
    today: dict[str, float | int]
    this_month: dict[str, float | int]

    @classmethod
    def from_data(cls, data: SpendSummaryData) -> SpendSummary:
        return cls(caps=data.caps, today=data.today, this_month=data.this_month)


class MetricsResponse(BaseModel):
    spend: SpendSummary


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Spend counters against configured caps (Rule 8)",
)
def metrics(runtime: RuntimeDep) -> MetricsResponse:
    return MetricsResponse(spend=SpendSummary.from_data(runtime.budget.spend_summary()))
