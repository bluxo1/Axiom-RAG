"""Budget guard (Prompt.md Rule 8, Design.md §6).

Every paid provider call — LLM, embeddings, and (from Phase 3) web search —
passes through this guard:

1. **Pre-flight cap check** against the `spend_log` rows already persisted:
   the daily token cap and the monthly USD cap. Exceeding either raises
   `429 BUDGET_EXCEEDED` *before* the call is made — never a silent surprise
   bill.
2. **Per-request token limit** on the text about to be sent (prompt + context).
   A request over `max_request_tokens` is rejected up front instead of
   spending on a call that cannot ground anything.
3. **Post-call accounting**: the call's token usage is appended to `spend_log`
   with a cost estimate from the config-declared list price.

The counters live in Postgres, not process memory, so the caps hold across
workers and restarts (Design.md §2.2: "per-call spend counters... daily/monthly
caps checked against this table").

Error model: `BUDGET_EXCEEDED` is a 429 (retry later — the window resets),
deliberately distinct from `RATE_LIMITED` (per-session request rate, enforced
by the rate-limit middleware).

Token counts here are *estimates* with a fixed encoding (`cl100k_base`), used
only for cap enforcement — chunking keeps its own model-specific tokenizer.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from uuid import uuid4

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.sql.selectable import Select
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from tiktoken import get_encoding
from tiktoken.core import Encoding

from app.config import AxiomConfig
from app.core.errors import AxiomError, ErrorCode
from app.db.models import SpendLog
from app.db.session import Database


@lru_cache(maxsize=1)
def _encoding() -> Encoding:
    """The fixed encoding used for all spend estimates."""
    return get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Estimated token count of `text` (fixed encoding; not model-exact)."""
    return len(_encoding().encode(text))


def _month_start_utc(now: datetime) -> datetime:
    """UTC timestamp of the first instant of the current month."""
    return datetime(now.year, now.month, 1, tzinfo=UTC)


def _day_start_utc(now: datetime) -> datetime:
    """UTC timestamp of midnight of the current day."""
    return datetime(now.year, now.month, now.day, tzinfo=UTC)


def _tomorrow_start_utc(now: datetime) -> datetime:
    """UTC timestamp of midnight of the next day (when a daily cap resets)."""
    year, month, day = now.year, now.month, now.day
    if day == calendar.monthrange(year, month)[1]:
        if month == 12:
            return datetime(year + 1, 1, 1, tzinfo=UTC)
        return datetime(year, month + 1, 1, tzinfo=UTC)
    return datetime(year, month, day + 1, tzinfo=UTC)


@dataclass(frozen=True)
class Usage:
    """Usage for one provider call.

    `tokens` feeds the daily token cap. `cost_usd`, when set, overrides the
    token-derived cost estimate — used for per-call-priced calls like web
    search, which is not billed per token.
    """

    tokens: int
    cost_usd: float | None = None


@dataclass(frozen=True)
class SpendSummaryData:
    """Caps and usage — the Rule 8 slice of `GET /metrics`."""

    caps: dict[str, float | int]
    today: dict[str, float | int]
    this_month: dict[str, float | int]


class BudgetExceededError(AxiomError):
    """A configured spend cap would be crossed; the call is refused (429)."""

    def __init__(self, message: str, *, cap: str, details: dict[str, object]) -> None:
        super().__init__(
            ErrorCode.BUDGET_EXCEEDED,
            message,
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            details={"cap": cap, **details},
        )


class BudgetGuard:
    """Enforces Rule 8 caps against the `spend_log` table."""

    def __init__(self, config: AxiomConfig, database: Database) -> None:
        self._config = config
        self._db = database

    # ── pre-flight checks ─────────────────────────────────────────────────────

    def check_request(self, *, texts: dict[str, str]) -> None:
        """Reject the request up front if it exceeds the per-request token cap.

        `texts` maps a label (e.g. "prompt", "context") to the text that would
        be sent; labels appear in the rejection details so an operator can see
        what blew the budget.
        """
        limit = self._config.budget.max_request_tokens
        counts = {label: count_tokens(text) for label, text in texts.items()}
        total = sum(counts.values())
        if total > limit:
            raise BudgetExceededError(
                f"request needs {total} tokens, over the {limit}-token per-request cap.",
                cap="max_request_tokens",
                details={"requested_tokens": total, "limit": limit, "parts": counts},
            )

    def check_caps(self) -> None:
        """Reject the call if a daily token cap or monthly USD cap is exceeded.

        Checked against persisted `spend_log` rows, so the guard holds across
        workers and restarts.
        """
        budget = self._config.budget
        now = datetime.now(UTC)

        day_spend = self._aggregate(SpendLog.created_at >= _day_start_utc(now))
        if day_spend.tokens >= budget.daily_token_cap:
            raise BudgetExceededError(
                f"daily token cap reached ({budget.daily_token_cap}); resets at UTC midnight.",
                cap="daily_token_cap",
                details={
                    "limit": budget.daily_token_cap,
                    "used_tokens": day_spend.tokens,
                    "resets_at": _tomorrow_start_utc(now).isoformat(),
                },
            )

        month_spend = self._aggregate(SpendLog.created_at >= _month_start_utc(now))
        if month_spend.cost_usd >= budget.monthly_spend_usd:
            raise BudgetExceededError(
                f"monthly spend cap reached (${budget.monthly_spend_usd:.2f}).",
                cap="monthly_spend_usd",
                details={
                    "limit_usd": budget.monthly_spend_usd,
                    "used_usd": round(month_spend.cost_usd, 4),
                },
            )

    def check_batch(self, *, tokens: int) -> None:
        """Reject a multi-text call (e.g. an ingestion embed batch) that would
        push today's usage past the daily token cap.

        Distinct from `check_caps`: this accounts for the size of the call
        about to be made, so one huge document cannot consume (or exceed) an
        entire day's budget in a single request. The monthly USD cap is
        checked as in `check_caps`.
        """
        budget = self._config.budget
        now = datetime.now(UTC)

        day_spend = self._aggregate(SpendLog.created_at >= _day_start_utc(now))
        if day_spend.tokens + tokens > budget.daily_token_cap:
            headroom = max(0, budget.daily_token_cap - day_spend.tokens)
            raise BudgetExceededError(
                f"call needs {tokens} tokens; only {headroom} remain under the "
                f"{budget.daily_token_cap}-token daily cap.",
                cap="daily_token_cap",
                details={
                    "limit": budget.daily_token_cap,
                    "used_tokens": day_spend.tokens,
                    "requested_tokens": tokens,
                    "resets_at": _tomorrow_start_utc(now).isoformat(),
                },
            )

        month_spend = self._aggregate(SpendLog.created_at >= _month_start_utc(now))
        if month_spend.cost_usd >= budget.monthly_spend_usd:
            raise BudgetExceededError(
                f"monthly spend cap reached (${budget.monthly_spend_usd:.2f}).",
                cap="monthly_spend_usd",
                details={
                    "limit_usd": budget.monthly_spend_usd,
                    "used_usd": round(month_spend.cost_usd, 4),
                },
            )

    # ── post-call accounting ──────────────────────────────────────────────────────

    def record(self, *, provider: str, kind: str, model: str, usage: Usage) -> None:
        """Append one `spend_log` row for a completed (or billed) call."""
        if usage.cost_usd is not None:
            cost = usage.cost_usd
        else:
            cost = usage.tokens / 1_000_000 * self._price_for(kind)
        with self._db.session() as session:
            session.add(
                SpendLog(
                    spend_id=str(uuid4()),
                    provider=provider,
                    kind=kind,
                    model=model,
                    tokens=usage.tokens,
                    estimated_cost_usd=cost,
                )
            )

    # ── read-side aggregation (also backs GET /metrics) ───────────────────────

    def spend_summary(self) -> SpendSummaryData:
        """Caps and usage totals for today / this month (`GET /metrics`)."""
        budget = self._config.budget
        now = datetime.now(UTC)
        day = self._aggregate(SpendLog.created_at >= _day_start_utc(now))
        month = self._aggregate(SpendLog.created_at >= _month_start_utc(now))
        return SpendSummaryData(
            caps={
                "max_request_tokens": budget.max_request_tokens,
                "daily_token_cap": budget.daily_token_cap,
                "monthly_spend_usd": budget.monthly_spend_usd,
            },
            today={"tokens": day.tokens, "estimated_cost_usd": round(day.cost_usd, 6)},
            this_month={
                "tokens": month.tokens,
                "estimated_cost_usd": round(month.cost_usd, 6),
            },
        )

    @dataclass(frozen=True)
    class _Aggregate:
        tokens: int
        cost_usd: float

    def _aggregate(self, *filters: ColumnElement[bool]) -> _Aggregate:
        """Sum tokens and estimated cost over `spend_log` rows matching filters."""
        stmt: Select[tuple[int, float]] = select(
            func.coalesce(func.sum(SpendLog.tokens), 0).label("tokens"),
            func.coalesce(func.sum(SpendLog.estimated_cost_usd), 0).label("cost"),
        )
        for condition in filters:
            stmt = stmt.where(condition)
        with self._db.session() as session:
            tokens, cost = session.execute(stmt).one()
        return self._Aggregate(tokens=int(tokens), cost_usd=float(cost))

    def _price_for(self, kind: str) -> float:
        """Per-1M-token estimate for a call kind (Rule 8: capped, not assumed)."""
        if kind == "embedding":
            return self._config.embedding.price_per_million_usd
        return self._config.generation.price_per_million_usd
