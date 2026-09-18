"""Budget guard and /metrics (Prompt.md Rule 8, Design.md §1.3, §6).

Offline: the guard is exercised against SQLite-backed `spend_log` rows with
the standard offline runtime (CLAUDE.md: never live calls in CI).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import AxiomConfig
from app.core.budget import BudgetExceededError, BudgetGuard, Usage, count_tokens
from app.db.models import SpendLog
from app.services.runtime import Runtime

METRICS = "/api/v1/metrics"
CHAT = "/api/v1/chat"
DOCS = "/api/v1/documents"


def _spend_rows(runtime: Runtime) -> list[SpendLog]:
    with runtime.db.session() as session:
        return list(session.query(SpendLog).all())


def _seed(
    runtime: Runtime,
    *,
    spend_id: str,
    tokens: int,
    cost_usd: float = 0.0,
    created_at: datetime | None = None,
) -> None:
    with runtime.db.session() as session:
        session.add(
            SpendLog(
                spend_id=spend_id,
                provider="openai",
                kind="llm",
                model="seed-model",
                tokens=tokens,
                estimated_cost_usd=cost_usd,
                created_at=created_at,
            )
        )


def test_llm_call_records_spend(config: AxiomConfig, runtime: Runtime) -> None:
    runtime.llm.complete(system_prompt="answer only from context", user_prompt="hello")

    rows = _spend_rows(runtime)
    assert len(rows) == 1
    assert rows[0].kind == "llm"
    assert rows[0].model == config.generation.model
    assert rows[0].tokens > 0
    assert rows[0].estimated_cost_usd >= 0


def test_embedding_calls_record_spend(config: AxiomConfig, runtime: Runtime) -> None:
    runtime.embedder.embed_query("what is the policy?")
    runtime.embedder.embed_texts(["chunk one", "chunk two"])

    rows = _spend_rows(runtime)
    assert {row.kind for row in rows} == {"embedding"}
    assert sum(row.tokens for row in rows) > 0


def test_per_request_token_cap_is_enforced(config: AxiomConfig, runtime: Runtime) -> None:
    huge_prompt = "word " * (config.budget.max_request_tokens * 2)

    with pytest.raises(BudgetExceededError) as excinfo:
        runtime.llm.complete(system_prompt=huge_prompt, user_prompt="hello")

    assert (excinfo.value.details or {})["cap"] == "max_request_tokens"
    assert excinfo.value.status_code == 429
    # Nothing was spent: the call never happened.
    assert _spend_rows(runtime) == []


def test_daily_token_cap_blocks_the_next_call(config: AxiomConfig, runtime: Runtime) -> None:
    _seed(runtime, spend_id="seed-day", tokens=config.budget.daily_token_cap)

    with pytest.raises(BudgetExceededError) as excinfo:
        runtime.llm.complete(system_prompt="small", user_prompt="hello")

    assert (excinfo.value.details or {})["cap"] == "daily_token_cap"


def test_monthly_usd_cap_blocks_the_next_call(config: AxiomConfig, runtime: Runtime) -> None:
    _seed(
        runtime,
        spend_id="seed-month",
        tokens=1,
        cost_usd=config.budget.monthly_spend_usd,
    )

    with pytest.raises(BudgetExceededError) as excinfo:
        runtime.llm.complete(system_prompt="small", user_prompt="hello")

    assert (excinfo.value.details or {})["cap"] == "monthly_spend_usd"


def test_oversized_batch_is_refused(config: AxiomConfig, runtime: Runtime) -> None:
    big_text = "word " * (config.budget.daily_token_cap * 2)

    with pytest.raises(BudgetExceededError) as excinfo:
        runtime.embedder.embed_texts([big_text])

    assert (excinfo.value.details or {})["cap"] == "daily_token_cap"
    assert _spend_rows(runtime) == []


def test_batch_that_fits_the_daily_cap_passes(config: AxiomConfig, runtime: Runtime) -> None:
    vectors = runtime.embedder.embed_texts(["alpha", "beta"])

    assert len(vectors) == 2
    assert len(vectors[0]) == runtime.embedder.dimensions


def test_ingestion_failure_still_leaves_no_partial_spend(
    config: AxiomConfig, runtime: Runtime
) -> None:
    """A rejected batch must not record spend rows (only billed calls do)."""
    big_text = "word " * (config.budget.daily_token_cap * 2)

    with pytest.raises(BudgetExceededError):
        runtime.embedder.embed_texts([big_text])

    assert _spend_rows(runtime) == []


def test_chat_endpoint_returns_429_budget_exceeded(
    config: AxiomConfig, runtime: Runtime, client: TestClient
) -> None:
    upload = client.post(DOCS, files={"file": ("kb.txt", b"grounded content", "text/plain")})
    assert upload.status_code == 200

    _seed(runtime, spend_id="seed-day", tokens=config.budget.daily_token_cap)

    response = client.post(CHAT, json={"question": "anything?"})

    assert response.status_code == 429
    body = response.json()["error"]
    assert body["code"] == "BUDGET_EXCEEDED"
    assert body["details"]["cap"] == "daily_token_cap"


def test_metrics_reports_spend_counters(
    config: AxiomConfig, runtime: Runtime, client: TestClient
) -> None:
    runtime.embedder.embed_query("hello metrics")

    response = client.get(METRICS)

    assert response.status_code == 200
    spend = response.json()["spend"]
    assert spend["caps"]["daily_token_cap"] == config.budget.daily_token_cap
    assert spend["today"]["tokens"] > 0
    assert spend["this_month"]["tokens"] >= spend["today"]["tokens"]


def test_metrics_is_empty_when_nothing_spent(client: TestClient) -> None:
    response = client.get(METRICS)

    assert response.status_code == 200
    spend = response.json()["spend"]
    assert spend["today"]["tokens"] == 0
    assert spend["this_month"]["estimated_cost_usd"] == 0


def test_count_tokens_is_stable() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello world") == count_tokens("hello world") > 0


def test_aggregation_excludes_rows_outside_the_window(
    config: AxiomConfig, runtime: Runtime
) -> None:
    """Only rows from the current day/month count toward the caps."""
    guard = BudgetGuard(config, runtime.db)
    now = datetime.now(UTC)
    _seed(
        runtime,
        spend_id="old-row",
        tokens=1_000_000,
        cost_usd=99.0,
        created_at=now - timedelta(days=35),  # always a previous month
    )
    _seed(runtime, spend_id="today-row", tokens=10, cost_usd=0.01, created_at=now)

    summary = guard.spend_summary()

    assert summary.today["tokens"] == 10
    assert summary.this_month["tokens"] == 10
    assert summary.this_month["estimated_cost_usd"] == pytest.approx(0.01, abs=1e-6)


def test_llm_and_embedding_kinds_are_priced_separately(
    config: AxiomConfig, runtime: Runtime
) -> None:
    guard = BudgetGuard(config, runtime.db)
    if config.embedding.price_per_million_usd == config.generation.price_per_million_usd:
        pytest.skip("config declares identical prices")

    guard.record(provider="openai", kind="embedding", model="e", usage=Usage(tokens=1_000_000))
    guard.record(provider="openai", kind="llm", model="m", usage=Usage(tokens=1_000_000))

    rows = _spend_rows(runtime)
    costs = sorted(row.estimated_cost_usd for row in rows)
    assert costs[0] < costs[1]
