"""Rate limiting (PRD.md §8 security NFR, Design.md §6).

Two layers are tested: the bucket arithmetic in isolation (with a fake clock,
so refill is deterministic) and the middleware's HTTP behaviour — a structured
429 with `Retry-After`, never a bare error.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.config import AxiomConfig, Knobs, RateLimitSection, Settings
from app.core.rate_limit import SESSION_HEADER, TokenBucketLimiter, default_identity
from app.main import create_app
from tests.helpers import make_runtime

HEALTH = "/api/v1/health"

# ─── Bucket arithmetic ────────────────────────────────────────────────────────


class FakeClock:
    """Monotonic clock under test control."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_burst_is_capped_at_capacity() -> None:
    limiter = TokenBucketLimiter(
        capacity=3, refill_per_second=1.0, max_identities=10, clock=FakeClock()
    )

    allowed = [limiter.check("a").allowed for _ in range(4)]

    assert allowed == [True, True, True, False]


def test_tokens_refill_over_time() -> None:
    clock = FakeClock()
    limiter = TokenBucketLimiter(capacity=2, refill_per_second=1.0, max_identities=10, clock=clock)
    limiter.check("a")
    limiter.check("a")
    assert limiter.check("a").allowed is False

    clock.advance(1.0)

    assert limiter.check("a").allowed is True


def test_refill_never_exceeds_capacity() -> None:
    clock = FakeClock()
    limiter = TokenBucketLimiter(capacity=2, refill_per_second=1.0, max_identities=10, clock=clock)
    limiter.check("a")
    clock.advance(3600.0)

    allowed = [limiter.check("a").allowed for _ in range(3)]

    assert allowed == [True, True, False]


def test_retry_after_reflects_the_refill_rate() -> None:
    limiter = TokenBucketLimiter(
        capacity=1, refill_per_second=0.5, max_identities=10, clock=FakeClock()
    )
    limiter.check("a")

    decision = limiter.check("a")

    assert decision.allowed is False
    assert decision.retry_after_seconds == pytest.approx(2.0)


def test_identities_have_separate_budgets() -> None:
    limiter = TokenBucketLimiter(
        capacity=1, refill_per_second=1.0, max_identities=10, clock=FakeClock()
    )

    assert limiter.check("a").allowed is True
    assert limiter.check("a").allowed is False
    assert limiter.check("b").allowed is True


def test_bucket_store_stays_bounded() -> None:
    """An unbounded identity map is a memory leak on a public endpoint."""
    clock = FakeClock()
    limiter = TokenBucketLimiter(capacity=1, refill_per_second=1.0, max_identities=20, clock=clock)

    for index in range(200):
        clock.advance(0.001)
        limiter.check(f"identity-{index}")

    assert limiter.tracked_identities <= 20


@pytest.mark.parametrize(
    ("capacity", "refill", "identities"),
    [(0, 1.0, 10), (-1, 1.0, 10), (1, 0, 10), (1, -1.0, 10), (1, 1.0, 0)],
)
def test_invalid_limiter_parameters_are_rejected(
    capacity: float, refill: float, identities: int
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TokenBucketLimiter(capacity=capacity, refill_per_second=refill, max_identities=identities)


# ─── Identity ─────────────────────────────────────────────────────────────────


def test_session_header_becomes_the_identity() -> None:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(SESSION_HEADER.lower().encode(), b"s-1")],
        "client": ("10.0.0.1", 1234),
    }

    assert default_identity(Request(scope)) == "session:s-1"


def test_client_address_is_the_fallback_identity() -> None:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "client": ("10.0.0.1", 1234),
    }

    assert default_identity(Request(scope)) == "ip:10.0.0.1"


def test_identity_of_last_resort() -> None:
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}

    assert default_identity(Request(scope)) == "anonymous"


# ─── Middleware behaviour ─────────────────────────────────────────────────────

TIGHT_LIMIT = RateLimitSection(
    enabled=True,
    capacity=2,
    refill_per_second=0.0001,
    exempt_paths=("/health",),
    max_tracked_identities=100,
)


def _app_with_limit(knobs: Knobs, section: RateLimitSection) -> FastAPI:
    config = AxiomConfig(Settings(), knobs.model_copy(update={"rate_limit": section}))
    return create_app(config, runtime=make_runtime(config))


@pytest.fixture
def tight_client(knobs: Knobs) -> Iterator[TestClient]:
    """A client with a two-request burst and effectively no refill."""
    app = _app_with_limit(knobs, TIGHT_LIMIT)
    with TestClient(app) as test_client:
        yield test_client
    app.state.runtime.db.dispose()


def test_over_budget_requests_get_a_structured_429(tight_client: TestClient) -> None:
    assert tight_client.get(HEALTH).status_code == 200
    assert tight_client.get(HEALTH).status_code == 200

    response = tight_client.get(HEALTH)

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMITED"
    assert int(response.headers["Retry-After"]) >= 1


def test_allowed_responses_carry_budget_headers(tight_client: TestClient) -> None:
    response = tight_client.get(HEALTH)

    assert response.headers["X-RateLimit-Limit"] == "2"
    assert response.headers["X-RateLimit-Remaining"] == "1"


def test_exempt_paths_are_never_limited(tight_client: TestClient) -> None:
    """A limited liveness probe would make orchestrators restart a healthy app."""
    statuses = [tight_client.get("/health").status_code for _ in range(10)]

    assert statuses == [200] * 10


def test_separate_sessions_get_separate_budgets(tight_client: TestClient) -> None:
    for _ in range(3):
        tight_client.get(HEALTH, headers={SESSION_HEADER: "exhausted"})

    response = tight_client.get(HEALTH, headers={SESSION_HEADER: "fresh"})

    assert response.status_code == 200


def test_limiter_can_be_disabled(knobs: Knobs) -> None:
    section = RateLimitSection(
        enabled=False,
        capacity=1,
        refill_per_second=0.0001,
        exempt_paths=(),
        max_tracked_identities=10,
    )

    app = _app_with_limit(knobs, section)
    with TestClient(app) as test_client:
        statuses = [test_client.get(HEALTH).status_code for _ in range(5)]
    app.state.runtime.db.dispose()

    assert statuses == [200] * 5
