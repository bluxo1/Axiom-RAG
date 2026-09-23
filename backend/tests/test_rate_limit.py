"""Rate limiting (PRD.md §8 security NFR, Design.md §6).

Two layers are tested: the bucket arithmetic in isolation (with a fake clock,
so refill is deterministic) and the middleware's HTTP behaviour — a structured
429 with `Retry-After`, never a bare error.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.config import AxiomConfig, Knobs, RateLimitSection, Settings
from app.core.rate_limit import TokenBucketLimiter, default_identity
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
    """Unseen clients share overflow capacity instead of evicting prior buckets."""
    clock = FakeClock()
    limiter = TokenBucketLimiter(capacity=1, refill_per_second=1.0, max_identities=20, clock=clock)

    for index in range(200):
        clock.advance(0.001)
        limiter.check(f"identity-{index}")

    assert limiter.tracked_identities <= 20
    assert limiter.check("a-new-identity").allowed is False


def test_overflow_bucket_is_bounded_when_limit_is_one() -> None:
    limiter = TokenBucketLimiter(
        capacity=1, refill_per_second=1.0, max_identities=1, clock=FakeClock()
    )

    assert limiter.check("first").allowed is True
    assert limiter.check("second").allowed is False
    assert limiter.tracked_identities == 1


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


def test_caller_supplied_session_header_does_not_change_identity() -> None:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"x-session-id", b"s-1")],
        "client": ("10.0.0.1", 1234),
    }

    assert default_identity(Request(scope)) == "ip:10.0.0.1"


def test_client_address_is_the_fallback_identity() -> None:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "client": ("10.0.0.1", 1234),
    }

    assert default_identity(Request(scope)) == "ip:10.0.0.1"


def test_forwarded_headers_are_ignored_outside_production() -> None:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (b"cf-connecting-ip", b"198.51.100.8"),
            (b"x-forwarded-for", b"203.0.113.9"),
        ],
        "client": ("10.0.0.1", 1234),
    }

    assert default_identity(Request(scope)) == "ip:10.0.0.1"


def test_production_identity_uses_valid_cloudflare_client_ip() -> None:
    config = SimpleNamespace(settings=SimpleNamespace(axiom_env="prod"))
    app = SimpleNamespace(state=SimpleNamespace(config=config))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (b"cf-connecting-ip", b"2001:db8::123"),
            (b"x-forwarded-for", b"198.51.100.8"),
        ],
        "client": ("10.0.0.1", 1234),
        "app": app,
    }

    assert default_identity(Request(scope)) == "ip:2001:db8::123"


def test_invalid_production_proxy_header_falls_back_to_peer() -> None:
    config = SimpleNamespace(settings=SimpleNamespace(axiom_env="prod"))
    app = SimpleNamespace(state=SimpleNamespace(config=config))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"cf-connecting-ip", b"198.51.100.8, 203.0.113.9")],
        "client": ("10.0.0.1", 1234),
        "app": app,
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
    config = AxiomConfig(Settings(_env_file=None), knobs.model_copy(update={"rate_limit": section}))
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


def test_session_header_cannot_bypass_the_peer_rate_limit(tight_client: TestClient) -> None:
    for _ in range(3):
        tight_client.get(HEALTH, headers={"X-Session-Id": "exhausted"})

    response = tight_client.get(HEALTH, headers={"X-Session-Id": "fresh"})

    assert response.status_code == 429


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
