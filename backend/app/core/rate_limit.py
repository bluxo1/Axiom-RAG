"""Per-session rate limiting (PRD.md §8 security NFR, Design.md §6).

A token bucket per identity: `capacity` requests available at once, refilled at
`refill_per_second`. Dev is generous; the Phase 4 hardening sweep tightens prod.

Scope of this implementation: the bucket store is in-process, so limits are
per-worker. That is deliberate for Phase 0 — a shared store (Redis) is part of
the Phase 4 sweep, not the skeleton.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import HTTP_429_TOO_MANY_REQUESTS
from starlette.types import ASGIApp

from app.core.errors import ErrorCode, error_response

SESSION_HEADER = "X-Session-Id"


@dataclass(frozen=True)
class RateLimitDecision:
    """Outcome of one bucket check."""

    allowed: bool
    remaining: float
    retry_after_seconds: float


@dataclass
class _Bucket:
    tokens: float
    last_seen: float


class TokenBucketLimiter:
    """Thread-safe in-process token bucket, keyed by caller identity."""

    def __init__(
        self,
        *,
        capacity: float,
        refill_per_second: float,
        max_identities: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be positive")
        if max_identities <= 0:
            raise ValueError("max_identities must be positive")

        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._max_identities = max_identities
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def tracked_identities(self) -> int:
        """Number of live buckets. Bounded by `max_identities`."""
        with self._lock:
            return len(self._buckets)

    def check(self, identity: str) -> RateLimitDecision:
        """Consume one token for `identity`, refilling for elapsed time first."""
        with self._lock:
            now = self._clock()
            bucket = self._buckets.get(identity)
            if bucket is None:
                self._evict_if_full()
                bucket = _Bucket(tokens=self._capacity, last_seen=now)
                self._buckets[identity] = bucket
            else:
                elapsed = max(0.0, now - bucket.last_seen)
                bucket.tokens = min(
                    self._capacity, bucket.tokens + elapsed * self._refill_per_second
                )
            bucket.last_seen = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return RateLimitDecision(
                    allowed=True, remaining=bucket.tokens, retry_after_seconds=0.0
                )

            deficit = 1.0 - bucket.tokens
            return RateLimitDecision(
                allowed=False,
                remaining=bucket.tokens,
                retry_after_seconds=deficit / self._refill_per_second,
            )

    def _evict_if_full(self) -> None:
        """Drop the idlest buckets so the store stays bounded.

        Called with the lock held. Evicting a bucket only ever *grants* budget,
        never revokes it, so a dropped caller is not penalised.
        """
        if len(self._buckets) < self._max_identities:
            return
        target = max(1, self._max_identities // 10)
        idlest = sorted(self._buckets.items(), key=lambda item: item[1].last_seen)
        for identity, _ in idlest[:target]:
            del self._buckets[identity]


def default_identity(request: Request) -> str:
    """Identify the caller: session header first, client address otherwise.

    Design.md §6 specifies a per-session bucket. Sessions arrive from Phase 1
    (`POST /chat {session_id}`); until then, and for any request without the
    header, the peer address is the identity.
    """
    session_id = request.headers.get(SESSION_HEADER, "").strip()
    if session_id:
        return f"session:{session_id}"
    client = request.client
    if client is not None and client.host:
        return f"ip:{client.host}"
    return "anonymous"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests over budget with a structured 429."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: TokenBucketLimiter,
        exempt_paths: Iterable[str] = (),
        identify: Callable[[Request], str] = default_identity,
    ) -> None:
        super().__init__(app)
        self._limiter = limiter
        self._exempt_paths = frozenset(exempt_paths)
        self._identify = identify

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        decision = self._limiter.check(self._identify(request))
        limit = str(math.floor(self._limiter.capacity))

        if not decision.allowed:
            retry_after = max(1, math.ceil(decision.retry_after_seconds))
            return error_response(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                code=ErrorCode.RATE_LIMITED,
                message="Too many requests. Slow down and retry.",
                details={"retry_after_seconds": retry_after},
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": limit,
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = limit
        response.headers["X-RateLimit-Remaining"] = str(math.floor(decision.remaining))
        return response
