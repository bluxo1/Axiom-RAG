"""Baseline security response headers (PRD.md §8 security NFR).

Part of the Phase 4 hardening sweep for a public deploy: every response carries
a small set of conservative headers so a browser cannot be nudged into
MIME-sniffing, framing the API, or leaking the full referrer cross-origin. The
API serves JSON (and a Swagger UI at `/docs`), so the policy is deliberately
strict and static — no per-route tuning, no unsafe-inline scripts.

HSTS is emitted only when enabled (it is meaningful only over TLS, and would
wrongly pin plain-HTTP dev origins), so it defaults off and the deploy turns it
on behind the TLS-terminating proxy.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

# Static, conservative defaults. `frame-ancestors 'none'` and X-Frame-Options
# both refuse framing (old and new browsers); the CSP suits a JSON API plus the
# self-hosted Swagger UI, which needs inline styles but no inline scripts.
_DEFAULT_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
        "img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; connect-src 'self'"
    ),
}

_HSTS_HEADER = "Strict-Transport-Security"
_HSTS_VALUE = "max-age=63072000; includeSubDomains"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach baseline security headers to every response.

    Existing headers are never overwritten, so a handler that sets its own
    stricter policy keeps it.
    """

    def __init__(self, app: ASGIApp, *, enable_hsts: bool = False) -> None:
        super().__init__(app)
        self._headers = dict(_DEFAULT_HEADERS)
        if enable_hsts:
            self._headers[_HSTS_HEADER] = _HSTS_VALUE

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        for name, value in self._headers.items():
            response.headers.setdefault(name, value)
        return response
