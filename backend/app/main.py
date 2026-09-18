"""FastAPI application factory.

Phase 0 wires the skeleton: configuration, structured errors, rate limiting,
CORS, and the liveness probe. Every later phase adds routers here, never
another app.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.health import HealthResponse, read_health
from app.api.v1.router import api_router
from app.config import AxiomConfig, get_config
from app.core.errors import register_exception_handlers
from app.core.logging_setup import configure_logging
from app.core.rate_limit import RateLimitMiddleware, TokenBucketLimiter

DESCRIPTION = """\
Citation-grounded RAG agent. Start from what you can prove.

Every claim is tied to a retrieved, verified source; low-confidence answers are
visibly flagged; when retrieval fails, Axiom falls back to live web search
instead of guessing.
"""


def create_app(config: AxiomConfig | None = None) -> FastAPI:
    """Build the application. Pass `config` to override the process default."""
    config = config if config is not None else get_config()
    configure_logging(config.settings.log_level)

    app = FastAPI(
        title=config.app.name,
        description=DESCRIPTION,
        version=__version__,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.config = config

    register_exception_handlers(app)

    # Middleware added later runs earlier: CORS must wrap the rate limiter so a
    # browser can read a 429 instead of seeing an opaque CORS failure.
    if config.rate_limit.enabled:
        app.add_middleware(
            RateLimitMiddleware,
            limiter=TokenBucketLimiter(
                capacity=config.rate_limit.capacity,
                refill_per_second=config.rate_limit.refill_per_second,
                max_identities=config.rate_limit.max_tracked_identities,
            ),
            exempt_paths=config.rate_limit.exempt_paths,
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.app.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    app.include_router(api_router, prefix=config.app.api_prefix)

    # Unversioned liveness alias. The versioned path in Design.md §1.3 is the
    # contract; this alias is what Phases.md Phase 0 exits on and what the
    # container healthcheck and load balancers probe.
    app.add_api_route(
        "/health",
        read_health,
        methods=["GET"],
        response_model=HealthResponse,
        tags=["meta"],
        summary="Liveness probe (unversioned alias)",
        include_in_schema=False,
    )

    return app


app = create_app()
