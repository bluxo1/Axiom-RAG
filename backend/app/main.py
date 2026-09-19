"""FastAPI application factory.

Phase 0 wires the skeleton: configuration, structured errors, rate limiting,
CORS, and the liveness probe. Every later phase adds routers here, never
another app.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.health import HealthResponse, read_health
from app.api.v1.router import api_router
from app.config import AxiomConfig, get_config
from app.core.errors import register_exception_handlers
from app.core.logging_setup import configure_logging
from app.core.rate_limit import RateLimitMiddleware, TokenBucketLimiter
from app.core.security_headers import SecurityHeadersMiddleware
from app.db.session import Database
from app.services.runtime import Runtime

logger = logging.getLogger(__name__)

DESCRIPTION = """\
Citation-grounded RAG agent. Start from what you can prove.

Every claim is tied to a retrieved, verified source; low-confidence answers are
visibly flagged; when retrieval fails, Axiom falls back to live web search
instead of guessing.
"""


def create_app(config: AxiomConfig | None = None, runtime: Runtime | None = None) -> FastAPI:
    """Build the application.

    Pass `config` to override the process default; pass `runtime` to inject a
    database and fake providers (tests do this). Otherwise a `Runtime` is built
    from config with a lazily-connecting database and the configured providers.
    """
    config = config if config is not None else get_config()
    configure_logging(config.settings.log_level)

    # Whoever creates the database disposes it: when the caller injects a
    # runtime (tests), they own its lifecycle; when we build one, we do.
    owns_database = runtime is None
    if runtime is None:
        runtime = Runtime(config, Database(config.settings.database_url))
    active_runtime = runtime

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Create tables on startup. A schema tool (Alembic) replaces this once a
        # migration must run against a database holding real data. Kept
        # non-fatal so liveness still answers when the database is down — the
        # first /documents or /chat call then returns a clear error.
        try:
            active_runtime.db.create_all()
        except Exception:
            logger.exception("could not initialize the database schema at startup")
        yield
        if owns_database:
            active_runtime.db.dispose()

    app = FastAPI(
        title=config.app.name,
        description=DESCRIPTION,
        version=__version__,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.runtime = runtime

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
    # Added last => outermost => runs on every response (including 429s and CORS
    # preflights), so the baseline security headers are always present.
    app.add_middleware(
        SecurityHeadersMiddleware,
        enable_hsts=config.settings.enable_hsts,
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
