"""Aggregate router for `/api/v1`.

Phase 0 ships liveness only. `/documents` and `/chat` arrive in Phase 1,
`/sessions/{id}` in Phase 1, `/metrics` in Phase 3 (Phases.md).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import health

api_router = APIRouter()
api_router.include_router(health.router)
