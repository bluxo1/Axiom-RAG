"""Aggregate router for `/api/v1`.

Phase 1 adds `/documents`, `/search`, `/chat`, `/sessions/{id}`, and `/metrics`
(spend counters only — flagged/fallback rates land in Phase 3) alongside
liveness.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat, documents, health, metrics, search, sessions

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
api_router.include_router(sessions.router)
api_router.include_router(metrics.router)
