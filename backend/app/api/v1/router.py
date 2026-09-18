"""Aggregate router for `/api/v1`.

Phase 1 adds `/documents`, `/chat`, and `/sessions/{id}` alongside liveness.
`/metrics` arrives in Phase 3 with the router-decision logging (Phases.md).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import chat, documents, health, search, sessions

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(chat.router)
api_router.include_router(sessions.router)
