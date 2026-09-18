"""Liveness probe (Design.md §1.3).

Deliberately dependency-free: `/health` answers "this process is up and its
configuration loaded", not "Postgres and Chroma are reachable". Container
orchestrators restart on a failed liveness probe, and restarting the API does
not fix a downed database. Datastore readiness is reported by `GET /metrics`
when Phase 3 adds it.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.api.deps import ConfigDep
from app.config import Environment

router = APIRouter(tags=["meta"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: Environment


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def read_health(config: ConfigDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=config.app.name,
        version=__version__,
        environment=config.settings.axiom_env,
    )
