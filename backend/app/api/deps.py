"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends
from starlette.requests import Request

from app.config import AxiomConfig


def get_app_config(request: Request) -> AxiomConfig:
    """Return the configuration the app factory was built with.

    Sourced from `app.state` rather than reloaded, so a test (or a future
    multi-tenant setup) that builds an app with a specific config gets exactly
    that config in handlers — no second read of `config.yaml`.
    """
    return cast(AxiomConfig, request.app.state.config)


ConfigDep = Annotated[AxiomConfig, Depends(get_app_config)]
