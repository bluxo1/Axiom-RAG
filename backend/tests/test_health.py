"""Liveness probe.

Phase 0's exit criterion is `/health` returning 200 (Phases.md), while
Design.md §1.3 places health under the `/api/v1` prefix. Both paths are served
and both are asserted here, so neither doc's contract can regress unnoticed.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import __version__


def test_unversioned_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Axiom",
        "version": __version__,
        "environment": "dev",
    }


def test_versioned_health_returns_200(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_the_config_the_app_was_built_with(client: TestClient) -> None:
    """Handlers read `app.state.config`, not a fresh load from disk."""
    assert client.get("/health").json()["service"] == "Axiom"


def test_openapi_schema_is_served(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/health" in response.json()["paths"]
