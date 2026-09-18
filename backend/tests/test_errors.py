"""Structured errors.

Prompt.md §API CONTRACT: *structured errors only — never a silent 500.* An
unhandled exception leaking a framework-shaped body (or an empty one) is the
failure this suite exists to catch, because a client cannot branch on it and
the grounding rules become unenforceable from the outside.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import AxiomConfig
from app.core.errors import AxiomError, ErrorCode, ErrorResponse
from app.main import create_app
from app.services.runtime import Runtime

BOOM = "/api/v1/_boom"
TEAPOT = "/api/v1/_teapot"


@pytest.fixture
def faulty_app(config: AxiomConfig, runtime: Runtime) -> FastAPI:
    """An app with routes that fail on purpose."""
    app = create_app(config, runtime=runtime)

    @app.get(BOOM)
    def boom() -> None:
        raise RuntimeError("kaboom")

    @app.get(TEAPOT)
    def teapot() -> None:
        raise AxiomError(
            ErrorCode.VALIDATION_ERROR,
            "No documents ingested yet.",
            status_code=400,
            details={"cta": "upload a document"},
        )

    return app


def test_unhandled_exception_returns_the_envelope(faulty_app: FastAPI) -> None:
    with TestClient(faulty_app, raise_server_exceptions=False) as client:
        response = client.get(BOOM)

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "INTERNAL_ERROR", "message": "Internal server error."}
    }


def test_unhandled_exception_does_not_leak_internals(faulty_app: FastAPI) -> None:
    with TestClient(faulty_app, raise_server_exceptions=False) as client:
        response = client.get(BOOM)

    assert "kaboom" not in response.text
    assert "Traceback" not in response.text


def test_unhandled_exception_is_logged(
    faulty_app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """Rules.md §6 bans swallowing errors: the traceback must reach the log."""
    with TestClient(faulty_app, raise_server_exceptions=False) as client, caplog.at_level("ERROR"):
        client.get(BOOM)

    assert "unhandled_error" in caplog.text
    assert "kaboom" in caplog.text


def test_axiom_error_carries_its_code_and_details(faulty_app: FastAPI) -> None:
    with TestClient(faulty_app) as client:
        response = client.get(TEAPOT)

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "No documents ingested yet.",
            "details": {"cta": "upload a document"},
        }
    }


def test_unknown_route_returns_the_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_wrong_method_returns_the_envelope(client: TestClient) -> None:
    response = client.post("/api/v1/health")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_every_error_response_matches_the_schema(client: TestClient) -> None:
    """One envelope, validated by the model the API documents."""
    response = client.get("/api/v1/does-not-exist")

    parsed = ErrorResponse.model_validate(response.json())

    assert parsed.error.code is ErrorCode.NOT_FOUND
