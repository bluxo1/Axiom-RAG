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
from pydantic import SecretStr

from app.config import AxiomConfig, Knobs, Settings
from app.core.errors import AxiomError, ErrorCode, ErrorResponse
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder, OpenAIEmbedder
from app.llm.guarded import GuardedEmbedder, GuardedLLM
from app.llm.provider import OpenAILLM
from app.main import create_app
from app.services.runtime import Runtime
from tests.helpers import TEST_DATABASE_URL


def test_api_base_flows_into_runtime_builders(knobs: Knobs) -> None:
    """ADR-0003: an OpenAI-compatible endpoint reaches both provider clients.

    LlamaIndex client construction is offline, so the builders can be invoked
    here; the assertions prove `Runtime` passes `settings.openai_api_base`
    through instead of silently hardcoding api.openai.com.
    """
    base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    cfg = AxiomConfig(
        Settings(
            _env_file=None,
            openai_api_key=SecretStr("test-key"),
            openai_api_base=base,
        ),
        knobs,
    )
    runtime = Runtime(cfg, Database(TEST_DATABASE_URL))

    llm = runtime._build_llm()
    embedder = runtime._build_embedder()

    # Builders wrap in the Rule 8 budget guard; unwrap to the real provider.
    assert isinstance(llm, GuardedLLM)
    assert isinstance(embedder, GuardedEmbedder)
    inner_llm = llm._inner
    inner_embedder = embedder._inner
    assert isinstance(inner_llm, OpenAILLM)
    assert isinstance(inner_embedder, OpenAIEmbedder)
    assert str(inner_llm._client.api_base) == base
    assert str(inner_embedder._client.api_base) == base


BOOM = "/api/v1/_boom"
TEAPOT = "/api/v1/_teapot"
DOCS = "/api/v1/documents"


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


# ─── 503 PROVIDER_UNAVAILABLE (README documents it; it must be reachable) ─────


def _runtime_app(config: AxiomConfig) -> tuple[FastAPI, Database]:
    """An app whose providers build lazily from real config (no fakes)."""
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    return create_app(config, runtime=Runtime(config, database)), database


def test_missing_openai_key_returns_structured_503(config: AxiomConfig) -> None:
    """A missing key surfaces as 503 PROVIDER_UNAVAILABLE, never a silent 500."""
    app, database = _runtime_app(config)

    try:
        with TestClient(app) as client:
            response = client.post(DOCS, files={"file": ("kb.txt", b"content", "text/plain")})
    finally:
        database.dispose()

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "PROVIDER_UNAVAILABLE"
    assert "OPENAI_API_KEY" in body["error"]["message"]


def test_unreachable_vector_store_returns_structured_503(knobs: Knobs) -> None:
    """An unreachable Chroma is a 503 too — distinct from a bug (500).

    Settings are explicit (offline embedder injected; Chroma pointed at a
    port nothing listens on) so the test makes no network call beyond a
    refused localhost connection.
    """
    cfg = AxiomConfig(
        Settings(
            _env_file=None,
            openai_api_key=SecretStr("test-key"),
            chroma_host="127.0.0.1",
            chroma_port=1,
        ),
        knobs,
    )
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    app = create_app(cfg, runtime=Runtime(cfg, database, embedder=HashingEmbedder(64)))

    try:
        with TestClient(app) as client:
            response = client.post(DOCS, files={"file": ("kb.txt", b"content", "text/plain")})
    finally:
        database.dispose()

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "PROVIDER_UNAVAILABLE"
    assert "vector store" in body["error"]["message"]
