"""Baseline security response headers (PRD.md §8, Phase 4 hardening sweep)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import AxiomConfig, Settings, load_knobs
from app.main import create_app
from app.services.runtime import Runtime
from tests.helpers import CONFIG_PATH, make_runtime

_EXPECTED = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def test_health_response_carries_security_headers(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    for name, value in _EXPECTED.items():
        assert response.headers[name] == value
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]


def test_hsts_is_absent_by_default(client: TestClient) -> None:
    # enable_hsts defaults off (plain-HTTP dev); the deploy turns it on.
    response = client.get("/health")
    assert "Strict-Transport-Security" not in response.headers


def test_hsts_is_emitted_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_HSTS", "true")
    config = AxiomConfig(Settings(), load_knobs(CONFIG_PATH))
    assert config.settings.enable_hsts is True

    runtime: Runtime = make_runtime(config)
    try:
        with TestClient(create_app(config, runtime=runtime)) as client:
            response = client.get("/health")
        assert response.headers["Strict-Transport-Security"].startswith("max-age=")
    finally:
        runtime.db.dispose()


def test_error_responses_also_carry_security_headers(client: TestClient) -> None:
    # A 404 goes through the exception handler; headers must still be attached.
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"
