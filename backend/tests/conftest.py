"""Shared fixtures.

Tests are hermetic: the ambient environment is cleared so a developer's local
`.env` or shell exports cannot change an assertion's outcome.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import AxiomConfig, Knobs, Settings, load_knobs
from app.main import create_app
from app.services.runtime import Runtime
from tests.helpers import BACKEND_ENV_VARS, CONFIG_PATH, make_runtime, read_raw_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove backend environment variables for the duration of a test."""
    for name in BACKEND_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def raw_config() -> dict[str, Any]:
    return read_raw_config()


@pytest.fixture
def knobs() -> Knobs:
    """The committed `config.yaml`, validated."""
    return load_knobs(CONFIG_PATH)


@pytest.fixture
def config(knobs: Knobs) -> AxiomConfig:
    return AxiomConfig(Settings(), knobs)


@pytest.fixture
def runtime(config: AxiomConfig) -> Iterator[Runtime]:
    """An offline runtime (SQLite + fake providers) for endpoint tests.

    Disposed after the test so the in-memory SQLite connection is closed rather
    than reclaimed by the garbage collector — an unclosed connection raises a
    ResourceWarning that `filterwarnings = ["error"]` would turn into a failure.
    """
    built = make_runtime(config)
    yield built
    built.db.dispose()


@pytest.fixture
def client(config: AxiomConfig, runtime: Runtime) -> Iterator[TestClient]:
    with TestClient(create_app(config, runtime=runtime)) as test_client:
        yield test_client
