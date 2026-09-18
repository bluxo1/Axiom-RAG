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
from tests.helpers import BACKEND_ENV_VARS, CONFIG_PATH, read_raw_config


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
def client(config: AxiomConfig) -> Iterator[TestClient]:
    with TestClient(create_app(config)) as test_client:
        yield test_client
