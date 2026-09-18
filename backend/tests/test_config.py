"""Configuration loading.

Phase 0's worst failure mode is a silently wrong knob: a mistyped key falling
back to a default, weights that do not sum to 1, or thresholds in the wrong
order. Any of those corrupts confidence routing later without an obvious
symptom, so every one of them must fail loudly at load time.

The documented values (Phases.md Phase 0: thresholds 0.75/0.45, k=8, chunk
512/64) are asserted directly, so drift between `config.yaml` and the spec
fails the build.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from app.config import (
    AxiomConfig,
    ConfigError,
    Knobs,
    Settings,
    load_knobs,
    resolve_config_path,
)
from tests.helpers import CONFIG_PATH, write_config

# ─── Documented values (Phases.md Phase 0) ────────────────────────────────────


def test_committed_config_loads(knobs: Knobs) -> None:
    assert knobs.app.name == "Axiom"
    assert knobs.app.api_prefix == "/api/v1"


def test_confidence_thresholds_match_the_spec(knobs: Knobs) -> None:
    assert knobs.confidence.thresholds.high == 0.75
    assert knobs.confidence.thresholds.low == 0.45


def test_confidence_weights_sum_to_one(knobs: Knobs) -> None:
    weights = knobs.confidence.weights
    assert weights.retrieval + weights.faithfulness + weights.coverage == pytest.approx(1.0)


def test_retrieval_k_matches_the_spec(knobs: Knobs) -> None:
    assert knobs.retrieval.top_k == 8


def test_chunking_matches_the_spec(knobs: Knobs) -> None:
    assert knobs.chunking.size_tokens == 512
    assert knobs.chunking.overlap_tokens == 64


def test_budget_caps_are_present(knobs: Knobs) -> None:
    """Rule 8: spend is capped, not assumed."""
    assert knobs.budget.max_request_tokens > 0
    assert knobs.budget.daily_token_cap > 0
    assert knobs.budget.monthly_spend_usd > 0


def test_rate_limiting_is_enabled_with_health_exempt(knobs: Knobs) -> None:
    """PRD.md §8 wants rate limits; orchestrators need liveness to stay open."""
    assert knobs.rate_limit.enabled is True
    assert "/health" in knobs.rate_limit.exempt_paths


def test_vector_store_pins_cosine_similarity(knobs: Knobs) -> None:
    """ADR-0002: the metric is pinned so dev and prod scores are comparable."""
    assert knobs.vector_store.similarity == "cosine"


def test_collection_is_namespaced_by_embedding_model(config: AxiomConfig) -> None:
    """ADR-0001: a model switch must not mix incompatible vectors."""
    assert config.embedding.slug == "text_embedding_3_small"
    assert config.collection_name == "chunks_text_embedding_3_small"


# ─── Invalid configuration must fail loudly ───────────────────────────────────


def test_unknown_key_is_rejected(tmp_path: Path, raw_config: dict[str, Any]) -> None:
    """A typo must not silently fall back to a default."""
    raw_config["retrieval"]["top_kk"] = 8
    path = write_config(tmp_path, raw_config)

    with pytest.raises(ConfigError, match="invalid configuration"):
        load_knobs(path)


def test_missing_section_is_rejected(tmp_path: Path, raw_config: dict[str, Any]) -> None:
    del raw_config["confidence"]
    path = write_config(tmp_path, raw_config)

    with pytest.raises(ConfigError, match="invalid configuration"):
        load_knobs(path)


def test_weights_that_do_not_sum_to_one_are_rejected(
    tmp_path: Path, raw_config: dict[str, Any]
) -> None:
    raw_config["confidence"]["weights"] = {
        "retrieval": 0.5,
        "faithfulness": 0.5,
        "coverage": 0.5,
    }
    path = write_config(tmp_path, raw_config)

    with pytest.raises(ConfigError, match=re.escape("must sum to 1.0")):
        load_knobs(path)


def test_inverted_thresholds_are_rejected(tmp_path: Path, raw_config: dict[str, Any]) -> None:
    raw_config["confidence"]["thresholds"] = {"high": 0.45, "low": 0.75}
    path = write_config(tmp_path, raw_config)

    with pytest.raises(ConfigError, match="low must be below high"):
        load_knobs(path)


def test_overlap_larger_than_chunk_is_rejected(tmp_path: Path, raw_config: dict[str, Any]) -> None:
    raw_config["chunking"] = {"size_tokens": 512, "overlap_tokens": 512}
    path = write_config(tmp_path, raw_config)

    with pytest.raises(ConfigError, match="smaller than size_tokens"):
        load_knobs(path)


def test_non_positive_top_k_is_rejected(tmp_path: Path, raw_config: dict[str, Any]) -> None:
    raw_config["retrieval"]["top_k"] = 0
    path = write_config(tmp_path, raw_config)

    with pytest.raises(ConfigError, match="invalid configuration"):
        load_knobs(path)


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("app: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid YAML"):
        load_knobs(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="could not read"):
        load_knobs(tmp_path / "absent.yaml")


def test_absolute_missing_path_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        resolve_config_path(tmp_path / "absent.yaml")


def test_relative_path_resolves_from_the_repository_root() -> None:
    """Works whether a process starts in `backend/` or at the repo root."""
    assert resolve_config_path(Path("config.yaml")) == CONFIG_PATH


# ─── Environment overrides (Prompt.md Rule 8, Design.md §6) ───────────────────


def test_budget_defaults_come_from_yaml(knobs: Knobs) -> None:
    config = AxiomConfig(Settings(), knobs)

    assert config.budget.max_request_tokens == knobs.budget.max_request_tokens
    assert config.budget.daily_token_cap == knobs.budget.daily_token_cap
    assert config.budget.monthly_spend_usd == knobs.budget.monthly_spend_usd


def test_env_overrides_budget_caps(knobs: Knobs, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_REQUEST_TOKENS", "123")
    monkeypatch.setenv("DAILY_TOKEN_CAP", "4567")
    monkeypatch.setenv("MONTHLY_SPEND_USD", "1.25")

    config = AxiomConfig(Settings(), knobs)

    assert config.budget.max_request_tokens == 123
    assert config.budget.daily_token_cap == 4567
    assert config.budget.monthly_spend_usd == 1.25


def test_blank_env_var_falls_back_to_yaml(knobs: Knobs, monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env.example` ships blank keys; blank must mean "not configured"."""
    monkeypatch.setenv("MAX_REQUEST_TOKENS", "")
    monkeypatch.setenv("OPENAI_API_KEY", "   ")

    config = AxiomConfig(Settings(), knobs)

    assert config.budget.max_request_tokens == knobs.budget.max_request_tokens
    assert config.settings.openai_api_key is None


def test_invalid_env_override_is_rejected(knobs: Knobs, monkeypatch: pytest.MonkeyPatch) -> None:
    """An env var must clear the same bar as a committed value."""
    monkeypatch.setenv("DAILY_TOKEN_CAP", "0")

    with pytest.raises(ValueError, match="daily_token_cap"):
        AxiomConfig(Settings(), knobs)


def test_secrets_are_not_exposed_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rules.md §2: no secrets in logs or tracebacks."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-log-me")

    settings = Settings()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-do-not-log-me"
    assert "sk-do-not-log-me" not in repr(settings)
