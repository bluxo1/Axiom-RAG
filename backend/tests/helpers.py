"""Test helpers: repository paths and config fixtures on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Every environment variable the backend reads (see `.env.example`).
BACKEND_ENV_VARS = (
    "AXIOM_ENV",
    "LOG_LEVEL",
    "CONFIG_PATH",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "DATABASE_URL",
    "CHROMA_HOST",
    "CHROMA_PORT",
    "MAX_REQUEST_TOKENS",
    "DAILY_TOKEN_CAP",
    "MONTHLY_SPEND_USD",
)


def read_raw_config() -> dict[str, Any]:
    """The committed `config.yaml` as plain data, for mutation in tests."""
    parsed = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"{CONFIG_PATH} must contain a mapping")
    return parsed


def write_config(tmp_path: Path, raw: dict[str, Any]) -> Path:
    """Write a config mapping to a temp file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path
