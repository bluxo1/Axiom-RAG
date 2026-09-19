"""Test helpers: repository paths, config fixtures, and an offline runtime."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from app.config import AxiomConfig
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder
from app.llm.provider import INSUFFICIENT_EVIDENCE_JSON, ScriptedLLM
from app.search.provider import ScriptedWebSearch, WebResult
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"

# In-memory SQLite (Design.md §5 tech table: SQLite is the relational dev
# alternative). Keeps tests offline — no Postgres, no Chroma, no API keys.
TEST_DATABASE_URL = "sqlite+pysqlite:///:memory:"

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
    "ENABLE_HSTS",
    "MAX_REQUEST_TOKENS",
    "DAILY_TOKEN_CAP",
    "MONTHLY_SPEND_USD",
)


def structured_answer(claims: Sequence[tuple[str, Sequence[str]]], *, answer: str = "ok") -> str:
    """Build a structured-answer JSON string (Design.md §3.3) for test doubles.

    `claims` is a sequence of `(claim_text, [chunk_id, ...])`; markers are the
    raw chunk_ids (as the model would emit them), and `citations` is derived so
    every cited marker maps to its chunk_id.
    """
    cited: list[str] = []
    for _text, chunk_ids in claims:
        for chunk_id in chunk_ids:
            if chunk_id not in cited:
                cited.append(chunk_id)
    payload = {
        "status": "answered",
        "answer": answer,
        "claims": [{"text": text, "citation_ids": list(chunk_ids)} for text, chunk_ids in claims],
        "citations": [{"citation_id": chunk_id, "chunk_id": chunk_id} for chunk_id in cited],
    }
    return json.dumps(payload)


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


def make_runtime(
    config: AxiomConfig,
    *,
    llm_responses: Iterable[str] = (),
    llm_default: str = INSUFFICIENT_EVIDENCE_JSON,
    embed_dimensions: int = 64,
    web_results: Sequence[WebResult] = (),
) -> Runtime:
    """An offline runtime: SQLite + deterministic fake providers.

    The same `HashingEmbedder` embeds documents and queries, so cosine retrieval
    in `InMemoryVectorStore` is meaningful; `ScriptedLLM` returns queued
    structured-JSON answers (CLAUDE.md: recorded fixtures, never live calls).
    `ScriptedWebSearch` defaults to returning nothing, so the Phase 3 fallback
    path is wired but yields the honest refusal unless a test supplies results.
    """
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    return Runtime(
        config,
        database,
        embedder=HashingEmbedder(embed_dimensions),
        llm=ScriptedLLM(llm_responses, default=llm_default),
        vector_store=InMemoryVectorStore(),
        web_search=ScriptedWebSearch(list(web_results)),
    )
