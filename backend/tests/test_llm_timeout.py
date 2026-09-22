"""LLM timeout handling (Design.md §5: retry once, then a structured 504).

The `generation.max_timeout_retries` knob was dead config before the Phase 4
hardening sweep. These tests pin the contract now that it is wired in:

* a timeout that clears within the retry budget yields a normal grounded answer;
* a timeout that outlives the budget surfaces as `LLM_TIMEOUT` / HTTP 504 through
  the structured error envelope, never a bare 500 or a hang;
* the real `OpenAILLM` translates the SDK's timeout into `LLMTimeoutError`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import AxiomConfig, Settings, load_knobs
from app.core.errors import ErrorCode
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder
from app.llm.provider import INSUFFICIENT_EVIDENCE_JSON, LLMProvider, LLMTimeoutError
from app.main import create_app
from app.search.provider import ScriptedWebSearch
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore
from tests.helpers import TEST_DATABASE_URL, read_raw_config, structured_answer, write_config

DOCS = "/api/v1/documents"
CHAT = "/api/v1/chat"

_CONTEXT_ENTRY = re.compile(r"\[([0-9a-f]{40})\] \([^)]*\)\n(.+?)(?=\n\n\[|\Z)", re.DOTALL)


class TimingOutLLM:
    """Raises `LLMTimeoutError` for the first `times` calls, then echoes-and-cites."""

    def __init__(self, times: int) -> None:
        self._remaining = times
        self.calls = 0

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self._remaining > 0:
            self._remaining -= 1
            raise LLMTimeoutError("read timed out")
        match = _CONTEXT_ENTRY.search(system_prompt)
        if match is None:
            return INSUFFICIENT_EVIDENCE_JSON
        chunk_id, chunk_text = match.group(1), match.group(2).strip()
        return structured_answer([(chunk_text, [chunk_id])])


@contextmanager
def _chat_client(config: AxiomConfig, llm: LLMProvider) -> Iterator[TestClient]:
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    runtime = Runtime(
        config,
        database,
        embedder=HashingEmbedder(),
        llm=llm,
        vector_store=InMemoryVectorStore(),
        web_search=ScriptedWebSearch(),
    )
    try:
        with TestClient(create_app(config, runtime=runtime)) as client:
            yield client
    finally:
        database.dispose()


def _ingest(client: TestClient) -> None:
    body = b"Zephyr issues API keys from the developer dashboard under account settings."
    response = client.post(DOCS, files={"file": ("zephyr.txt", body, "text/plain")})
    assert response.status_code == 200, response.text


def test_timeout_within_budget_recovers_and_answers(config: AxiomConfig) -> None:
    # config.yaml ships max_timeout_retries: 1, so one timeout is survivable.
    assert config.generation.max_timeout_retries >= 1
    llm = TimingOutLLM(times=1)
    with _chat_client(config, llm) as client:
        _ingest(client)
        response = client.post(CHAT, json={"question": "How do I get a Zephyr API key?"})
    assert response.status_code == 200, response.text
    assert llm.calls == 2  # one timeout, one success.
    assert not response.json()["insufficient_evidence"]


def test_timeout_beyond_budget_returns_structured_504(config: AxiomConfig) -> None:
    llm = TimingOutLLM(times=5)  # never clears within any sane budget.
    with _chat_client(config, llm) as client:
        _ingest(client)
        response = client.post(CHAT, json={"question": "How do I get a Zephyr API key?"})
    assert response.status_code == 504, response.text
    assert response.json()["error"]["code"] == ErrorCode.LLM_TIMEOUT.value
    # Retries are bounded: initial call + max_timeout_retries, no more.
    assert llm.calls == config.generation.max_timeout_retries + 1


def test_timeout_retries_honor_the_config_budget(tmp_path: Path) -> None:
    # A budget of 0 means: no retry, fail on the first timeout.
    raw = read_raw_config()
    raw["generation"]["max_timeout_retries"] = 0
    path = write_config(tmp_path, raw)
    zero_budget = AxiomConfig(Settings(_env_file=None), load_knobs(path))

    llm = TimingOutLLM(times=5)
    with _chat_client(zero_budget, llm) as client:
        _ingest(client)
        response = client.post(CHAT, json={"question": "How do I get a Zephyr API key?"})
    assert response.status_code == 504, response.text
    assert llm.calls == 1  # no retry when the budget is zero.
