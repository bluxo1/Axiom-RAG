"""GET /metrics: routing rates (Rule 7) + spend counters (Rule 8), Design.md §1.3."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.config import AxiomConfig
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder
from app.llm.provider import INSUFFICIENT_EVIDENCE_JSON, LLMProvider
from app.main import create_app
from app.search.provider import ScriptedWebSearch, WebResult, WebSearchProvider
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore
from tests.helpers import TEST_DATABASE_URL, structured_answer

DOCS = "/api/v1/documents"
CHAT = "/api/v1/chat"
METRICS = "/api/v1/metrics"

# Context entries render as "[<chunk_id>] (where)\n<text>".
_CONTEXT_ENTRY = re.compile(r"\[([0-9a-f]{40})\] \([^)]*\)\n(.+?)(?=\n\n\[|\Z)", re.DOTALL)


class _GroundedLLM:
    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        match = _CONTEXT_ENTRY.search(system_prompt)
        if match is None:
            return INSUFFICIENT_EVIDENCE_JSON
        return structured_answer([(match.group(2).strip(), [match.group(1)])])


@contextmanager
def _client(
    config: AxiomConfig, llm: LLMProvider, *, web_search: WebSearchProvider | None = None
) -> Iterator[TestClient]:
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    runtime = Runtime(
        config,
        database,
        embedder=HashingEmbedder(),
        llm=llm,
        vector_store=InMemoryVectorStore(),
        web_search=web_search if web_search is not None else ScriptedWebSearch(),
    )
    try:
        with TestClient(create_app(config, runtime=runtime)) as client:
            yield client
    finally:
        database.dispose()


def _ingest(client: TestClient, text: bytes) -> None:
    assert client.post(DOCS, files={"file": ("kb.txt", text, "text/plain")}).status_code == 200


def test_metrics_reports_zero_rates_on_a_fresh_corpus(config: AxiomConfig) -> None:
    with _client(config, _GroundedLLM()) as client:
        body = client.get(METRICS).json()

    routing = body["routing"]
    assert routing["total_messages"] == 0
    assert routing["flagged_rate"] == 0.0
    assert routing["fallback_rate"] == 0.0
    assert "spend" in body  # the Rule 8 slice is still present.


def test_metrics_counts_a_grounded_answer_as_unflagged(config: AxiomConfig) -> None:
    with _client(config, _GroundedLLM()) as client:
        _ingest(client, b"The policy covers water damage and flooding events.")
        client.post(CHAT, json={"question": "Does the policy cover flooding?"})
        routing = client.get(METRICS).json()["routing"]

    assert routing["total_messages"] == 1
    assert routing["flagged"] == 0
    assert routing["flagged_rate"] == 0.0


def test_metrics_counts_a_refusal(config: AxiomConfig) -> None:
    class SilentLLM:
        def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
            return INSUFFICIENT_EVIDENCE_JSON

    with _client(config, SilentLLM()) as client:
        _ingest(client, b"Unrelated content about gardening tools.")
        client.post(CHAT, json={"question": "What is the capital of France?"})
        routing = client.get(METRICS).json()["routing"]

    assert routing["total_messages"] == 1
    assert routing["refused"] == 1
    assert routing["refusal_rate"] == 1.0


def test_metrics_counts_a_web_fallback(config: AxiomConfig) -> None:
    web = WebResult(
        title="Live",
        url="https://example.com/a",
        content="The policy covers water damage and flooding events.",
        relevance=0.9,
    )

    _web_entry = re.compile(r"\[(web#\d+)\] \([^)]*\)\n(.+?)(?=\n\n\[|\Z)", re.DOTALL)

    class KbRefusesThenWeb:
        def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
            match = _web_entry.search(system_prompt)
            if match is None:
                return INSUFFICIENT_EVIDENCE_JSON
            return structured_answer([(match.group(2).strip(), [match.group(1)])])

    with _client(config, KbRefusesThenWeb(), web_search=ScriptedWebSearch([web])) as client:
        _ingest(client, b"Unrelated content about gardening tools.")
        client.post(CHAT, json={"question": "Does the policy cover flooding?"})
        routing = client.get(METRICS).json()["routing"]

    assert routing["total_messages"] == 1
    assert routing["fallback_used"] == 1
    assert routing["fallback_rate"] == 1.0
