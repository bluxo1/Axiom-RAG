"""Chat and history end-to-end (Phases.md Phase 1, Design.md §1.2-1.3).

Offline: HashingEmbedder + InMemoryVectorStore + a scripted/echoing LLM. No live
calls (CLAUDE.md). The echoing LLM cites the first chunk_id it is given in the
context, so the citation-rewrite path is exercised without knowing ids up front.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.config import AxiomConfig
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder
from app.llm.provider import LLMProvider
from app.main import create_app
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore
from tests.helpers import TEST_DATABASE_URL

DOCS = "/api/v1/documents"
CHAT = "/api/v1/chat"
SESSIONS = "/api/v1/sessions"

_CHUNK_MARKER = re.compile(r"\[([0-9a-f]{40})\]")


class EchoingLLM:
    """Cites the first chunk in the provided context; grounded by construction."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        match = _CHUNK_MARKER.search(system_prompt)
        if match is None:
            return "INSUFFICIENT_EVIDENCE"
        return f"The documents address this. [{match.group(1)}]"


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
    )
    try:
        with TestClient(create_app(config, runtime=runtime)) as client:
            yield client
    finally:
        database.dispose()


def _ingest(client: TestClient, text: bytes) -> None:
    assert client.post(DOCS, files={"file": ("kb.txt", text, "text/plain")}).status_code == 200


def test_chat_without_documents_is_a_friendly_400(config: AxiomConfig) -> None:
    with _chat_client(config, EchoingLLM()) as client:
        response = client.post(CHAT, json={"question": "What is covered?"})

    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "VALIDATION_ERROR"
    assert "cta" in body["details"]


def test_grounded_answer_has_rewritten_citations(config: AxiomConfig) -> None:
    with _chat_client(config, EchoingLLM()) as client:
        _ingest(client, b"The policy covers water damage and flooding events.")

        response = client.post(CHAT, json={"question": "Does the policy cover flooding?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["insufficient_evidence"] is False
    assert "[c1]" in payload["answer"]
    assert len(payload["citations"]) == 1
    citation = payload["citations"][0]
    assert citation["id"] == "c1"
    assert citation["source"] == "kb"
    assert citation["quote"]
    assert payload["session_id"]


def test_insufficient_evidence_is_reported_without_citations(config: AxiomConfig) -> None:
    class SilentLLM:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "INSUFFICIENT_EVIDENCE"

    with _chat_client(config, SilentLLM()) as client:
        _ingest(client, b"Unrelated content about gardening tools.")
        response = client.post(CHAT, json={"question": "What is the capital of France?"})

    payload = response.json()
    assert payload["insufficient_evidence"] is True
    assert payload["citations"] == []


def test_whitespace_question_is_rejected(config: AxiomConfig) -> None:
    with _chat_client(config, EchoingLLM()) as client:
        _ingest(client, b"some content")
        response = client.post(CHAT, json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_history_replays_the_turn(config: AxiomConfig) -> None:
    with _chat_client(config, EchoingLLM()) as client:
        _ingest(client, b"The warranty lasts twelve months from purchase.")
        chat = client.post(CHAT, json={"question": "How long is the warranty?"}).json()
        session_id = chat["session_id"]

        history = client.get(f"{SESSIONS}/{session_id}")

    assert history.status_code == 200
    body = history.json()
    assert body["session_id"] == session_id
    assert len(body["messages"]) == 1
    turn = body["messages"][0]
    assert turn["question"] == "How long is the warranty?"
    assert "[c1]" in turn["answer"]
    assert turn["citations"][0]["id"] == "c1"


def test_history_for_unknown_session_is_404(config: AxiomConfig) -> None:
    with _chat_client(config, EchoingLLM()) as client:
        response = client.get(f"{SESSIONS}/nope")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_supplied_session_id_is_reused(config: AxiomConfig) -> None:
    with _chat_client(config, EchoingLLM()) as client:
        _ingest(client, b"Axiom cites its sources.")
        first = client.post(CHAT, json={"question": "What does Axiom do?", "session_id": "sess-1"})
        assert first.json()["session_id"] == "sess-1"

        client.post(CHAT, json={"question": "Again?", "session_id": "sess-1"})
        history = client.get(f"{SESSIONS}/sess-1").json()

    assert len(history["messages"]) == 2


@pytest.mark.parametrize("question", ["", None])
def test_missing_question_is_a_validation_error(config: AxiomConfig, question: str | None) -> None:
    with _chat_client(config, EchoingLLM()) as client:
        _ingest(client, b"content")
        response = client.post(CHAT, json={"question": question})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
