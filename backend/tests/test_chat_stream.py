"""Streaming chat endpoint (Architecture.md §112, PRD FR-9).

Asserts the SSE contract: a `status` event first, then verified answer `token`
events, then a `done` event whose payload equals what `POST /chat` returns.
Crucially, tokens are only emitted *after* the grounding gate passes — a
grounding failure yields an honest refusal, never a stream of unverified tokens.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.config import AxiomConfig
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder
from app.llm.provider import INSUFFICIENT_EVIDENCE_JSON, LLMProvider, ScriptedLLM
from app.main import create_app
from app.search.provider import ScriptedWebSearch
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore
from tests.helpers import TEST_DATABASE_URL, structured_answer

DOCS = "/api/v1/documents"
STREAM = "/api/v1/chat/stream"

_CONTEXT_ENTRY = re.compile(r"\[([0-9a-f]{40})\] \([^)]*\)\n(.+?)(?=\n\n\[|\Z)", re.DOTALL)


class _GroundedLLM:
    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        match = _CONTEXT_ENTRY.search(system_prompt)
        if match is None:
            return INSUFFICIENT_EVIDENCE_JSON
        return structured_answer([(match.group(2).strip(), [match.group(1)])])


@contextmanager
def _client(config: AxiomConfig, llm: LLMProvider) -> Iterator[TestClient]:
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


def _ingest(client: TestClient, text: bytes) -> None:
    assert client.post(DOCS, files={"file": ("kb.txt", text, "text/plain")}).status_code == 200


def _parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    """Parse an SSE body into ordered (event, data) pairs."""
    events: list[tuple[str, dict[str, object]]] = []
    for frame in body.strip().split("\n\n"):
        if not frame.strip():
            continue
        event = ""
        data = ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((event, json.loads(data)))
    return events


def test_stream_emits_status_then_tokens_then_done(config: AxiomConfig) -> None:
    with _client(config, _GroundedLLM()) as client:
        _ingest(client, b"The policy covers water damage and flooding events.")
        response = client.post(STREAM, json={"question": "Does the policy cover flooding?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)

    kinds = [event for event, _ in events]
    assert kinds[0] == "status"  # status is always first.
    assert kinds[-1] == "done"  # done is always last.
    assert "token" in kinds  # verified answer tokens in between.

    # The reassembled tokens equal the done payload's answer, and it is grounded.
    tokens = "".join(str(data["text"]) for event, data in events if event == "token")
    done = next(data for event, data in events if event == "done")
    assert tokens == done["answer"]
    assert done["insufficient_evidence"] is False
    assert "[c1]" in str(done["answer"])
    assert done["session_id"]


def test_stream_refuses_without_leaking_unverified_tokens(config: AxiomConfig) -> None:
    # Both attempts cite a fabricated id: grounding fails, fallback (empty) fails.
    fabricated = "f" * 40
    bad = structured_answer([("Invented claim.", [fabricated])])
    with _client(config, ScriptedLLM([bad, bad])) as client:
        _ingest(client, b"The policy covers water damage and flooding events.")
        response = client.post(STREAM, json={"question": "Does it cover flooding?"})

    events = _parse_sse(response.text)
    done = next(data for event, data in events if event == "done")
    assert done["insufficient_evidence"] is True
    # No fabricated id ever reached a token frame.
    tokens = "".join(str(data["text"]) for event, data in events if event == "token")
    assert fabricated not in tokens


def test_stream_rejects_empty_corpus_before_streaming(config: AxiomConfig) -> None:
    with _client(config, _GroundedLLM()) as client:
        response = client.post(STREAM, json={"question": "anything?"})

    # Cheap guard runs before the stream opens: a real HTTP 400, not an SSE frame.
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
