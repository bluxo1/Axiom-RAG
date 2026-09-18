"""Chat and history end-to-end (Phases.md Phase 1, Design.md §1.2-1.3).

Offline: HashingEmbedder + InMemoryVectorStore + a scripted/echoing LLM. No live
calls (CLAUDE.md). The echoing LLM cites the first chunk_id it is given in the
context, so the citation-rewrite path is exercised without knowing ids up front.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.config import AxiomConfig
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder
from app.llm.provider import INSUFFICIENT_EVIDENCE_JSON, LLMProvider, ScriptedLLM
from app.main import create_app
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore
from tests.helpers import TEST_DATABASE_URL, structured_answer

DOCS = "/api/v1/documents"
CHAT = "/api/v1/chat"
SESSIONS = "/api/v1/sessions"

# Context entries render as "[<chunk_id>] (where)\n<text>" (prompts.build_context).
_CONTEXT_ENTRY = re.compile(r"\[([0-9a-f]{40})\] \([^)]*\)\n(.+?)(?=\n\n\[|\Z)", re.DOTALL)


class EchoingLLM:
    """Grounded by construction: cites the first context chunk and uses that
    chunk's own text as the claim, so the support check passes."""

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
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
        def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
            return INSUFFICIENT_EVIDENCE_JSON

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


# ── Phase 2: citation grounding is a hard gate at the API boundary ────────────

_FABRICATED = "f" * 40  # a well-formed but never-retrieved chunk_id


def _grounded_chunk_id(system_prompt: str) -> str:
    match = _CONTEXT_ENTRY.search(system_prompt)
    assert match is not None
    return match.group(1)


class _CapturingLLM:
    """Wraps a payload builder that receives the retrieved (chunk_id, chunk_text)."""

    def __init__(self, build: Callable[[str, str], str]) -> None:
        self._build = build
        self.calls = 0

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        match = _CONTEXT_ENTRY.search(system_prompt)
        assert match is not None
        return self._build(match.group(1), match.group(2).strip())


@pytest.mark.parametrize(
    "build",
    [
        # A claim citing only a fabricated chunk_id.
        lambda _cid, _text: structured_answer([("Totally made up fact.", [_FABRICATED])]),
        # A real chunk_id on a claim the chunk does not support (no keyword overlap).
        lambda cid, _text: structured_answer([("Zebras orbit distant nebulae.", [cid])]),
    ],
)
def test_fabricated_or_unsupported_citation_never_reaches_response(
    config: AxiomConfig, build: Callable[[str, str], str]
) -> None:
    with _chat_client(config, _CapturingLLM(build)) as client:
        _ingest(client, b"The policy covers water damage and flooding events.")
        payload = client.post(CHAT, json={"question": "Does the policy cover flooding?"}).json()

    # Nothing survived grounding -> honest refusal, and the fake id is absent.
    assert payload["insufficient_evidence"] is True
    assert payload["citations"] == []
    assert _FABRICATED not in payload["answer"]


def test_partially_fabricated_claim_keeps_only_the_valid_citation(config: AxiomConfig) -> None:
    def build(chunk_id: str, chunk_text: str) -> str:
        return structured_answer([(chunk_text, [_FABRICATED, chunk_id])])

    with _chat_client(config, _CapturingLLM(build)) as client:
        _ingest(client, b"The warranty lasts twelve months from purchase.")
        payload = client.post(CHAT, json={"question": "How long is the warranty?"}).json()

    assert payload["insufficient_evidence"] is False
    assert _FABRICATED not in payload["answer"]
    assert [c["id"] for c in payload["citations"]] == ["c1"]


def test_grounding_failure_regenerates_once_then_refuses(config: AxiomConfig) -> None:
    # Both attempts cite a fabricated id: grounding fails twice -> refusal.
    bad = structured_answer([("Invented claim.", [_FABRICATED])])
    llm = ScriptedLLM([bad, bad])

    with _chat_client(config, llm) as client:
        _ingest(client, b"The policy covers water damage and flooding events.")
        payload = client.post(CHAT, json={"question": "Does it cover flooding?"}).json()

    assert payload["insufficient_evidence"] is True
    assert len(llm.calls) == 2  # original + exactly one regeneration


def test_regeneration_recovers_when_second_attempt_is_grounded(config: AxiomConfig) -> None:
    captured: dict[str, str] = {}

    class FirstBadThenGood:
        def __init__(self) -> None:
            self.calls = 0

        def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            captured["cid"] = _grounded_chunk_id(system_prompt)
            match = _CONTEXT_ENTRY.search(system_prompt)
            assert match is not None
            if self.calls == 1:
                return structured_answer([("Fake.", [_FABRICATED])])
            return structured_answer([(match.group(2).strip(), [captured["cid"]])])

    llm = FirstBadThenGood()
    with _chat_client(config, llm) as client:
        _ingest(client, b"The policy covers water damage and flooding events.")
        payload = client.post(CHAT, json={"question": "Does it cover flooding?"}).json()

    assert llm.calls == 2
    assert payload["insufficient_evidence"] is False
    assert payload["citations"][0]["id"] == "c1"


def test_malformed_json_is_repaired_once_then_502(config: AxiomConfig) -> None:
    llm = ScriptedLLM(["not json at all", "{still: broken"])

    with _chat_client(config, llm) as client:
        _ingest(client, b"content about anything")
        response = client.post(CHAT, json={"question": "anything?"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "INVALID_LLM_RESPONSE"
    assert len(llm.calls) == 2  # original + one repair pass


def test_malformed_json_repaired_to_valid_answer(config: AxiomConfig) -> None:
    class BrokenThenGood:
        def __init__(self) -> None:
            self.calls = 0

        def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "}{ not json"
            match = _CONTEXT_ENTRY.search(system_prompt)
            assert match is not None
            return structured_answer([(match.group(2).strip(), [match.group(1)])])

    llm = BrokenThenGood()
    with _chat_client(config, llm) as client:
        _ingest(client, b"The policy covers water damage and flooding events.")
        payload = client.post(CHAT, json={"question": "flooding?"}).json()

    assert llm.calls == 2
    assert payload["insufficient_evidence"] is False
    assert "[c1]" in payload["answer"]
