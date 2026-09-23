"""LLM provider failures surface as structured errors, never a bare 500.

Found by the live smoke test (2026-09-23): Gemini's OpenAI-compatible endpoint
returned an upstream failure on the web-fallback path and it escaped as
`INTERNAL_ERROR` 500. The contract pinned here (Design.md §5):

* the real `OpenAILLM` translates the SDK's `APIError` family into
  `LLMProviderError` (timeouts stay `LLMTimeoutError`);
* the chat path surfaces that as the structured 503 `LLM_PROVIDER_FAILED`,
  on the corpus leg *and* on the web-fallback leg — structured errors stay
  visible there so budget exhaustion is never masked as a refusal
  (Rule 8); only non-`AxiomError` bugs degrade to the refusal floor.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app.config import AxiomConfig
from app.core.errors import ErrorCode
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder
from app.llm.provider import (
    INSUFFICIENT_EVIDENCE_JSON,
    LLMProvider,
    LLMProviderError,
    LLMTimeoutError,
    OpenAILLM,
)
from app.main import create_app
from app.search.provider import ScriptedWebSearch, WebResult
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore
from tests.helpers import TEST_DATABASE_URL, structured_answer

DOCS = "/api/v1/documents"
CHAT = "/api/v1/chat"

# Corpus context entries in the system prompt: "[<40-hex chunk_id>] (...)\ntext".
_CONTEXT_ENTRY = re.compile(r"\[([0-9a-f]{40})\] \([^)]*\)\n(.+?)(?=\n\n\[|\Z)", re.DOTALL)
# Web chunks are cited by their synthetic ids (search/provider.py).
_WEB_CHUNK_ID = "web#1"

_WEB_RESULTS = [
    WebResult(
        title="Reykjavik — Wikipedia",
        url="https://en.wikipedia.org/wiki/Reykjavik",
        content="Reykjavik is the capital of Iceland with about 140,000 residents.",
        relevance=0.9,
    )
]


class FailingLLM:
    """Raises `LLMProviderError` on every call (the provider is down)."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        raise LLMProviderError("429 resource exhausted (quota exceeded)")


class RefusesThenFailingLLM:
    """Answers the corpus leg with an honest no-evidence verdict, then fails.

    The corpus answer grounds to nothing (no fabricated citations), which
    spends the grounding regeneration budget (`grounding.max_regenerations`),
    so the corpus leg consumes the first two calls. The router then proceeds
    to the web fallback — whose generation call (the third) raises
    `LLMProviderError`. This isolates the web leg's failure handling.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.calls <= 2:  # corpus leg: initial + grounding regeneration.
            return INSUFFICIENT_EVIDENCE_JSON
        raise LLMProviderError("503 model overloaded")


def _answer_citing(system_prompt: str, chunk_id: str) -> str:
    """A structured answer citing `chunk_id` with its verbatim context text."""
    if chunk_id.startswith("web#"):
        # Web chunks are cited directly; the text is not echoed in the prompt
        # with a stable format the test can parse, so reuse the question.
        return structured_answer([("The web result covers it.", [chunk_id])])
    match = _CONTEXT_ENTRY.search(system_prompt)
    if match is None:
        return INSUFFICIENT_EVIDENCE_JSON
    real_id, chunk_text = match.group(1), match.group(2).strip()
    return structured_answer([(chunk_text, [real_id if chunk_id == "*" else chunk_id])])


@contextmanager
def _chat_client(
    config: AxiomConfig,
    llm: LLMProvider,
    *,
    web_results: list[WebResult] | None = None,
) -> Iterator[TestClient]:
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    runtime = Runtime(
        config,
        database,
        embedder=HashingEmbedder(),
        llm=llm,
        vector_store=InMemoryVectorStore(),
        web_search=ScriptedWebSearch(web_results if web_results is not None else []),
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


def test_provider_failure_returns_structured_503(config: AxiomConfig) -> None:
    """A quota/availability failure is a 503 with a stable code, not a 500."""
    llm = FailingLLM()
    with _chat_client(config, llm) as client:
        _ingest(client)
        response = client.post(CHAT, json={"question": "How do I get a Zephyr API key?"})

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["error"]["code"] == ErrorCode.LLM_PROVIDER_FAILED.value
    assert "failed" in body["error"]["message"].lower()


def test_provider_failure_on_web_fallback_propagates_structured_503(
    config: AxiomConfig,
) -> None:
    """A provider failure on the web leg is the structured 503, never a 500.

    The corpus leg answers with an honest no-evidence verdict, so the router
    proceeds to the web fallback; the search succeeds but generation fails
    with `LLMProviderError`. Structured errors stay visible (Rule 8: budget
    exhaustion must not be masked as a refusal), so the envelope surfaces.
    """
    llm = RefusesThenFailingLLM()
    with _chat_client(config, llm, web_results=list(_WEB_RESULTS)) as client:
        _ingest(client)
        response = client.post(CHAT, json={"question": "What is the population of Reykjavik?"})

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["error"]["code"] == ErrorCode.LLM_PROVIDER_FAILED.value
    assert llm.calls == 3  # corpus leg (x2 with regen) + web leg, all reached.


def test_streaming_provider_failure_emits_sse_error_event(config: AxiomConfig) -> None:
    """The SSE path delivers the structured failure as an `error` event."""
    llm = FailingLLM()
    with _chat_client(config, llm) as client:
        _ingest(client)
        with client.stream("POST", "/api/v1/chat/stream", json={"question": "hi"}) as response:
            assert response.status_code == 200
            body = b"".join(response.iter_bytes()).decode()

    assert "event: error" in body
    assert ErrorCode.LLM_PROVIDER_FAILED.value in body


# ─── provider translation: SDK errors → our types (no live call) ─────────────


class _FakeCompletions:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def create(self, **_kwargs: object) -> object:
        raise self._exc


class _FakeChat:
    """Callable stand-in for the LlamaIndex client's `.chat(...)` method."""

    def __init__(self, exc: Exception) -> None:
        self.completions = _FakeCompletions(exc)
        self._exc = exc

    def __call__(self, *_messages: object, **_kwargs: object) -> object:
        raise self._exc


class _FakeOpenAI:
    """Stands in for the LlamaIndex-wrapped OpenAI client."""

    def __init__(self, exc: Exception) -> None:
        self.chat = _FakeChat(exc)


def _sdk_error(status_code: int, message: str) -> openai.APIStatusError:
    """Build an OpenAI SDK status error the way the SDK itself does."""
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError(message, response=response, body=None)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_sdk_error(429, "quota exceeded"), LLMProviderError),
        (_sdk_error(500, "model overloaded"), LLMProviderError),
        (_sdk_error(400, "rejected request"), LLMProviderError),
        (
            openai.APITimeoutError(
                httpx.Request("POST", "https://provider.test/v1/chat/completions")
            ),
            LLMTimeoutError,
        ),
    ],
    ids=["429-quota", "500-overloaded", "400-rejected", "timeout"],
)
def test_openai_llm_translates_sdk_errors(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, expected: type[Exception]
) -> None:
    """`OpenAILLM` maps the SDK's error family onto our two provider types."""
    llm = OpenAILLM(api_key="test-key", model="test-model", temperature=0.0, timeout_seconds=1.0)
    monkeypatch.setattr(llm, "_client", _FakeOpenAI(exc))

    with pytest.raises(expected):
        llm.complete_structured(system_prompt="system", user_prompt="user")


class _CorpusAnsweringLLM:
    """Answers the corpus leg by citing its real retrieved chunk id."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return _answer_citing(system_prompt, "*")


def test_grounded_answer_still_flows_through(config: AxiomConfig) -> None:
    """Sanity: the same client path answers normally when providers behave."""
    with _chat_client(config, _CorpusAnsweringLLM()) as client:
        _ingest(client)
        response = client.post(CHAT, json={"question": "How do I get a Zephyr API key?"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["insufficient_evidence"]
    assert body["fallback_used"] is False
