"""LLM providers (Rules.md §2: all generation goes through this interface).

Phase 2 generation is structured: the model returns a JSON object
(`{status, answer, claims, citations}`, Design.md §3.3), not prose. The provider
returns that JSON as raw text; parsing and validation happen in
`rag/generation.py`, and citation grounding in `rag/grounding.py`.

`ScriptedLLM` is the test double: it returns queued JSON responses (or a default
`insufficient_evidence` payload), so the chat path runs with no live call.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

# Default structured payload for a test double with nothing queued: an honest
# refusal, so an unscripted call can never fabricate a grounded answer.
INSUFFICIENT_EVIDENCE_JSON = (
    '{"status": "insufficient_evidence", "answer": "", "claims": [], "citations": []}'
)


@runtime_checkable
class LLMProvider(Protocol):
    """A single grounded, structured-output completion call."""

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str: ...


class ScriptedLLM:
    """Returns pre-set JSON responses in order; falls back to a refusal default.

    For tests only — deterministic and offline. `calls` records every call so a
    test can assert the regenerate/repair loop made exactly N generations.
    """

    def __init__(
        self, responses: Iterable[str] = (), *, default: str = INSUFFICIENT_EVIDENCE_JSON
    ) -> None:
        self._responses: deque[str] = deque(responses)
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._responses:
            return self._responses.popleft()
        return self._default


class OpenAILLM:
    """OpenAI chat completion via LlamaIndex (Architecture.md §5).

    JSON mode (`response_format={"type": "json_object"}`) makes the model return
    a JSON object, which `rag/generation.py` validates against the schema.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        timeout_seconds: float,
    ) -> None:
        from llama_index.llms.openai import OpenAI

        self._client = OpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout_seconds,
        )

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        from llama_index.core.llms import ChatMessage, MessageRole

        response = self._client.chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt),
            ],
            response_format={"type": "json_object"},
        )
        return (response.message.content or "").strip()
