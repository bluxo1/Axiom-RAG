"""LLM providers (Rules.md §2: all generation goes through this interface).

Phase 1 needs one operation: a grounded completion (system prompt with context
+ the user's question -> answer text). The structured-output schema and the
faithfulness self-check (Design.md §3.2-3.3) are Phase 2 additions to this
interface.

`ScriptedLLM` is the test double: it returns queued responses (or echoes a
default), so the chat path runs with no live call.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """A single grounded-completion call."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str: ...


class ScriptedLLM:
    """Returns pre-set responses in order; falls back to a default when empty.

    For tests only — deterministic and offline.
    """

    def __init__(
        self, responses: Iterable[str] = (), *, default: str = "INSUFFICIENT_EVIDENCE"
    ) -> None:
        self._responses: deque[str] = deque(responses)
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._responses:
            return self._responses.popleft()
        return self._default


class OpenAILLM:
    """OpenAI chat completion via LlamaIndex (Architecture.md §5)."""

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

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        from llama_index.core.llms import ChatMessage, MessageRole

        response = self._client.chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=user_prompt),
            ]
        )
        return (response.message.content or "").strip()
