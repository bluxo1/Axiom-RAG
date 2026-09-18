"""Providers that pass the Rule 8 budget guard on every call.

`GuardedLLM` and `GuardedEmbedder` wrap a provider and apply, in order:

1. `GuardedLLM`: the per-request token cap on the text about to be sent
   (system prompt + question — the spec's `MAX_REQUEST_TOKENS`), then the
   daily/monthly caps, then the call, then usage recorded to `spend_log`.
2. `GuardedEmbedder`: the daily/monthly caps, then the call, then usage
   recorded to `spend_log`. The per-request cap deliberately does *not* apply
   to embedding batches: ingestion must be able to embed a whole document
   (hundreds of chunks), and that spend is already bounded by the daily token
   cap and the monthly USD cap. The provider batches API calls itself
   (`embedding.batch_size`).

Usage tokens are a tiktoken estimate (fixed encoding); a provider that reports
real usage can replace the estimate later without changing the guard.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.budget import BudgetGuard, Usage, count_tokens
from app.llm.embeddings import Embedder
from app.llm.provider import LLMProvider
from app.search.provider import WebResult, WebSearchProvider


class GuardedLLM:
    """LLMProvider decorator: per-request cap, caps, then spend logging."""

    def __init__(self, inner: LLMProvider, guard: BudgetGuard, *, model: str) -> None:
        self._inner = inner
        self._guard = guard
        self._model = model

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        self._guard.check_request(
            texts={"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        self._guard.check_caps()
        answer = self._inner.complete_structured(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        tokens = count_tokens(system_prompt) + count_tokens(user_prompt) + count_tokens(answer)
        self._guard.record(
            provider="llm",
            kind="llm",
            model=self._model,
            usage=Usage(tokens=tokens),
        )
        return answer


class GuardedEmbedder:
    """Embedder decorator: daily/monthly caps + spend logging on every embed."""

    def __init__(self, inner: Embedder, guard: BudgetGuard, *, model: str) -> None:
        self._inner = inner
        self._guard = guard
        self._model = model

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        if not batch:
            return []
        # A batch is one paid call: it must fit in today's remaining budget,
        # so a single huge document cannot consume (or exceed) a whole day.
        self._guard.check_batch(tokens=sum(count_tokens(text) for text in batch))
        vectors = self._inner.embed_texts(batch)
        self._guard.record(
            provider="embedding",
            kind="embedding",
            model=self._model,
            usage=Usage(tokens=sum(count_tokens(text) for text in batch)),
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        self._guard.check_caps()
        vector = self._inner.embed_query(text)
        self._guard.record(
            provider="embedding",
            kind="embedding",
            model=self._model,
            usage=Usage(tokens=count_tokens(text)),
        )
        return vector


class GuardedWebSearch:
    """WebSearchProvider decorator: caps check, then per-call spend logging.

    Web search is billed per call, not per token (Rule 8), so the recorded row
    carries a flat `cost_usd` and a token estimate (query + result text) that
    still contributes to the daily token cap.
    """

    def __init__(
        self, inner: WebSearchProvider, guard: BudgetGuard, *, provider: str, price_usd: float
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._provider = provider
        self._price_usd = price_usd

    def search(self, query: str, *, max_results: int) -> list[WebResult]:
        self._guard.check_caps()
        results = self._inner.search(query, max_results=max_results)
        tokens = count_tokens(query) + sum(count_tokens(result.content) for result in results)
        self._guard.record(
            provider=self._provider,
            kind="search",
            model=self._provider,
            usage=Usage(tokens=tokens, cost_usd=self._price_usd),
        )
        return results
