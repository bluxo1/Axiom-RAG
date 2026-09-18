"""Web-search providers (Rules.md §2: all search goes through this interface).

`WebSearchProvider` is what the fallback path depends on. `ScriptedWebSearch`
is the offline test double (no live call, per CLAUDE.md). `TavilyWebSearch` and
`BraveWebSearch` are the real providers, built lazily in the runtime with an API
key — never reached in tests.

`web_results_to_chunks` adapts results into `RetrievedChunk`s so web evidence
flows through the same generation → grounding → confidence path as the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.rag.types import RetrievedChunk

# A synthetic doc_id groups every web result of one turn; chunk_ids are
# `web#1, web#2, ...` so the model can cite them like any other context id.
WEB_DOC_ID = "web"


@dataclass(frozen=True)
class WebResult:
    """One hit from a web-search API.

    `relevance` is the provider's own 0-1 relevance (Tavily returns it; Brave is
    given a rank-decayed stand-in), used to seed the chunk's retrieval score.
    """

    title: str
    url: str
    content: str
    relevance: float = 0.5


@runtime_checkable
class WebSearchProvider(Protocol):
    """A single live web search returning ranked results."""

    def search(self, query: str, *, max_results: int) -> list[WebResult]: ...


class ScriptedWebSearch:
    """Returns pre-set results; for tests only — deterministic and offline.

    `calls` records each query so a test can assert the fallback fired exactly
    once (and only when the corpus could not ground the answer).
    """

    def __init__(self, results: list[WebResult] | None = None) -> None:
        self._results = results or []
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int) -> list[WebResult]:
        self.calls.append(query)
        return self._results[:max_results]


def web_results_to_chunks(results: list[WebResult]) -> list[RetrievedChunk]:
    """Adapt web results into retrievable chunks (source="web", live url).

    The provider's 0-1 relevance is mapped into the cosine convention
    (`2*relevance - 1`) the rest of the pipeline scores in, so the confidence
    scorer's normalization recovers the original relevance.
    """
    chunks: list[RetrievedChunk] = []
    for index, result in enumerate(results, start=1):
        chunks.append(
            RetrievedChunk(
                chunk_id=f"{WEB_DOC_ID}#{index}",
                doc_id=WEB_DOC_ID,
                doc_name=result.title or result.url,
                text=result.content,
                score=2.0 * result.relevance - 1.0,
                page=None,
                source="web",
                url=result.url,
            )
        )
    return chunks


class TavilyWebSearch:
    """Tavily Search API (Architecture.md §3.6, default provider)."""

    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds

    def search(self, query: str, *, max_results: int) -> list[WebResult]:
        import httpx

        response = httpx.post(
            self._ENDPOINT,
            json={
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            WebResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                content=str(item.get("content", "")),
                relevance=float(item.get("score", 0.5)),
            )
            for item in payload.get("results", [])
        ]


class BraveWebSearch:
    """Brave Search API (Architecture.md §3.6, dev alternative)."""

    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
        self._api_key = api_key
        self._timeout = timeout_seconds

    def search(self, query: str, *, max_results: int) -> list[WebResult]:
        import httpx

        response = httpx.get(
            self._ENDPOINT,
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": self._api_key, "Accept": "application/json"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        results = response.json().get("web", {}).get("results", [])
        # Brave has no per-result score; decay by rank so earlier hits weigh more.
        return [
            WebResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                content=str(item.get("description", "")),
                relevance=1.0 / (rank + 1),
            )
            for rank, item in enumerate(results[:max_results])
        ]
