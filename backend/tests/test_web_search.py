"""Web-search provider double and the result→chunk adapter (Architecture.md §3.6).

Offline: only `ScriptedWebSearch` and the pure `web_results_to_chunks` adapter
are exercised (the real Tavily/Brave providers make live calls and are never hit
in tests, per CLAUDE.md).
"""

from __future__ import annotations

from app.confidence.scoring import normalize_similarity
from app.search.provider import (
    WEB_DOC_ID,
    ScriptedWebSearch,
    WebResult,
    WebSearchProvider,
    web_results_to_chunks,
)


def _result(
    title: str = "T", url: str = "https://example.com", relevance: float = 0.5
) -> WebResult:
    return WebResult(title=title, url=url, content="some content", relevance=relevance)


def test_scripted_search_records_calls_and_clamps_to_max_results() -> None:
    search: WebSearchProvider = ScriptedWebSearch([_result(), _result(), _result()])
    results = search.search("a query", max_results=2)

    assert len(results) == 2
    assert isinstance(search, ScriptedWebSearch)
    assert search.calls == ["a query"]  # every call is recorded for assertions.


def test_empty_scripted_search_returns_nothing() -> None:
    assert ScriptedWebSearch().search("q", max_results=5) == []


def test_web_results_to_chunks_labels_source_and_url() -> None:
    chunks = web_results_to_chunks([_result(title="Live", url="https://ex.com/a", relevance=0.9)])

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source == "web"
    assert chunk.url == "https://ex.com/a"
    assert chunk.doc_id == WEB_DOC_ID
    assert chunk.chunk_id == "web#1"
    assert chunk.doc_name == "Live"
    assert chunk.page is None


def test_web_results_to_chunks_maps_relevance_into_cosine_convention() -> None:
    # score = 2*relevance - 1, so the confidence scorer's normalization recovers
    # the original 0-1 relevance the provider reported.
    chunks = web_results_to_chunks([_result(relevance=0.9)])
    assert chunks[0].score == 2.0 * 0.9 - 1.0
    assert normalize_similarity(chunks[0].score) == 0.9


def test_web_results_to_chunks_numbers_ids_in_order() -> None:
    chunks = web_results_to_chunks([_result(), _result(), _result()])
    assert [c.chunk_id for c in chunks] == ["web#1", "web#2", "web#3"]


def test_missing_title_falls_back_to_url_as_doc_name() -> None:
    chunks = web_results_to_chunks([_result(title="", url="https://ex.com/x")])
    assert chunks[0].doc_name == "https://ex.com/x"
