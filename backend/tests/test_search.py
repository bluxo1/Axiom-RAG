"""Retrieval endpoint (Phases.md Phase 1: top-k semantic search)."""

from __future__ import annotations

from fastapi.testclient import TestClient

DOCS = "/api/v1/documents"
SEARCH = "/api/v1/search"


def _ingest(client: TestClient, name: str, body: bytes) -> None:
    assert client.post(DOCS, files={"file": (name, body, "text/plain")}).status_code == 200


def test_search_on_empty_corpus_returns_no_hits(client: TestClient) -> None:
    response = client.post(SEARCH, json={"query": "anything"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "anything"
    assert payload["results"] == []


def test_search_ranks_the_relevant_document_first(client: TestClient) -> None:
    _ingest(client, "flood.txt", b"The policy covers water damage and flooding events.")
    _ingest(client, "garden.txt", b"Pruning roses and planting tulips in spring soil.")

    results = client.post(SEARCH, json={"query": "flooding water damage"}).json()["results"]

    assert results
    assert results[0]["doc"] == "flood.txt"
    scores = [hit["score"] for hit in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_clamped_to_the_configured_maximum(client: TestClient) -> None:
    for index in range(12):
        _ingest(client, f"doc{index}.txt", f"unique token{index} shared content".encode())

    # Ask for more than config allows (config.retrieval.top_k == 8).
    results = client.post(SEARCH, json={"query": "shared content", "top_k": 50}).json()["results"]

    assert len(results) == 8


def test_empty_query_is_rejected(client: TestClient) -> None:
    assert client.post(SEARCH, json={"query": ""}).status_code == 422
