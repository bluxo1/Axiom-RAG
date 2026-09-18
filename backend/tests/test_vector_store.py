"""In-memory vector store: cosine ranking, top-k, deletion (ADR-0002)."""

from __future__ import annotations

import pytest

from app.rag.types import TextChunk
from app.vector.store import InMemoryVectorStore


def _chunk(chunk_id: str, doc_id: str = "doc-1") -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=f"{doc_id}.txt",
        index=0,
        text=f"text {chunk_id}",
        page=None,
    )


def test_query_ranks_by_cosine_similarity() -> None:
    store = InMemoryVectorStore()
    store.add(
        [_chunk("near"), _chunk("far")],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.query([0.9, 0.1], k=2)

    assert [r.chunk_id for r in results] == ["near", "far"]
    assert results[0].score > results[1].score


def test_query_respects_top_k() -> None:
    store = InMemoryVectorStore()
    store.add([_chunk(f"c{i}") for i in range(5)], [[float(i), 1.0] for i in range(5)])

    assert len(store.query([1.0, 1.0], k=3)) == 3


def test_delete_document_removes_only_its_chunks() -> None:
    store = InMemoryVectorStore()
    store.add(
        [_chunk("a", "doc-1"), _chunk("b", "doc-2")],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    store.delete_document("doc-1")
    results = store.query([1.0, 1.0], k=10)

    assert [r.chunk_id for r in results] == ["b"]


def test_length_mismatch_is_rejected() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="same length"):
        store.add([_chunk("a")], [[1.0], [2.0]])
