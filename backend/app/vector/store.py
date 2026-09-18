"""Vector store interface and backends (ADR-0002).

`VectorStore` is what the pipeline depends on. `InMemoryVectorStore` is an exact
cosine implementation used by tests. `ChromaVectorStore` talks to the Chroma
server pinned in docker-compose; the client major is kept in step with that
image.

Scores are cosine similarity in `[-1, 1]` (higher is better), normalized the
same way across backends so `config.confidence` thresholds mean one thing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from app.rag.types import RetrievedChunk, TextChunk

# Chroma cannot store a null metadata value; this sentinel stands in for "no
# page" (TXT/MD/URL) and is mapped back to None on read.
_NO_PAGE = -1


class VectorStore(Protocol):
    def add(self, chunks: Sequence[TextChunk], embeddings: Sequence[list[float]]) -> None: ...

    def query(self, embedding: Sequence[float], k: int) -> list[RetrievedChunk]: ...

    def delete_document(self, doc_id: str) -> None: ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Exact cosine search over an in-process list. For tests and small dev sets."""

    def __init__(self) -> None:
        self._chunks: dict[str, TextChunk] = {}
        self._embeddings: dict[str, list[float]] = {}

    def add(self, chunks: Sequence[TextChunk], embeddings: Sequence[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._embeddings[chunk.chunk_id] = list(embedding)

    def query(self, embedding: Sequence[float], k: int) -> list[RetrievedChunk]:
        scored = [(self._score(embedding, chunk_id), chunk_id) for chunk_id in self._chunks]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[RetrievedChunk] = []
        for score, chunk_id in scored[:k]:
            chunk = self._chunks[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_name=chunk.doc_name,
                    text=chunk.text,
                    score=score,
                    page=chunk.page,
                )
            )
        return results

    def _score(self, embedding: Sequence[float], chunk_id: str) -> float:
        return _cosine(embedding, self._embeddings[chunk_id])

    def delete_document(self, doc_id: str) -> None:
        doomed = [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        for chunk_id in doomed:
            del self._chunks[chunk_id]
            del self._embeddings[chunk_id]


class ChromaVectorStore:
    """ChromaDB-backed store (dev vector store, ADR-0002).

    One collection per embedding model (`collection_name`), created with cosine
    space so distances convert cleanly to the similarity scores the rest of the
    pipeline expects.
    """

    def __init__(self, *, host: str, port: int, collection_name: str) -> None:
        import chromadb

        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: Sequence[TextChunk], embeddings: Sequence[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "doc_id": chunk.doc_id,
                    "doc_name": chunk.doc_name,
                    "page": chunk.page if chunk.page is not None else _NO_PAGE,
                }
                for chunk in chunks
            ],
        )

    def query(self, embedding: Sequence[float], k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=str(meta["doc_id"]),
                doc_name=str(meta["doc_name"]),
                text=document,
                # cosine space: distance = 1 - similarity.
                score=1.0 - float(distance),
                page=_page_from(meta),
            )
            for chunk_id, document, meta, distance in _iter_hits(result)
        ]

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})


def _page_from(meta: dict[str, Any]) -> int | None:
    page = meta.get("page", _NO_PAGE)
    return None if page == _NO_PAGE else int(page)


def _iter_hits(result: Any) -> list[tuple[str, str, dict[str, Any], float]]:  # noqa: ANN401
    """Flatten Chroma's per-query nested lists for a single query embedding."""
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return list(zip(ids, documents, metadatas, distances, strict=True))
