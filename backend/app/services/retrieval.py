"""Retrieval: embed a query and fetch the top-k chunks (Architecture.md §3.2).

Shared by the `/search` endpoint (Phases.md Phase 1: "Retrieval endpoint:
top-k semantic search") and the chat service, so both rank identically. `k`
defaults to `config.retrieval.top_k` and a caller override is clamped to it, so
the endpoint can never ask for an unbounded result set.
"""

from __future__ import annotations

from app.rag.types import RetrievedChunk
from app.services.runtime import Runtime


def retrieve(runtime: Runtime, *, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    configured = runtime.config.retrieval.top_k
    k = configured if top_k is None else max(1, min(top_k, configured))
    embedding = runtime.embedder.embed_query(query)
    return runtime.vector_store.query(embedding, k)
