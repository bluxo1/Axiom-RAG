"""Retrieval endpoint (Phases.md Phase 1: top-k semantic search).

Exposes the retrieval subsystem directly: embed a query, return the top-k chunks
and their similarity scores. Useful on its own and as the inspection surface the
Phase 4 eval harness reads. Generation and grounding stay in `/chat`.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import RuntimeDep
from app.api.v1.schemas import RetrievedChunkModel, SearchRequest, SearchResponse
from app.services.retrieval import retrieve

router = APIRouter(tags=["retrieval"])


@router.post("/search", response_model=SearchResponse, summary="Top-k semantic search")
def search(runtime: RuntimeDep, body: SearchRequest) -> SearchResponse:
    hits = retrieve(runtime, query=body.query, top_k=body.top_k)
    return SearchResponse(
        query=body.query,
        results=[RetrievedChunkModel.from_domain(hit) for hit in hits],
    )
