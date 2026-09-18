"""Request/response models for the v1 API (Design.md §1).

Phase 1 subset: the chat response carries `answer`, `citations`, and
`insufficient_evidence`. The `confidence`, `flagged`, and `fallback_used` fields
from Design.md §1.2 are added in Phases 2-3 by the code that computes them,
rather than shipped now as hardcoded placeholders (Rules.md §6).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.rag.types import ChatResult, Citation, RetrievedChunk


class DocumentSummary(BaseModel):
    """Ingestion result and list entry (Design.md §1.1)."""

    doc_id: str
    name: str
    chunks: int
    status: str


class DocumentList(BaseModel):
    documents: list[DocumentSummary]


class UrlIngestRequest(BaseModel):
    url: str = Field(min_length=1)


class CitationModel(BaseModel):
    """A source card (Design.md §1.2, §4)."""

    id: str
    chunk_id: str
    doc: str
    page: int | None = None
    quote: str
    score: float
    source: str

    @classmethod
    def from_domain(cls, citation: Citation) -> CitationModel:
        return cls(
            id=citation.id,
            chunk_id=citation.chunk_id,
            doc=citation.doc,
            page=citation.page,
            quote=citation.text,
            score=citation.score,
            source=citation.source,
        )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)


class RetrievedChunkModel(BaseModel):
    """One retrieval hit (Architecture.md §3.2)."""

    chunk_id: str
    doc: str
    page: int | None = None
    text: str
    score: float

    @classmethod
    def from_domain(cls, chunk: RetrievedChunk) -> RetrievedChunkModel:
        return cls(
            chunk_id=chunk.chunk_id,
            doc=chunk.doc_name,
            page=chunk.page,
            text=chunk.text,
            score=chunk.score,
        )


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunkModel]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    insufficient_evidence: bool
    session_id: str

    @classmethod
    def from_domain(cls, result: ChatResult) -> ChatResponse:
        return cls(
            answer=result.answer,
            citations=[CitationModel.from_domain(citation) for citation in result.citations],
            insufficient_evidence=result.insufficient_evidence,
            session_id=result.session_id,
        )


class HistoryMessage(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]
    insufficient_evidence: bool
    created_at: datetime


class SessionHistory(BaseModel):
    """Chat history for one session (Design.md §1.3)."""

    session_id: str
    messages: list[HistoryMessage]
