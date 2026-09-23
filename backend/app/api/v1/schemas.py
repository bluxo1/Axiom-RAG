"""Request/response models for the v1 API (Design.md §1).

The chat response carries `answer`, `citations`, `insufficient_evidence`, and —
from Phase 3 — the `confidence` block, `flagged`, `fallback_used`, and an
optional `warning` (Design.md §1.2), all computed by the confidence router.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.rag.types import ChatResult, Citation, ConfidenceBreakdown, RetrievedChunk


class DocumentSummary(BaseModel):
    """Ingestion result and list entry (Design.md §1.1)."""

    doc_id: str
    name: str
    chunks: int
    status: str


class DocumentList(BaseModel):
    documents: list[DocumentSummary]


class UrlIngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class CitationModel(BaseModel):
    """A source card (Design.md §1.2, §4). Web citations carry `url`, not a page."""

    id: str
    chunk_id: str
    doc: str
    page: int | None = None
    quote: str
    score: float
    source: str
    url: str | None = None

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
            url=citation.url,
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


class ConfidenceModel(BaseModel):
    """Hybrid confidence block (Design.md §1.2, Architecture.md §3.5)."""

    score: float
    level: str
    breakdown: dict[str, float]

    @classmethod
    def from_domain(cls, breakdown: ConfidenceBreakdown) -> ConfidenceModel:
        return cls(
            score=breakdown.score,
            level=breakdown.level,
            breakdown={
                "retrieval": breakdown.retrieval,
                "faithfulness": breakdown.faithfulness,
                "coverage": breakdown.coverage,
            },
        )


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    insufficient_evidence: bool
    session_id: str
    confidence: ConfidenceModel | None = None
    flagged: bool = False
    fallback_used: bool = False
    warning: str | None = None

    @classmethod
    def from_domain(cls, result: ChatResult) -> ChatResponse:
        return cls(
            answer=result.answer,
            citations=[CitationModel.from_domain(citation) for citation in result.citations],
            insufficient_evidence=result.insufficient_evidence,
            session_id=result.session_id,
            confidence=(
                ConfidenceModel.from_domain(result.confidence)
                if result.confidence is not None
                else None
            ),
            flagged=result.flagged,
            fallback_used=result.fallback_used,
            warning=result.warning,
        )


class HistoryMessage(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]
    insufficient_evidence: bool
    created_at: datetime
    confidence: ConfidenceModel | None = None
    flagged: bool = False
    fallback_used: bool = False
    warning: str | None = None

    @classmethod
    def from_stored(
        cls, *, question: str, answer_json: dict[str, Any], created_at: datetime
    ) -> HistoryMessage:
        """Rebuild a history turn from the stored verbatim payload."""
        confidence_raw = answer_json.get("confidence")
        confidence = (
            ConfidenceModel(
                score=float(confidence_raw["score"]),
                level=str(confidence_raw["level"]),
                breakdown={k: float(v) for k, v in (confidence_raw["breakdown"] or {}).items()},
            )
            if confidence_raw is not None
            else None
        )
        return cls(
            question=question,
            answer=str(answer_json.get("answer", "")),
            citations=[CitationModel(**citation) for citation in answer_json.get("citations", [])],
            insufficient_evidence=bool(answer_json.get("insufficient_evidence", False)),
            created_at=created_at,
            confidence=confidence,
            flagged=bool(answer_json.get("flagged", False)),
            fallback_used=bool(answer_json.get("fallback_used", False)),
            warning=answer_json.get("warning"),
        )


class SessionHistory(BaseModel):
    """Chat history for one session (Design.md §1.3)."""

    session_id: str
    messages: list[HistoryMessage]
