"""Domain types shared across the RAG pipeline.

Frozen dataclasses, not pydantic models: these move between pure functions and
never cross the HTTP boundary directly (the API layer maps them to response
schemas), so they need immutability and cheap construction, not validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedPage:
    """One page (PDF) or the whole body (TXT/MD/URL, page = None)."""

    text: str
    page: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Extracted text for a source, before chunking."""

    name: str
    pages: tuple[ParsedPage, ...]


@dataclass(frozen=True)
class TextChunk:
    """A chunk ready to embed and store.

    `chunk_id` is stable: `sha1(doc_id:index)` (Design.md §2.1), so re-ingesting
    the same document yields the same ids.
    """

    chunk_id: str
    doc_id: str
    doc_name: str
    index: int
    text: str
    page: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk offered as grounding context, with its retrieval score.

    Usually a chunk from the ingested corpus (`source="kb"`). A web-fallback
    result (Architecture.md §3.6) is the same shape with `source="web"` and a
    live `url`, so it re-enters generation → grounding → confidence identically.
    """

    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    score: float
    page: int | None = None
    source: str = "kb"
    url: str | None = None


@dataclass(frozen=True)
class Citation:
    """A verified source shown alongside an answer (Design.md §1.2).

    Only citations that passed the grounding gate (Architecture.md §3.4) become
    a `Citation`; each carries a sequential display id (`c1, c2, ...`). A web
    citation (`source="web"`) carries the live `url` instead of a page.
    """

    id: str  # display id: c1, c2, ...
    chunk_id: str
    doc: str
    text: str
    score: float
    source: str = "kb"
    page: int | None = None
    url: str | None = None


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """The three hybrid-confidence components and their weighted total.

    `score` is `w_retrieval·retrieval + w_faithfulness·faithfulness +
    w_coverage·coverage` (Architecture.md §3.5); `level` is the routed band.
    """

    score: float
    level: str  # "high" | "low" | "web"
    retrieval: float
    faithfulness: float
    coverage: float


@dataclass(frozen=True)
class ChatResult:
    """The outcome of one chat turn, before HTTP serialization."""

    answer: str
    citations: tuple[Citation, ...]
    retrieved_chunk_ids: tuple[str, ...]
    session_id: str
    insufficient_evidence: bool = False
    used_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    confidence: ConfidenceBreakdown | None = None
    router_decision: str = "refusal"  # answer | flag | fallback | refusal
    flagged: bool = False
    fallback_used: bool = False
    warning: str | None = None
