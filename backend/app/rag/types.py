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
    """A chunk returned by the vector store for a query, with its score."""

    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    score: float
    page: int | None = None


@dataclass(frozen=True)
class Citation:
    """A source shown alongside an answer.

    Phase 1 has no verifier, so every retrieved chunk used as context is offered
    as a citation with a sequential display id. The support/existence checks that
    gate these (Architecture.md §3.4) arrive in Phase 2.
    """

    id: str  # display id: c1, c2, ...
    chunk_id: str
    doc: str
    text: str
    score: float
    source: str = "kb"
    page: int | None = None


@dataclass(frozen=True)
class ChatResult:
    """The outcome of one chat turn, before HTTP serialization."""

    answer: str
    citations: tuple[Citation, ...]
    retrieved_chunk_ids: tuple[str, ...]
    session_id: str
    insufficient_evidence: bool = False
    used_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
