"""Ingestion: parse -> chunk -> embed -> store (Architecture.md §3.1).

The document row is created `processing`, then flipped to `ready` once its
chunks are embedded, written to the vector store, and persisted — or to `failed`
if any step raises. Vectors are written before the chunk rows so that a crash
leaves recoverable orphan vectors rather than chunk rows pointing at vectors
that were never stored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from app.db.models import Chunk, Document
from app.rag.chunking import chunk_document, make_token_splitter
from app.rag.parsing import enforce_upload_limit, parse_upload, parse_url
from app.rag.types import ParsedDocument, TextChunk
from app.services.runtime import Runtime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    """What `POST /documents` returns (Design.md §1.1)."""

    doc_id: str
    name: str
    chunks: int
    status: str


def ingest_upload(runtime: Runtime, *, name: str, data: bytes) -> IngestResult:
    """Ingest an uploaded file (PDF/TXT/MD), size-capped (PRD.md §8)."""
    enforce_upload_limit(data, max_upload_mb=runtime.config.ingestion.max_upload_mb)
    return _ingest(runtime, parse_upload(name, data))


def ingest_from_url(runtime: Runtime, *, url: str) -> IngestResult:
    """Ingest the main content of a URL (Design.md §1.1)."""
    return _ingest(runtime, parse_url(url))


def _ingest(runtime: Runtime, parsed: ParsedDocument) -> IngestResult:
    doc_id = str(uuid4())
    config = runtime.config

    with runtime.db.session() as session:
        session.add(Document(doc_id=doc_id, name=parsed.name, status="processing"))

    try:
        splitter = make_token_splitter(config.chunking.size_tokens, config.chunking.overlap_tokens)
        chunks = chunk_document(parsed, doc_id, splitter)
        if not chunks:
            raise _empty(parsed.name)

        embeddings = runtime.embedder.embed_texts([chunk.text for chunk in chunks])
        runtime.vector_store.add(chunks, embeddings)
        _persist_chunks(runtime, doc_id, chunks)
    except Exception:
        _mark_failed(runtime, doc_id)
        logger.exception("ingestion_failed doc_id=%s name=%s", doc_id, parsed.name)
        raise

    return IngestResult(doc_id=doc_id, name=parsed.name, chunks=len(chunks), status="ready")


def _persist_chunks(runtime: Runtime, doc_id: str, chunks: list[TextChunk]) -> None:
    with runtime.db.session() as session:
        session.add_all(
            Chunk(chunk_id=chunk.chunk_id, doc_id=doc_id, page=chunk.page, text=chunk.text)
            for chunk in chunks
        )
        document = session.get(Document, doc_id)
        if document is not None:
            document.status = "ready"


def _mark_failed(runtime: Runtime, doc_id: str) -> None:
    try:
        with runtime.db.session() as session:
            document = session.get(Document, doc_id)
            if document is not None:
                document.status = "failed"
    except Exception:
        logger.exception("could not mark doc_id=%s failed", doc_id)


def _empty(name: str) -> ValueError:
    from app.rag.parsing import ParsingError

    return ParsingError(f"'{name}' produced no chunks")
