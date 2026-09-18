"""Document listing and deletion (Design.md §1.1).

Deletion removes the vectors first, then the rows: a leftover vector is
harmless (it can no longer be resolved to a document and is overwritten on
re-ingest), whereas a leftover row pointing at deleted vectors would surface a
citation that cannot be opened.
"""

from __future__ import annotations

from sqlalchemy import func, select

from app.db.models import Chunk, Document
from app.services.ingestion import IngestResult
from app.services.runtime import Runtime


def list_documents(runtime: Runtime) -> list[IngestResult]:
    with runtime.db.session() as session:
        count_rows = session.execute(
            select(Chunk.doc_id, func.count(Chunk.chunk_id)).group_by(Chunk.doc_id)
        ).all()
        counts: dict[str, int] = {doc_id: int(total) for doc_id, total in count_rows}
        documents = session.execute(select(Document).order_by(Document.created_at)).scalars().all()
        return [
            IngestResult(
                doc_id=document.doc_id,
                name=document.name,
                chunks=int(counts.get(document.doc_id, 0)),
                status=document.status,
            )
            for document in documents
        ]


def delete_document(runtime: Runtime, doc_id: str) -> bool:
    """Delete a document and its chunks/vectors. Returns False if it did not exist."""
    with runtime.db.session() as session:
        document = session.get(Document, doc_id)
        if document is None:
            return False

    runtime.vector_store.delete_document(doc_id)
    with runtime.db.session() as session:
        document = session.get(Document, doc_id)
        if document is None:
            return False
        session.delete(document)  # chunks cascade
    return True
