"""Chat: retrieve -> grounded generation -> persist (Phases.md Phase 1).

Phase 1 deliberately has *no* verification, confidence scoring, or web fallback
(those are Phases 2-3). The path is: embed the question, retrieve top-k, prompt
the model to answer only from that context, rewrite the citation markers it
emitted for retrieved chunks, and persist the turn. An empty retrieval or an
`INSUFFICIENT_EVIDENCE` reply yields an honest "not in the documents" answer,
never an ungrounded guess (CLAUDE.md: INSUFFICIENT_EVIDENCE over improvisation).
"""

from __future__ import annotations

import time
from uuid import uuid4

from starlette.status import HTTP_400_BAD_REQUEST

from app.core.errors import AxiomError, ErrorCode
from app.db.models import Document, Message, Session
from app.rag.citations import assign_display_citations
from app.rag.prompts import INSUFFICIENT_EVIDENCE, build_system_prompt
from app.rag.types import ChatResult
from app.services.retrieval import retrieve
from app.services.runtime import Runtime

NO_EVIDENCE_ANSWER = (
    "I couldn't find anything about that in the ingested documents, so I won't guess. "
    "Try rephrasing, or add a document that covers it."
)


def answer_question(
    runtime: Runtime, *, question: str, session_id: str | None = None
) -> ChatResult:
    """Answer one question against the ingested corpus and persist the turn."""
    question = question.strip()
    if not question:
        raise AxiomError(
            ErrorCode.VALIDATION_ERROR,
            "question must not be empty.",
            status_code=HTTP_400_BAD_REQUEST,
        )

    _require_ingested_documents(runtime)
    session_id = _ensure_session(runtime, session_id)

    started = time.perf_counter()
    retrieved = retrieve(runtime, query=question)

    if not retrieved:
        return _persist_insufficient(runtime, question, session_id, started, retrieved_ids=())

    system_prompt = build_system_prompt(retrieved)
    raw_answer = runtime.llm.complete(system_prompt=system_prompt, user_prompt=question)
    retrieved_ids = tuple(chunk.chunk_id for chunk in retrieved)

    if raw_answer.strip() == INSUFFICIENT_EVIDENCE:
        return _persist_insufficient(runtime, question, session_id, started, retrieved_ids)

    answer, citations, used_ids = assign_display_citations(raw_answer, retrieved)
    result = ChatResult(
        answer=answer,
        citations=citations,
        retrieved_chunk_ids=retrieved_ids,
        session_id=session_id,
        insufficient_evidence=False,
        used_chunk_ids=used_ids,
    )
    _persist(runtime, question, result, _elapsed_ms(started))
    return result


def _persist_insufficient(
    runtime: Runtime,
    question: str,
    session_id: str,
    started: float,
    retrieved_ids: tuple[str, ...],
) -> ChatResult:
    result = ChatResult(
        answer=NO_EVIDENCE_ANSWER,
        citations=(),
        retrieved_chunk_ids=retrieved_ids,
        session_id=session_id,
        insufficient_evidence=True,
        used_chunk_ids=(),
    )
    _persist(runtime, question, result, _elapsed_ms(started))
    return result


def _require_ingested_documents(runtime: Runtime) -> None:
    """Design.md §5: no docs ingested -> 400 with a friendly CTA."""
    with runtime.db.session() as session:
        ready = session.query(Document).filter(Document.status == "ready").first()
    if ready is None:
        raise AxiomError(
            ErrorCode.VALIDATION_ERROR,
            "No documents have been ingested yet.",
            status_code=HTTP_400_BAD_REQUEST,
            details={"cta": "Upload a document before asking a question."},
        )


def _ensure_session(runtime: Runtime, session_id: str | None) -> str:
    with runtime.db.session() as session:
        if session_id is None:
            new_id = str(uuid4())
            session.add(Session(session_id=new_id))
            return new_id
        if session.get(Session, session_id) is None:
            session.add(Session(session_id=session_id))
    return session_id


def _persist(runtime: Runtime, question: str, result: ChatResult, latency_ms: int) -> None:
    payload = {
        "answer": result.answer,
        "insufficient_evidence": result.insufficient_evidence,
        "citations": [
            {
                "id": citation.id,
                "chunk_id": citation.chunk_id,
                "doc": citation.doc,
                "page": citation.page,
                "quote": citation.text,
                "score": citation.score,
                "source": citation.source,
            }
            for citation in result.citations
        ],
    }
    with runtime.db.session() as session:
        session.add(
            Message(
                msg_id=str(uuid4()),
                session_id=result.session_id,
                question=question,
                answer_json=payload,
                retrieved_chunk_ids=list(result.retrieved_chunk_ids),
                latency_ms=latency_ms,
            )
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
