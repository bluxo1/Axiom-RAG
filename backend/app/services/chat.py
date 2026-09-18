"""Chat: retrieve -> structured generation -> grounding gate -> persist (Phase 2).

The query path (Architecture.md §4):

1. Embed the question, retrieve top-k chunks. Empty retrieval -> honest refusal.
2. Prompt the model for a *structured* answer (`{status, answer, claims,
   citations}`) and parse it. Malformed JSON is retried once (repair budget,
   Design.md §5), then surfaced as a 502.
3. **Grounding is a hard gate** (`rag/grounding.py`): every citation must exist
   in the retrieved set and its chunk must support the claim. Unsupported claims
   are stripped and the answer is rebuilt from the survivors, so a fabricated
   citation can never reach the response. Grounding failure regenerates once
   (Phases.md Phase 2: max 1 retry), then returns the refusal card.

Confidence scoring, flagging, and web fallback are Phase 3 — not here.
"""

from __future__ import annotations

import time
from uuid import uuid4

from starlette.status import HTTP_400_BAD_REQUEST, HTTP_502_BAD_GATEWAY

from app.core.errors import AxiomError, ErrorCode
from app.db.models import Document, Message, Session
from app.rag.generation import MalformedStructuredAnswerError, parse_structured_answer
from app.rag.grounding import verify_and_ground
from app.rag.prompts import build_structured_system_prompt
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

    system_prompt = build_structured_system_prompt(retrieved)
    retrieved_ids = tuple(chunk.chunk_id for chunk in retrieved)
    support_min = runtime.config.grounding.support_keyword_overlap_min
    repair_budget = runtime.config.generation.max_repair_attempts
    regen_budget = runtime.config.grounding.max_regenerations

    while True:
        raw = runtime.llm.complete_structured(system_prompt=system_prompt, user_prompt=question)
        try:
            parsed = parse_structured_answer(raw)
        except MalformedStructuredAnswerError as exc:
            if repair_budget > 0:
                repair_budget -= 1
                continue  # Design.md §5: one auto-repair pass on malformed JSON.
            raise AxiomError(
                ErrorCode.INVALID_LLM_RESPONSE,
                "The language model returned a malformed response.",
                status_code=HTTP_502_BAD_GATEWAY,
            ) from exc

        outcome = verify_and_ground(parsed, retrieved, support_keyword_overlap_min=support_min)
        if outcome.grounded:
            result = ChatResult(
                answer=outcome.answer,
                citations=outcome.citations,
                retrieved_chunk_ids=retrieved_ids,
                session_id=session_id,
                insufficient_evidence=False,
                used_chunk_ids=outcome.used_chunk_ids,
            )
            _persist(runtime, question, result, _elapsed_ms(started))
            return result

        if regen_budget > 0:
            regen_budget -= 1
            continue  # Phases.md Phase 2: regenerate once on grounding failure.
        return _persist_insufficient(runtime, question, session_id, started, retrieved_ids)


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
