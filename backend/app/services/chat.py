"""Chat: retrieve → generate → ground → route (answer / flag / fallback / refuse).

The full query path (Architecture.md §4, §3.5, §3.6):

1. Embed the question, retrieve top-k chunks.
2. Generate a *structured* answer and pass it through the grounding hard gate
   (`rag/grounding.py`): a fabricated or unsupported citation can never reach
   the response. Malformed JSON is repaired once (Design.md §5), then a 502;
   grounding failure regenerates once (Phase 2 budget).
3. **Confidence routing** (`app/confidence`, Phase 3), which runs *only after*
   grounding passes: score = w·retrieval + w·faithfulness + w·coverage.
   * `>= high` → answer as-is.
   * `[low, high)` → answer with a visible low-confidence badge (flagged).
   * `< high` with no corpus grounding, or `< low` → **web fallback**: live
     search results re-enter generation → grounding → confidence, labelled
     `source="web"` with live URLs (Rule 5).
4. If nothing grounds anywhere (and fallback is unavailable or also fails), the
   honest refusal card (Rule 6). Every routing decision is persisted (Rule 7).
"""

from __future__ import annotations

import time
from uuid import uuid4

from starlette.status import HTTP_400_BAD_REQUEST, HTTP_502_BAD_GATEWAY

from app.confidence.router import RouterDecision, route
from app.confidence.scoring import score_confidence
from app.core.errors import AxiomError, ErrorCode
from app.db.models import Document, Message, Session
from app.rag.generation import MalformedStructuredAnswerError, parse_structured_answer
from app.rag.grounding import GroundingOutcome, verify_and_ground
from app.rag.prompts import build_structured_system_prompt
from app.rag.types import ChatResult, ConfidenceBreakdown, RetrievedChunk
from app.search.provider import web_results_to_chunks
from app.services.retrieval import retrieve
from app.services.runtime import Runtime

NO_EVIDENCE_ANSWER = (
    "I couldn't find anything about that in the ingested documents or on the web, so I "
    "won't guess. Try rephrasing, or add a document that covers it."
)
LOW_CONFIDENCE_WARNING = "Sources are weak — verify before relying on this."
WEB_FALLBACK_NOTE = "Answered via live web search — verify against the linked sources."


def answer_question(
    runtime: Runtime, *, question: str, session_id: str | None = None
) -> ChatResult:
    """Answer one question against the corpus, with web fallback, and persist it."""
    question, session_id = prepare_question(runtime, question=question, session_id=session_id)
    return answer_prepared(runtime, question=question, session_id=session_id)


def prepare_question(runtime: Runtime, *, question: str, session_id: str | None) -> tuple[str, str]:
    """Run the cheap up-front guards and resolve the session.

    Split out from `answer_prepared` so the streaming endpoint can surface these
    as normal HTTP 4xx *before* the response body starts (Design.md §5), then
    stream the LLM pipeline. Returns the trimmed question and the session id.
    """
    question = question.strip()
    if not question:
        raise AxiomError(
            ErrorCode.VALIDATION_ERROR,
            "question must not be empty.",
            status_code=HTTP_400_BAD_REQUEST,
        )
    _require_ingested_documents(runtime)
    return question, _ensure_session(runtime, session_id)


def answer_prepared(runtime: Runtime, *, question: str, session_id: str) -> ChatResult:
    """Retrieve → generate → ground → route → persist a *prepared* question.

    Assumes `prepare_question` already validated the question, confirmed a
    corpus exists, and resolved `session_id`.
    """
    started = time.perf_counter()
    result = _route_answer(runtime, question, session_id)
    _persist(runtime, question, result, _elapsed_ms(started))
    return result


def _route_answer(runtime: Runtime, question: str, session_id: str) -> ChatResult:
    """Run retrieval → grounding → confidence routing, falling back to web."""
    retrieved = retrieve(runtime, query=question)
    retrieved_ids = tuple(chunk.chunk_id for chunk in retrieved)

    kb_result: ChatResult | None = None
    if retrieved:
        outcome = _generate_grounded(runtime, question, retrieved)
        if outcome.grounded:
            score, breakdown = _score(runtime, retrieved, outcome)
            decision = route(score, runtime.config.confidence.thresholds)
            if decision is not RouterDecision.FALLBACK:
                return _grounded_result(
                    outcome, breakdown, session_id, retrieved_ids, decision, fallback_used=False
                )
            # score < low: keep this grounded-but-weak answer as the honest
            # floor if the web fallback cannot do better.
            kb_result = _grounded_result(
                outcome,
                breakdown,
                session_id,
                retrieved_ids,
                RouterDecision.FLAG,
                fallback_used=False,
            )

    web_result = _try_web_fallback(runtime, question, session_id, retrieved_ids)
    if web_result is not None:
        return web_result
    if kb_result is not None:
        return kb_result
    return _insufficient(session_id, retrieved_ids)


def _generate_grounded(
    runtime: Runtime, question: str, retrieved: list[RetrievedChunk]
) -> GroundingOutcome:
    """Generate a structured answer and run the grounding gate, with retries.

    Malformed JSON is repaired once then surfaced as a 502; a grounding failure
    regenerates once (Design.md §5, Phases.md Phase 2 budgets). Returns the
    grounding outcome — `grounded=False` when nothing survived after the retry.
    """
    system_prompt = build_structured_system_prompt(retrieved)
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
        if outcome.grounded or regen_budget <= 0:
            return outcome
        regen_budget -= 1  # Phases.md Phase 2: regenerate once on grounding failure.


def _try_web_fallback(
    runtime: Runtime,
    question: str,
    session_id: str,
    retrieved_ids: tuple[str, ...],
) -> ChatResult | None:
    """Attempt a live web-search answer; None if unavailable or ungrounded.

    Web results re-enter the same generation → grounding → confidence path
    (Architecture.md §3.6). A grounded web answer is labelled `fallback_used`
    with `source="web"` citations; anything else returns None so the caller can
    fall through to the weak-corpus floor or the honest refusal.
    """
    if not runtime.config.fallback.enabled:
        return None
    try:
        results = runtime.web_search.search(
            question, max_results=runtime.config.fallback.max_results
        )
    except AxiomError:
        raise  # budget/availability errors are structured; let them surface.
    except Exception:
        # A flaky search API must not 500 the chat; fall through to refusal.
        return None

    web_chunks = web_results_to_chunks(results)
    if not web_chunks:
        return None

    outcome = _generate_grounded(runtime, question, web_chunks)
    if not outcome.grounded:
        return None

    _, breakdown = _score(runtime, web_chunks, outcome, level="web")
    return _grounded_result(
        outcome,
        breakdown,
        session_id,
        retrieved_ids,
        RouterDecision.FALLBACK,
        fallback_used=True,
    )


def _score(
    runtime: Runtime,
    retrieved: list[RetrievedChunk],
    outcome: GroundingOutcome,
    *,
    level: str | None = None,
) -> tuple[float, ConfidenceBreakdown]:
    """Compute the hybrid confidence score and its breakdown."""
    weights = runtime.config.confidence.weights
    score, retrieval, faithfulness, coverage = score_confidence(
        weights=weights,
        retrieved=retrieved,
        used_chunk_ids=outcome.used_chunk_ids,
        stats=outcome.stats,
    )
    resolved_level = level if level is not None else _level_for(runtime, score)
    return score, ConfidenceBreakdown(
        score=round(score, 4),
        level=resolved_level,
        retrieval=round(retrieval, 4),
        faithfulness=round(faithfulness, 4),
        coverage=round(coverage, 4),
    )


def _level_for(runtime: Runtime, score: float) -> str:
    """Badge band for a corpus answer: high (green) or low (yellow)."""
    return "high" if score >= runtime.config.confidence.thresholds.high else "low"


def _grounded_result(
    outcome: GroundingOutcome,
    breakdown: ConfidenceBreakdown,
    session_id: str,
    retrieved_ids: tuple[str, ...],
    decision: RouterDecision,
    *,
    fallback_used: bool,
) -> ChatResult:
    flagged = breakdown.level == "low"
    warning = WEB_FALLBACK_NOTE if fallback_used else (LOW_CONFIDENCE_WARNING if flagged else None)
    return ChatResult(
        answer=outcome.answer,
        citations=outcome.citations,
        retrieved_chunk_ids=retrieved_ids,
        session_id=session_id,
        insufficient_evidence=False,
        used_chunk_ids=outcome.used_chunk_ids,
        confidence=breakdown,
        router_decision=decision.value,
        flagged=flagged,
        fallback_used=fallback_used,
        warning=warning,
    )


def _insufficient(session_id: str, retrieved_ids: tuple[str, ...]) -> ChatResult:
    return ChatResult(
        answer=NO_EVIDENCE_ANSWER,
        citations=(),
        retrieved_chunk_ids=retrieved_ids,
        session_id=session_id,
        insufficient_evidence=True,
        used_chunk_ids=(),
        confidence=None,
        router_decision=RouterDecision.REFUSAL.value,
        flagged=False,
        fallback_used=False,
        warning=None,
    )


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
    confidence = result.confidence
    breakdown = (
        {
            "retrieval": confidence.retrieval,
            "faithfulness": confidence.faithfulness,
            "coverage": confidence.coverage,
        }
        if confidence is not None
        else None
    )
    # answer_json is the verbatim client payload (history replays it exactly);
    # the dedicated columns below are what GET /metrics aggregates (Rule 7).
    payload = {
        "answer": result.answer,
        "insufficient_evidence": result.insufficient_evidence,
        "flagged": result.flagged,
        "fallback_used": result.fallback_used,
        "warning": result.warning,
        "confidence": (
            {
                "score": confidence.score,
                "level": confidence.level,
                "breakdown": breakdown,
            }
            if confidence is not None
            else None
        ),
        "citations": [
            {
                "id": citation.id,
                "chunk_id": citation.chunk_id,
                "doc": citation.doc,
                "page": citation.page,
                "quote": citation.text,
                "score": citation.score,
                "source": citation.source,
                "url": citation.url,
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
                confidence=confidence.score if confidence is not None else None,
                confidence_breakdown=breakdown,
                router_decision=result.router_decision,
                retrieved_chunk_ids=list(result.retrieved_chunk_ids),
                flagged=result.flagged,
                fallback_used=result.fallback_used,
                latency_ms=latency_ms,
            )
        )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
