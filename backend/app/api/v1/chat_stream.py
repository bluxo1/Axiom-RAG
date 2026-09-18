"""Streaming chat endpoint (Design.md §1.2, Architecture.md §112, PRD FR-9).

Server-Sent Events, in this strict order (the grounding hard gate is never
bypassed for latency):

1. `status` — emitted immediately ("grounding"), so the UI can show activity
   while retrieval + generation + verification run.
2. `token` — the *verified* answer, streamed piece by piece. Answer tokens are
   only ever sent **after** verification passes; a grounding failure and its
   one retry complete first, so a fabricated citation can never reach the wire.
3. `done` — the full structured payload (citations, confidence, flags), byte
   identical to what `POST /chat` returns, so the client renders the two the
   same way.

Cheap guards (empty question, no corpus) are run before the stream opens and
surface as normal HTTP 4xx. Once the body is streaming the status line has
already been sent, so a pipeline error (e.g. a 502 upstream) is delivered as an
`error` event rather than an HTTP status.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter
from starlette.responses import StreamingResponse

from app.api.deps import RuntimeDep
from app.api.v1.schemas import ChatRequest, ChatResponse
from app.core.errors import AxiomError
from app.rag.types import ChatResult
from app.services.chat import answer_prepared, prepare_question
from app.services.runtime import Runtime

router = APIRouter(tags=["chat"])

# Roughly a word at a time — enough to look live without shipping one byte per
# event. The answer is already complete and verified before we chunk it.
_TOKEN_CHUNK = 24


def _sse(event: str, data: object) -> str:
    """Format one SSE frame: named event + JSON data payload."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _iter_answer_tokens(answer: str) -> Iterator[str]:
    """Yield the verified answer in small, word-boundary-ish chunks."""
    for start in range(0, len(answer), _TOKEN_CHUNK):
        yield answer[start : start + _TOKEN_CHUNK]


def _stream(runtime: Runtime, question: str, session_id: str) -> Iterator[str]:
    # The status event goes out first, before the (blocking) pipeline runs.
    yield _sse("status", {"state": "grounding"})
    try:
        result: ChatResult = answer_prepared(runtime, question=question, session_id=session_id)
    except AxiomError as exc:
        yield _sse("error", {"code": exc.code.value, "message": exc.message})
        return

    # Verification has passed by now; only verified tokens are streamed.
    for piece in _iter_answer_tokens(result.answer):
        yield _sse("token", {"text": piece})
    yield _sse("done", ChatResponse.from_domain(result).model_dump())


@router.post("/chat/stream", summary="Ask a grounded question (SSE stream)")
def chat_stream(runtime: RuntimeDep, body: ChatRequest) -> StreamingResponse:
    # Cheap guards first, so empty-question / no-corpus stay real HTTP 4xx.
    question, session_id = prepare_question(
        runtime, question=body.question, session_id=body.session_id
    )
    return StreamingResponse(
        _stream(runtime, question, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
