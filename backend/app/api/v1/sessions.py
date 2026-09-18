"""Chat-history endpoint (Design.md §1.3, GET /sessions/{id})."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.status import HTTP_404_NOT_FOUND

from app.api.deps import RuntimeDep
from app.api.v1.schemas import CitationModel, HistoryMessage, SessionHistory
from app.core.errors import AxiomError, ErrorCode
from app.services.sessions import get_history

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/{session_id}", response_model=SessionHistory, summary="Chat history for a session")
def session_history(runtime: RuntimeDep, session_id: str) -> SessionHistory:
    turns = get_history(runtime, session_id)
    if turns is None:
        raise AxiomError(
            ErrorCode.NOT_FOUND,
            f"session '{session_id}' not found.",
            status_code=HTTP_404_NOT_FOUND,
        )
    return SessionHistory(
        session_id=session_id,
        messages=[
            HistoryMessage(
                question=turn.question,
                answer=turn.answer,
                citations=[CitationModel.model_validate(citation) for citation in turn.citations],
                insufficient_evidence=turn.insufficient_evidence,
                created_at=turn.created_at,
            )
            for turn in turns
        ],
    )
