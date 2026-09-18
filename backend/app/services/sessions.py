"""Chat history retrieval (Design.md §1.3, GET /sessions/{id}).

History replays the exact structured payload each turn returned (stored in
`messages.answer_json`), so the UI renders past answers and their citation cards
identically to when they were first shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.db.models import Session
from app.services.runtime import Runtime


@dataclass(frozen=True)
class HistoryTurn:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    insufficient_evidence: bool
    created_at: datetime


def get_history(runtime: Runtime, session_id: str) -> list[HistoryTurn] | None:
    """Return a session's turns oldest-first, or None if the session is unknown."""
    with runtime.db.session() as session:
        record = session.get(Session, session_id)
        if record is None:
            return None
        return [
            HistoryTurn(
                question=message.question,
                answer=str(message.answer_json.get("answer", "")),
                citations=list(message.answer_json.get("citations", [])),
                insufficient_evidence=bool(message.answer_json.get("insufficient_evidence", False)),
                created_at=message.created_at,
            )
            for message in record.messages
        ]
