"""Chat endpoint (Design.md §1.2)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import RuntimeDep
from app.api.v1.schemas import ChatRequest, ChatResponse
from app.services.chat import answer_question

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Ask a grounded question")
def chat(runtime: RuntimeDep, body: ChatRequest) -> ChatResponse:
    result = answer_question(runtime, question=body.question, session_id=body.session_id)
    return ChatResponse.from_domain(result)
