"""Structured generation output: schema + parsing (Design.md §3.3).

The Phase 2 generation call returns JSON, not prose:

```json
{"status": "answered" | "insufficient_evidence",
 "answer": "...",
 "claims": [{"text": "...", "citation_ids": ["<raw chunk marker>", ...]}],
 "citations": [{"citation_id": "<raw chunk marker>", "chunk_id": "<retrieved id>"}]}
```

At generation time the model's `citation_ids` are raw markers keyed to the
`[chunk_id]` tokens in the context; the grounding gate (`rag/grounding.py`)
rewrites the *verified* ones to display ids `c1, c2, ...`.

These are pydantic models, not the frozen dataclasses in `rag/types.py`, because
this is untrusted model output that must be *validated* at the boundary. A
payload that does not match the schema raises `MalformedStructuredAnswerError`, which
the chat service maps to one repair pass and then a 502 (Design.md §5) — never a
silent 500, never raw output reaching the verifier.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class MalformedStructuredAnswerError(Exception):
    """The LLM returned something that is not a valid structured answer.

    A plain exception, not an `AxiomError`: this module is pure and free of HTTP
    concerns. The chat service catches it, spends one repair attempt, and only
    then raises the 502 envelope.
    """


class RawClaim(BaseModel):
    """One factual claim and the raw citation markers the model attached."""

    model_config = ConfigDict(extra="ignore")

    text: str
    citation_ids: tuple[str, ...] = ()


class RawCitation(BaseModel):
    """A raw marker -> retrieved chunk_id mapping, as emitted by the model."""

    model_config = ConfigDict(extra="ignore")

    citation_id: str
    chunk_id: str


class RawStructuredAnswer(BaseModel):
    """The model's structured answer, before grounding.

    `extra="ignore"` tolerates stray keys a model might emit (e.g. a spurious
    `confidence`) without failing the parse — they are never read. The Phase 3
    fields (confidence/flagged/fallback) are deliberately absent here.
    """

    model_config = ConfigDict(extra="ignore")

    status: Literal["answered", "insufficient_evidence"]
    answer: str = ""
    claims: tuple[RawClaim, ...] = ()
    citations: tuple[RawCitation, ...] = ()


def parse_structured_answer(raw: str) -> RawStructuredAnswer:
    """Parse and validate the raw JSON string from the LLM.

    Raises `MalformedStructuredAnswerError` on invalid JSON or a payload that does
    not match the schema.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise MalformedStructuredAnswerError(f"response is not valid JSON: {exc}") from exc
    try:
        return RawStructuredAnswer.model_validate(data)
    except ValidationError as exc:
        raise MalformedStructuredAnswerError(f"response does not match the schema: {exc}") from exc
