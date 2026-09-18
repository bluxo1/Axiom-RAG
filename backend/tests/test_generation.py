"""Structured-answer parsing (Design.md §3.3, rag/generation.py)."""

from __future__ import annotations

import json

import pytest

from app.rag.generation import (
    MalformedStructuredAnswerError,
    RawStructuredAnswer,
    parse_structured_answer,
)


def test_parses_a_valid_answered_payload() -> None:
    raw = json.dumps(
        {
            "status": "answered",
            "answer": "Yes. [abc]",
            "claims": [{"text": "It is covered.", "citation_ids": ["abc"]}],
            "citations": [{"citation_id": "abc", "chunk_id": "abc"}],
        }
    )

    parsed = parse_structured_answer(raw)

    assert isinstance(parsed, RawStructuredAnswer)
    assert parsed.status == "answered"
    assert parsed.claims[0].citation_ids == ("abc",)
    assert parsed.citations[0].chunk_id == "abc"


def test_parses_insufficient_evidence_payload() -> None:
    parsed = parse_structured_answer(
        '{"status": "insufficient_evidence", "answer": "", "claims": [], "citations": []}'
    )

    assert parsed.status == "insufficient_evidence"
    assert parsed.claims == ()


def test_invalid_json_is_malformed() -> None:
    with pytest.raises(MalformedStructuredAnswerError):
        parse_structured_answer("this is not json")


def test_missing_status_is_malformed() -> None:
    with pytest.raises(MalformedStructuredAnswerError):
        parse_structured_answer('{"answer": "hi", "claims": [], "citations": []}')


def test_unknown_status_value_is_malformed() -> None:
    with pytest.raises(MalformedStructuredAnswerError):
        parse_structured_answer('{"status": "maybe", "answer": "", "claims": [], "citations": []}')


def test_extra_top_level_key_is_tolerated_not_surfaced() -> None:
    """A stray field (e.g. a premature Phase 3 `confidence`) must not fail parsing."""
    parsed = parse_structured_answer(
        '{"status": "answered", "answer": "x", "claims": [], "citations": [], "confidence": 0.9}'
    )

    assert parsed.status == "answered"
    assert not hasattr(parsed, "confidence")
