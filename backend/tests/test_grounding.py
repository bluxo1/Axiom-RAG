"""Citation grounding — the hard gate (Architecture.md §3.4, rag/grounding.py).

The money test (Phases.md Phase 2): a fabricated chunk_id can never survive
verification, so it can never reach the answer or the citation cards.
"""

from __future__ import annotations

from app.rag.generation import RawCitation, RawClaim, RawStructuredAnswer
from app.rag.grounding import GroundingOutcome, verify_and_ground
from app.rag.types import RetrievedChunk

SUPPORT_MIN = 0.3  # config.yaml grounding.support_keyword_overlap_min


def _chunk(
    chunk_id: str, text: str, *, doc: str = "doc.pdf", page: int | None = 1
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id="doc-1", doc_name=doc, text=text, score=0.9, page=page
    )


def _answer(
    claims: list[tuple[str, list[str]]],
    *,
    status: str = "answered",
) -> RawStructuredAnswer:
    """Build a raw structured answer; citation markers are the chunk_ids."""
    cited: list[str] = []
    for _text, ids in claims:
        for cid in ids:
            if cid not in cited:
                cited.append(cid)
    return RawStructuredAnswer(
        status=status,  # type: ignore[arg-type]
        answer="model prose that is never displayed",
        claims=tuple(RawClaim(text=text, citation_ids=tuple(ids)) for text, ids in claims),
        citations=tuple(RawCitation(citation_id=cid, chunk_id=cid) for cid in cited),
    )


def _ground(answer: RawStructuredAnswer, retrieved: list[RetrievedChunk]) -> GroundingOutcome:
    return verify_and_ground(answer, retrieved, support_keyword_overlap_min=SUPPORT_MIN)


def test_valid_citation_passes_and_gets_display_id() -> None:
    retrieved = [_chunk("aaa", "The policy covers flooding and water damage.")]
    answer = _answer([("The policy covers flooding.", ["aaa"])])

    outcome = _ground(answer, retrieved)

    assert outcome.grounded is True
    assert outcome.answer == "The policy covers flooding. [c1]"
    assert [c.id for c in outcome.citations] == ["c1"]
    assert outcome.citations[0].chunk_id == "aaa"
    assert outcome.used_chunk_ids == ("aaa",)


def test_fabricated_chunk_id_is_stripped() -> None:
    retrieved = [_chunk("aaa", "The policy covers flooding and water damage.")]
    answer = _answer([("The policy covers flooding.", ["zzz"])])

    outcome = _ground(answer, retrieved)

    assert outcome.grounded is False  # only claim had a non-existent citation
    assert "zzz" not in outcome.answer


def test_all_fabricated_yields_refusal() -> None:
    retrieved = [_chunk("aaa", "Warranty lasts twelve months.")]
    answer = _answer([("Made up.", ["nope"]), ("Also made up.", ["nan"])])

    outcome = _ground(answer, retrieved)

    assert outcome.grounded is False
    assert outcome.citations == ()


def test_existing_chunk_that_does_not_support_the_claim_is_rejected() -> None:
    retrieved = [_chunk("aaa", "The warranty lasts twelve months from purchase.")]
    answer = _answer([("Zebras orbit distant purple nebulae nightly.", ["aaa"])])

    outcome = _ground(answer, retrieved)

    assert outcome.grounded is False  # exists, but keyword overlap < 0.3


def test_support_check_accepts_partial_overlap_at_threshold() -> None:
    # claim keywords: {flooding, damage, coverage} ; chunk contains flooding+damage
    # -> 2/3 = 0.67 >= 0.3
    retrieved = [_chunk("aaa", "Flooding and storm damage are handled here.")]
    answer = _answer([("flooding damage coverage", ["aaa"])])

    outcome = _ground(answer, retrieved)

    assert outcome.grounded is True


def test_insufficient_evidence_status_is_refusal_regardless_of_claims() -> None:
    retrieved = [_chunk("aaa", "The policy covers flooding.")]
    answer = _answer([("The policy covers flooding.", ["aaa"])], status="insufficient_evidence")

    outcome = _ground(answer, retrieved)

    assert outcome.grounded is False


def test_partially_fabricated_claim_keeps_valid_citation_drops_fake() -> None:
    retrieved = [_chunk("aaa", "The policy covers flooding and water damage.")]
    answer = _answer([("The policy covers flooding.", ["zzz", "aaa"])])

    outcome = _ground(answer, retrieved)

    assert outcome.grounded is True
    assert [c.chunk_id for c in outcome.citations] == ["aaa"]
    assert outcome.answer == "The policy covers flooding. [c1]"


def test_display_ids_are_sequential_in_first_cited_order() -> None:
    retrieved = [
        _chunk("aaa", "Flooding coverage is included in the policy."),
        _chunk("bbb", "The warranty period lasts twelve months."),
    ]
    answer = _answer(
        [
            ("Flooding coverage is included.", ["bbb", "aaa"]),  # bbb cited first
            ("The warranty lasts twelve months.", ["bbb"]),
        ]
    )

    outcome = _ground(answer, retrieved)

    # 'bbb' supports the warranty claim, 'aaa' supports flooding; 'bbb' on the
    # flooding claim shares no keywords -> dropped there, kept on warranty.
    assert outcome.grounded is True
    ids = {c.chunk_id: c.id for c in outcome.citations}
    assert ids["aaa"] == "c1"
    assert ids["bbb"] == "c2"


def test_displayed_answer_uses_claim_text_not_model_prose() -> None:
    retrieved = [_chunk("aaa", "The policy covers flooding and water damage.")]
    answer = _answer([("The policy covers flooding.", ["aaa"])])

    outcome = _ground(answer, retrieved)

    assert "model prose" not in outcome.answer
    assert outcome.answer.startswith("The policy covers flooding.")
