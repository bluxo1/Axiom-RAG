"""Citation marker rewriting (Design.md §3.3)."""

from __future__ import annotations

from app.rag.citations import assign_display_citations
from app.rag.types import RetrievedChunk


def _chunk(chunk_id: str, doc: str = "doc.pdf", page: int | None = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        doc_name=doc,
        text=f"text for {chunk_id}",
        score=0.9,
        page=page,
    )


def test_known_markers_become_sequential_display_ids() -> None:
    retrieved = [_chunk("aaa"), _chunk("bbb")]

    answer, citations, used = assign_display_citations("First [aaa]. Second [bbb].", retrieved)

    assert answer == "First [c1]. Second [c2]."
    assert [c.id for c in citations] == ["c1", "c2"]
    assert [c.chunk_id for c in citations] == ["aaa", "bbb"]
    assert used == ("aaa", "bbb")


def test_display_ids_follow_first_appearance_not_retrieval_order() -> None:
    retrieved = [_chunk("aaa"), _chunk("bbb")]

    answer, citations, _ = assign_display_citations("Only [bbb] here, then [aaa].", retrieved)

    assert answer == "Only [c1] here, then [c2]."
    assert [c.chunk_id for c in citations] == ["bbb", "aaa"]


def test_repeated_marker_reuses_the_same_display_id() -> None:
    retrieved = [_chunk("aaa")]

    answer, citations, _ = assign_display_citations("[aaa] and again [aaa].", retrieved)

    assert answer == "[c1] and again [c1]."
    assert len(citations) == 1


def test_unknown_marker_is_left_untouched() -> None:
    """Phase 1 does not strip; Phase 2's grounding gate handles fabricated ids."""
    retrieved = [_chunk("aaa")]

    answer, citations, used = assign_display_citations("Real [aaa], fake [zzz].", retrieved)

    assert answer == "Real [c1], fake [zzz]."
    assert [c.chunk_id for c in citations] == ["aaa"]
    assert "zzz" not in used


def test_citation_carries_source_metadata() -> None:
    retrieved = [_chunk("aaa", doc="policy.pdf", page=3)]
    answer, citations, _ = assign_display_citations("[aaa]", retrieved)

    assert answer == "[c1]"
    citation = citations[0]
    assert citation.doc == "policy.pdf"
    assert citation.page == 3
    assert citation.source == "kb"
    assert citation.text == "text for aaa"
