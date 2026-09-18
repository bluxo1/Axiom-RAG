"""Chunking (Architecture.md §3.1, Design.md §2.1).

The id/page logic is tested with a trivial splitter, so these assertions pin
behaviour independent of the LlamaIndex splitter's tokenization.
"""

from __future__ import annotations

from app.rag.chunking import chunk_document, chunk_id_for, make_token_splitter
from app.rag.types import ParsedDocument, ParsedPage


def _split_on_bar(text: str) -> list[str]:
    return text.split("|")


def test_chunk_id_is_stable_and_hex() -> None:
    first = chunk_id_for("doc-abc", 3)
    assert first == chunk_id_for("doc-abc", 3)
    assert len(first) == 40
    assert first != chunk_id_for("doc-abc", 4)
    assert first != chunk_id_for("doc-xyz", 3)


def test_index_runs_across_pages_and_pages_are_preserved() -> None:
    document = ParsedDocument(
        name="doc.pdf",
        pages=(
            ParsedPage(text="a|b", page=1),
            ParsedPage(text="c", page=2),
        ),
    )

    chunks = chunk_document(document, "doc-1", _split_on_bar)

    assert [chunk.text for chunk in chunks] == ["a", "b", "c"]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.page for chunk in chunks] == [1, 1, 2]
    assert [chunk.chunk_id for chunk in chunks] == [chunk_id_for("doc-1", i) for i in range(3)]


def test_empty_and_whitespace_pieces_are_dropped() -> None:
    document = ParsedDocument(name="d", pages=(ParsedPage(text="a||  |b", page=None),))

    chunks = chunk_document(document, "doc-1", _split_on_bar)

    assert [chunk.text for chunk in chunks] == ["a", "b"]
    # Index only advances for kept chunks, so ids stay contiguous.
    assert [chunk.index for chunk in chunks] == [0, 1]


def test_token_splitter_overlaps_and_covers_long_text() -> None:
    splitter = make_token_splitter(size_tokens=32, overlap_tokens=8)
    text = " ".join(f"word{i}" for i in range(200))

    pieces = splitter(text)

    assert len(pieces) > 1
    # Every piece has content and the whole text is covered.
    assert all(piece.strip() for piece in pieces)
    assert "word0" in pieces[0]
    assert "word199" in pieces[-1]
