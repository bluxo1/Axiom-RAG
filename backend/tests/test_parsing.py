"""Source parsing (Architecture.md §3.1, Design.md §1.1)."""

from __future__ import annotations

import pytest

from app.rag.parsing import ParsingError, parse_pdf, parse_text, parse_upload


def _minimal_pdf(text: str) -> bytes:
    """Assemble a valid single-page PDF with extractable text.

    Offsets in the xref table are computed so pypdf parses it without a rebuild.
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("latin-1")
    objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)

    xref_pos = len(out)
    size = len(objects) + 1
    out += b"xref\n0 %d\n" % size
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (size, xref_pos)
    return bytes(out)


# ─── Text / Markdown ──────────────────────────────────────────────────────────


def test_parse_text_decodes_and_has_no_page() -> None:
    document = parse_text(b"# Title\n\nBody text.", "notes.md")

    assert document.name == "notes.md"
    assert len(document.pages) == 1
    assert document.pages[0].page is None
    assert "Body text." in document.pages[0].text


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ParsingError, match="no text"):
        parse_text(b"   \n  ", "empty.txt")


# ─── Dispatch ─────────────────────────────────────────────────────────────────


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(ParsingError, match="unsupported file type"):
        parse_upload("archive.zip", b"data")


def test_upload_dispatches_markdown_to_text() -> None:
    document = parse_upload("readme.md", b"hello world")

    assert document.pages[0].text == "hello world"


# ─── PDF ──────────────────────────────────────────────────────────────────────


def test_parse_pdf_extracts_text_with_page_numbers() -> None:
    document = parse_pdf(_minimal_pdf("Axiom test page"), "doc.pdf")

    assert len(document.pages) == 1
    assert document.pages[0].page == 1
    assert "Axiom" in document.pages[0].text


def test_corrupt_pdf_is_rejected() -> None:
    with pytest.raises(ParsingError):
        parse_pdf(b"not really a pdf", "broken.pdf")
