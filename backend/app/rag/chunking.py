"""Chunking: parsed text -> stable, embeddable chunks (Architecture.md §3.1).

Splitting is token-based (target 512 tokens, 64 overlap — from `config.yaml`).
The actual splitter is LlamaIndex's `SentenceSplitter` (Architecture.md §5: the
mandated RAG glue), but `chunk_document` takes the splitter as a parameter, so
the id/page logic is a pure function testable with a trivial splitter and no
heavy import.

Pages are split independently so a chunk's page number stays accurate; the chunk
index runs across the whole document, so `chunk_id = sha1(doc_id:index)`
(Design.md §2.1) is stable regardless of how pages divide.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import cast

from app.rag.types import ParsedDocument, TextChunk

Splitter = Callable[[str], list[str]]


def chunk_id_for(doc_id: str, index: int) -> str:
    """`sha1(doc_id:index)` (Design.md §2.1). Stable across re-ingestion."""
    return hashlib.sha1(f"{doc_id}:{index}".encode()).hexdigest()


def chunk_document(
    document: ParsedDocument,
    doc_id: str,
    splitter: Splitter,
) -> list[TextChunk]:
    """Split every page and assign stable ids and page numbers.

    Empty or whitespace-only pieces are dropped: an empty chunk embeds to noise
    and can never support a citation.
    """
    chunks: list[TextChunk] = []
    index = 0
    for page in document.pages:
        for piece in splitter(page.text):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id_for(doc_id, index),
                    doc_id=doc_id,
                    doc_name=document.name,
                    index=index,
                    text=text,
                    page=page.page,
                )
            )
            index += 1
    return chunks


def make_token_splitter(size_tokens: int, overlap_tokens: int) -> Splitter:
    """Build a token-based splitter backed by LlamaIndex `SentenceSplitter`.

    Imported lazily so importing this module (and unit-testing `chunk_document`)
    does not require the LlamaIndex stack.
    """
    from llama_index.core.node_parser import SentenceSplitter

    parser = SentenceSplitter(chunk_size=size_tokens, chunk_overlap=overlap_tokens)

    def split(text: str) -> list[str]:
        return cast("list[str]", parser.split_text(text))

    return split
