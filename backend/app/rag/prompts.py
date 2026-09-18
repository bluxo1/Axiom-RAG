"""Grounded-generation prompt construction (Design.md §3.1).

Pure string building. The system prompt pins the grounding rules; the context
block lists each retrieved chunk behind its raw `[chunk_id]` marker, which is
what the model is told to cite. Rewriting verified markers to display ids
`[c1], [c2], ...` happens after generation (Phase 1: a light pass in the chat
service; Phase 2: the verifier does it as the hard gate).
"""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.types import RetrievedChunk

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = (
    "You are Axiom, an assistant that answers ONLY from the provided context.\n\n"
    "Rules:\n"
    "1. Every factual claim MUST end with a citation marker [chunk_id] from the context.\n"
    "2. NEVER cite a chunk_id that is not present in the context.\n"
    f"3. If the context does not contain the answer, reply exactly: {INSUFFICIENT_EVIDENCE}\n"
    "4. Do not use outside knowledge.\n\n"
    "Context:\n"
    "{context}"
)


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a citable context block.

    Each entry leads with its raw `[chunk_id]` marker so the model cites exactly
    that token, followed by a source hint (doc name and page) and the text.
    """
    blocks: list[str] = []
    for chunk in chunks:
        where = chunk.doc_name if chunk.page is None else f"{chunk.doc_name}, p.{chunk.page}"
        blocks.append(f"[{chunk.chunk_id}] ({where})\n{chunk.text}")
    return "\n\n".join(blocks)


def build_system_prompt(chunks: Sequence[RetrievedChunk]) -> str:
    """The full system prompt with the context block filled in."""
    return SYSTEM_PROMPT.format(context=build_context(chunks))
