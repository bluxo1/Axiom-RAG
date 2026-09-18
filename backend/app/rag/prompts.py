"""Grounded-generation prompt construction (Design.md §3.1, §3.3).

Pure string building. The system prompt pins the grounding rules and the
structured-output contract; the context block lists each retrieved chunk behind
its raw `[chunk_id]` marker, which is what the model is told to cite. The
verifier (`rag/grounding.py`) rewrites verified markers to display ids
`[c1], [c2], ...` after generation, as the hard gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.types import RetrievedChunk

INSUFFICIENT_EVIDENCE = "insufficient_evidence"

# Design.md §3.1 + §3.3: the grounded rules, expressed against the JSON schema
# the model must return (structured-output mode). Rule 3 becomes a `status`
# value rather than a bare string. Built by concatenation, not str.format, so
# the literal JSON braces in the schema description need no escaping.
STRUCTURED_INSTRUCTIONS = (
    "You are Axiom, an assistant that answers ONLY from the provided context.\n\n"
    "Return a single JSON object with exactly these keys:\n"
    '  "status": "answered" or "insufficient_evidence"\n'
    '  "answer": a short answer string (empty when status is insufficient_evidence)\n'
    '  "claims": a list of {"text": <one factual sentence>, '
    '"citation_ids": [<chunk_id>, ...]}\n'
    '  "citations": a list of {"citation_id": <chunk_id>, "chunk_id": <chunk_id>}\n\n'
    "Rules:\n"
    "1. Every claim's citation_ids MUST be chunk_ids taken from the context below.\n"
    "2. NEVER cite a chunk_id that is not present in the context.\n"
    '3. If the context does not contain the answer, return {"status": '
    '"insufficient_evidence", "answer": "", "claims": [], "citations": []}.\n'
    "4. Do not use outside knowledge. Answer only from the context.\n\n"
    "Context:\n"
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


def build_structured_system_prompt(chunks: Sequence[RetrievedChunk]) -> str:
    """The full structured-output system prompt with the context block appended."""
    return STRUCTURED_INSTRUCTIONS + build_context(chunks)
