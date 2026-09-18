"""Marker rewriting: raw `[chunk_id]` -> display `[c1], [c2], ...` (Design.md §3.3).

Pure and deterministic. Phase 1 has no verifier, so this rewrites the markers
the model emitted for chunks that were actually retrieved, in first-seen order,
and returns the matching citation cards. Markers that name a chunk not in the
retrieved set are left untouched — Phase 2's grounding gate is what strips or
rejects those (Architecture.md §3.4); Phase 1 does not silently drop them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.rag.types import Citation, RetrievedChunk

_MARKER = re.compile(r"\[([^\[\]]+)\]")


def assign_display_citations(
    answer: str,
    retrieved: Sequence[RetrievedChunk],
) -> tuple[str, tuple[Citation, ...], tuple[str, ...]]:
    """Rewrite known markers and build the citation list.

    Returns the rewritten answer, the citations (one per cited chunk, in display
    order), and the cited chunk_ids.
    """
    by_id = {chunk.chunk_id: chunk for chunk in retrieved}
    display_of: dict[str, str] = {}
    order: list[str] = []

    def rewrite(match: re.Match[str]) -> str:
        chunk_id = match.group(1).strip()
        if chunk_id not in by_id:
            return match.group(0)
        if chunk_id not in display_of:
            order.append(chunk_id)
            display_of[chunk_id] = f"c{len(order)}"
        return f"[{display_of[chunk_id]}]"

    rewritten = _MARKER.sub(rewrite, answer)
    citations = tuple(
        Citation(
            id=display_of[chunk_id],
            chunk_id=chunk_id,
            doc=by_id[chunk_id].doc_name,
            text=by_id[chunk_id].text,
            score=by_id[chunk_id].score,
            page=by_id[chunk_id].page,
            source="kb",
        )
        for chunk_id in order
    )
    return rewritten, citations, tuple(order)
