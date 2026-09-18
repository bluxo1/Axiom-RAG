"""Citation grounding — the hard gate (Architecture.md §3.4, Prompt.md §PIPELINE 4).

Runs on the model's structured answer *before* it can reach the user. For every
claim:

1. **Existence check** — a citation whose `chunk_id` is not in the retrieved set
   is dropped (Rule 2: a chunk_id not retrieved is a bug, whatever the model
   emitted).
2. **Support check** — the cited chunk's text must share enough keywords with
   the claim (fraction >= `grounding.support_keyword_overlap_min`). This is the
   cheap entailment proxy; NLI on all claims is post-v1.

A claim with no surviving citation is **stripped entirely** — never silently
kept (Rule 1). The displayed answer is then rebuilt *from the surviving claims
only*, so a fabricated or unsupported claim is structurally unable to appear in
the output: the model's free-form `answer` field is never shown. Surviving
citations are renumbered to display ids `c1, c2, ...` in first-cited order, and
the claim text carries the matching `[cN]` markers.

If nothing survives, grounding fails (`grounded=False`) and the chat service
returns the honest refusal card (Rule 6: "I don't know" beats a hallucination).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.rag.generation import RawStructuredAnswer
from app.rag.types import Citation, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")
_MIN_TOKEN_LEN = 3

# Common words carry no grounding signal; excluding them stops a claim from
# being "supported" by an incidental "the"/"and" overlap. This is algorithmic
# structure, not a tunable knob — the *threshold* is the config value.
_STOPWORD_TEXT = (
    "the and for are but not you all any can her was one our out his has had how its "
    "who did yes she him this that with from have will your they them then than when "
    "what which were been does into some such only also there these those their would "
    "could should about"
)
_STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())


def _keywords(text: str) -> frozenset[str]:
    """Content tokens of `text`: lowercased, length-filtered, stopwords removed."""
    return frozenset(
        token
        for token in _TOKEN.findall(text.lower())
        if len(token) >= _MIN_TOKEN_LEN and token not in _STOPWORDS
    )


def _supports(claim_text: str, chunk_text: str, min_overlap: float) -> bool:
    """True when enough of the claim's keywords appear in the cited chunk.

    Overlap is the fraction of *claim* keywords found in the chunk, so a short,
    specific claim must be well covered by its source. A claim with no content
    keywords cannot be supported (there is nothing to ground).
    """
    claim_keywords = _keywords(claim_text)
    if not claim_keywords:
        return False
    overlap = len(claim_keywords & _keywords(chunk_text)) / len(claim_keywords)
    return overlap >= min_overlap


@dataclass(frozen=True)
class GroundingStats:
    """Counts the confidence scorer needs (Architecture.md §3.5).

    * `faithfulness = verified_citations / proposed_citations` — how many of the
      citations the model emitted actually held up.
    * `coverage = grounded_claims / total_claims` — how much of the answer is
      backed by at least one verified citation.

    A ratio over an empty denominator is 0.0 (nothing proposed cannot be
    faithful; no claims cannot be covered).
    """

    total_claims: int = 0
    grounded_claims: int = 0
    proposed_citations: int = 0
    verified_citations: int = 0

    @property
    def faithfulness(self) -> float:
        if self.proposed_citations == 0:
            return 0.0
        return self.verified_citations / self.proposed_citations

    @property
    def coverage(self) -> float:
        if self.total_claims == 0:
            return 0.0
        return self.grounded_claims / self.total_claims


@dataclass(frozen=True)
class GroundingOutcome:
    """Result of the hard gate.

    `grounded=False` means nothing survived verification: the caller returns the
    insufficient-evidence refusal. When `grounded=True`, `answer` is rebuilt from
    surviving claims and every `[cN]` marker in it has a matching citation.
    `stats` feeds the confidence scorer either way (Architecture.md §3.5).
    """

    grounded: bool
    answer: str = ""
    citations: tuple[Citation, ...] = ()
    used_chunk_ids: tuple[str, ...] = ()
    stats: GroundingStats = field(default_factory=GroundingStats)


def verify_and_ground(
    parsed: RawStructuredAnswer,
    retrieved: Sequence[RetrievedChunk],
    *,
    support_keyword_overlap_min: float,
) -> GroundingOutcome:
    """Verify citations and rebuild a grounded answer, or fail to a refusal."""
    if parsed.status == "insufficient_evidence":
        return GroundingOutcome(grounded=False)

    by_id = {chunk.chunk_id: chunk for chunk in retrieved}
    marker_to_chunk = {citation.citation_id: citation.chunk_id for citation in parsed.citations}

    # (claim_text, [verified chunk_id, ...]) for claims that keep >=1 citation.
    surviving: list[tuple[str, list[str]]] = []
    proposed_citations = 0
    verified_citations = 0
    for claim in parsed.claims:
        valid: list[str] = []
        for marker in claim.citation_ids:
            proposed_citations += 1
            chunk_id = marker_to_chunk.get(marker, marker)  # tolerate direct chunk_id markers
            if chunk_id not in by_id:  # existence check
                continue
            if not _supports(claim.text, by_id[chunk_id].text, support_keyword_overlap_min):
                continue
            verified_citations += 1
            if chunk_id not in valid:
                valid.append(chunk_id)
        if valid:
            surviving.append((claim.text, valid))

    stats = GroundingStats(
        total_claims=len(parsed.claims),
        grounded_claims=len(surviving),
        proposed_citations=proposed_citations,
        verified_citations=verified_citations,
    )

    if not surviving:
        return GroundingOutcome(grounded=False, stats=stats)

    display_of: dict[str, str] = {}
    order: list[str] = []
    rebuilt: list[str] = []
    for text, chunk_ids in surviving:
        markers = []
        for chunk_id in chunk_ids:
            if chunk_id not in display_of:
                order.append(chunk_id)
                display_of[chunk_id] = f"c{len(order)}"
            markers.append(f"[{display_of[chunk_id]}]")
        rebuilt.append(f"{text.rstrip()} {''.join(markers)}")

    citations = tuple(
        Citation(
            id=display_of[chunk_id],
            chunk_id=chunk_id,
            doc=by_id[chunk_id].doc_name,
            text=by_id[chunk_id].text,
            score=by_id[chunk_id].score,
            page=by_id[chunk_id].page,
            source=by_id[chunk_id].source,
            url=by_id[chunk_id].url,
        )
        for chunk_id in order
    )
    return GroundingOutcome(
        grounded=True,
        answer=" ".join(rebuilt),
        citations=citations,
        used_chunk_ids=tuple(order),
        stats=stats,
    )
