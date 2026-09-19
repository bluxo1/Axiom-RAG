"""Custom eval scorers (EVAL.md §1).

RAGAS ships faithfulness and answer-relevancy but **not** citation precision, so
it is implemented here on top of the grounding verifier's support check — the
same `_supports` predicate the hard gate uses, so the eval measures exactly what
production enforces.

* **citation precision** — of every (claim, cited-chunk) pair that reached the
  displayed answer, the fraction whose chunk text actually supports the claim.
  Because the hard gate strips unsupported citations before display, a correct
  pipeline scores 1.0; a value below 1.0 means an unsupported citation escaped,
  which is the failure this metric exists to catch.
* **unsupported-claim escape rate** — count of displayed citations that do *not*
  support their claim. Target: 0 (EVAL.md §1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.grounding import _supports
from app.rag.types import Citation


@dataclass(frozen=True)
class CitationPrecision:
    """Aggregate citation-precision over a set of answered turns."""

    supported: int
    total: int

    @property
    def precision(self) -> float:
        """Fraction of displayed citations that support their claim (1.0 if none)."""
        return self.supported / self.total if self.total else 1.0

    @property
    def escaped(self) -> int:
        """Displayed citations that do not support their claim (target: 0)."""
        return self.total - self.supported


def citation_precision(
    answer: str,
    citations: Sequence[Citation],
    *,
    support_keyword_overlap_min: float,
) -> CitationPrecision:
    """Score one answer's citations against the claim text they annotate.

    Each citation's display marker `[cN]` appears in `answer` next to the
    sentence it supports; we check the cited chunk text against that sentence
    with the same support predicate the grounding gate uses.
    """
    supported = 0
    total = 0
    for citation in citations:
        sentence = _sentence_for_marker(answer, citation.id)
        total += 1
        if _supports(sentence, citation.text, support_keyword_overlap_min):
            supported += 1
    return CitationPrecision(supported=supported, total=total)


def _sentence_for_marker(answer: str, display_id: str) -> str:
    """The answer text of the claim carrying `[display_id]`.

    The grounded answer is `claim1 [c1] claim2 [c2] ...`; return the claim whose
    marker matches, falling back to the whole answer if the marker isn't found
    (so the support check still has the full context to work with).
    """
    marker = f"[{display_id}]"
    index = answer.find(marker)
    if index == -1:
        return answer
    # Walk back to the start of this claim: just after the previous "] ".
    start = answer.rfind("] ", 0, index)
    begin = start + 2 if start != -1 else 0
    return answer[begin:index]
