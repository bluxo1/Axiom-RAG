"""Hybrid confidence scoring (Architecture.md §3.5).

```
confidence = w_retrieval    * retrieval_score
           + w_faithfulness * llm_faithfulness
           + w_coverage     * citation_coverage
```

The three components come from two places already computed upstream:

* **retrieval** — mean similarity of the chunks actually cited in the grounded
  answer, normalized from cosine `[-1, 1]` to `[0, 1]`. Using the *used* chunks
  (not the whole top-k) makes the score reflect the evidence the answer really
  stands on: one strong hit buried in weak retrieval still grounds a strong
  answer, and weak-but-cited evidence pulls the score down.
* **faithfulness** and **coverage** — ratios from the grounding pass
  (`GroundingStats`): how many proposed citations survived, and how many claims
  kept a citation.

Weights live in `config.yaml` and sum to 1.0 (asserted at load), so the score
is always in `[0, 1]`.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import ConfidenceWeights
from app.rag.grounding import GroundingStats
from app.rag.types import RetrievedChunk


def normalize_similarity(cosine: float) -> float:
    """Map a cosine similarity in `[-1, 1]` to `[0, 1]`, clamped."""
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def retrieval_score(retrieved: Sequence[RetrievedChunk], used_chunk_ids: Sequence[str]) -> float:
    """Normalized mean similarity of the chunks the grounded answer cites.

    No used chunks (nothing grounded) → 0.0: there is no evidence to be
    confident in.
    """
    used = set(used_chunk_ids)
    scores = [chunk.score for chunk in retrieved if chunk.chunk_id in used]
    if not scores:
        return 0.0
    return normalize_similarity(sum(scores) / len(scores))


def score_confidence(
    *,
    weights: ConfidenceWeights,
    retrieved: Sequence[RetrievedChunk],
    used_chunk_ids: Sequence[str],
    stats: GroundingStats,
) -> tuple[float, float, float, float]:
    """Return `(score, retrieval, faithfulness, coverage)`, each in `[0, 1]`."""
    retrieval = retrieval_score(retrieved, used_chunk_ids)
    faithfulness = stats.faithfulness
    coverage = stats.coverage
    score = (
        weights.retrieval * retrieval
        + weights.faithfulness * faithfulness
        + weights.coverage * coverage
    )
    return score, retrieval, faithfulness, coverage
