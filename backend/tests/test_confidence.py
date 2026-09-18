"""Hybrid confidence scoring and threshold routing (Architecture.md §3.5).

Pure-function unit tests: no DB, no providers. The scorer combines a retrieval
term (mean normalized similarity of the *cited* chunks) with the grounding
pass's faithfulness and coverage ratios; the router maps the result to
answer / flag / fallback against the configured thresholds.
"""

from __future__ import annotations

import pytest

from app.confidence.router import RouterDecision, route
from app.confidence.scoring import normalize_similarity, retrieval_score, score_confidence
from app.config import AxiomConfig, ConfidenceThresholds, ConfidenceWeights
from app.rag.grounding import GroundingStats
from app.rag.types import RetrievedChunk

WEIGHTS = ConfidenceWeights(retrieval=0.30, faithfulness=0.40, coverage=0.30)
THRESHOLDS = ConfidenceThresholds(high=0.75, low=0.45)


def _chunk(chunk_id: str, score: float, *, source: str = "kb") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d",
        doc_name="doc",
        text="text",
        score=score,
        page=None,
        source=source,
    )


@pytest.mark.parametrize(
    ("cosine", "expected"),
    [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0), (-2.0, 0.0), (2.0, 1.0)],
)
def test_normalize_similarity_maps_cosine_to_unit_interval(cosine: float, expected: float) -> None:
    assert normalize_similarity(cosine) == expected


def test_retrieval_score_uses_only_cited_chunks() -> None:
    retrieved = [_chunk("a", 1.0), _chunk("b", -1.0), _chunk("c", 0.0)]
    # Cites only the strong hit: score reflects that, not the weak neighbours.
    assert retrieval_score(retrieved, ["a"]) == 1.0
    # Cites the strong and the weak: mean cosine 0.0 -> normalized 0.5.
    assert retrieval_score(retrieved, ["a", "b"]) == 0.5


def test_retrieval_score_is_zero_when_nothing_is_cited() -> None:
    assert retrieval_score([_chunk("a", 1.0)], []) == 0.0


def test_score_confidence_is_the_weighted_sum() -> None:
    retrieved = [_chunk("a", 1.0)]  # normalized retrieval == 1.0
    stats = GroundingStats(
        total_claims=2, grounded_claims=1, proposed_citations=2, verified_citations=1
    )  # faithfulness 0.5, coverage 0.5
    score, retrieval, faithfulness, coverage = score_confidence(
        weights=WEIGHTS, retrieved=retrieved, used_chunk_ids=["a"], stats=stats
    )
    assert (retrieval, faithfulness, coverage) == (1.0, 0.5, 0.5)
    # 0.30*1.0 + 0.40*0.5 + 0.30*0.5 = 0.65
    assert score == pytest.approx(0.65)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.75, RouterDecision.ANSWER),  # boundary is inclusive
        (0.90, RouterDecision.ANSWER),
        (0.60, RouterDecision.FLAG),
        (0.45, RouterDecision.FLAG),  # boundary is inclusive
        (0.44, RouterDecision.FALLBACK),
        (0.0, RouterDecision.FALLBACK),
    ],
)
def test_route_maps_score_to_decision(score: float, expected: RouterDecision) -> None:
    assert route(score, THRESHOLDS) is expected


def test_route_never_returns_refusal() -> None:
    # REFUSAL is the orchestrator's call, never the router's (router.py docstring).
    decisions = {route(s / 100, THRESHOLDS) for s in range(0, 101)}
    assert RouterDecision.REFUSAL not in decisions


def test_weak_retrieval_can_never_be_confident(config: AxiomConfig) -> None:
    """Phases.md Phase 3 / Rule 3: weak retrieval is flagged or falls back — never
    routed to a confident ANSWER, even with a perfect grounding pass.

    With retrieval == 0 (every cited chunk at the cosine floor) the score is
    capped at `w_faithfulness + w_coverage` == 0.70, below the 0.75 high
    threshold, so the router can only FLAG or FALLBACK. Uses the committed
    config's real weights/thresholds so the property holds for production tuning.
    """
    weights = config.confidence.weights
    thresholds = config.confidence.thresholds

    # Weakest retrieval, strongest possible grounding pass.
    weak = [_chunk("a", -1.0)]  # normalized retrieval == 0.0
    perfect = GroundingStats(
        total_claims=1, grounded_claims=1, proposed_citations=1, verified_citations=1
    )
    score, retrieval, _, _ = score_confidence(
        weights=weights, retrieved=weak, used_chunk_ids=["a"], stats=perfect
    )
    assert retrieval == 0.0
    assert score < thresholds.high  # cannot be confident on weak retrieval alone.
    assert route(score, thresholds) is not RouterDecision.ANSWER
