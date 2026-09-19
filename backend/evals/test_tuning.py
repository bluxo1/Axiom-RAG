"""Threshold tuning is reproducible and never trades away the tagline (Phase 4).

These lock in the tuning conclusion the README quotes: the sweep keeps a zero
escape rate at every threshold, in-corpus recall holds until `high` crosses the
score floor, and the data-driven recommendation stays recall-preserving.
"""

from __future__ import annotations

import pytest

from app.config import AxiomConfig
from evals.dataset import GoldenEntry
from evals.harness import EvalRecord
from evals.tuning import (
    SWEEP_HIGHS,
    collect_records,
    evaluate_thresholds,
    recommend,
    sweep,
)


@pytest.fixture(scope="module")
def records(request: pytest.FixtureRequest) -> list[EvalRecord]:
    config = request.getfixturevalue("config")
    golden: list[GoldenEntry] = request.getfixturevalue("golden")
    return collect_records(config, golden)


def test_escape_rate_is_zero_across_the_whole_grid(
    records: list[EvalRecord], config: AxiomConfig
) -> None:
    low = config.confidence.thresholds.low
    for row in sweep(records, highs=SWEEP_HIGHS, low=low):
        assert row.escapes == 0, f"high={row.high} let {row.escapes} escape the corpus"


def test_recall_holds_below_the_floor_and_collapses_above_it(
    records: list[EvalRecord], config: AxiomConfig
) -> None:
    low = config.confidence.thresholds.low
    below = evaluate_thresholds(records, high=0.75, low=low)
    above = evaluate_thresholds(records, high=0.95, low=low)
    assert below.in_corpus_recall == 1.0, "in-corpus answers flagged below the score floor"
    assert above.in_corpus_recall < below.in_corpus_recall, "recall did not fall past the floor"


def test_recommendation_is_recall_preserving(
    records: list[EvalRecord], config: AxiomConfig
) -> None:
    rec = recommend(config, records)
    assert rec.low == config.confidence.thresholds.low
    assert rec.high <= rec.in_corpus_floor, "recommended high above the in-corpus floor"
    assert rec.tuned.in_corpus_recall == 1.0
    assert rec.tuned.escapes == 0
