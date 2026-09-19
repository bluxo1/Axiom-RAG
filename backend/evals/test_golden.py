"""Golden-set behaviour and citation precision (EVAL.md §1, §2).

Runs the whole golden set through the real pipeline offline and asserts:

* the category mix is 60% in-corpus / 20% out-of-corpus / 20% adversarial;
* in-corpus questions produce a grounded answer citing the expected document(s);
* out-of-corpus questions either fall back to the web or refuse — never a
  corpus-grounded answer;
* adversarial questions are always refused or flagged — never a confident
  confirmation of a false premise;
* citation precision across the set is >= 0.95 with a zero unsupported-claim
  escape rate (the two custom scorers, EVAL.md §1).
"""

from __future__ import annotations

import pytest

from app.config import AxiomConfig
from evals.dataset import GoldenEntry, category_counts
from evals.harness import EvalRecord, run_entry_offline
from evals.scoring import CitationPrecision, citation_precision


@pytest.fixture(scope="module")
def offline_records(request: pytest.FixtureRequest) -> list[EvalRecord]:
    config = request.getfixturevalue("config")
    golden: list[GoldenEntry] = request.getfixturevalue("golden")
    return [run_entry_offline(config, entry) for entry in golden]


def test_category_mix_is_60_20_20(golden: list[GoldenEntry]) -> None:
    counts = category_counts(golden)
    total = len(golden)
    assert total >= 30, f"golden set too small: {total} entries (EVAL.md wants 30-50)"
    # Allow a small rounding slack around the target proportions.
    assert abs(counts["in-corpus"] / total - 0.60) <= 0.05, counts
    assert abs(counts["out-of-corpus"] / total - 0.20) <= 0.05, counts
    assert abs(counts["adversarial"] / total - 0.20) <= 0.05, counts


def test_in_corpus_questions_cite_the_expected_document(
    offline_records: list[EvalRecord], golden: list[GoldenEntry]
) -> None:
    must_cite = {entry.id: entry.must_cite for entry in golden}
    for record in offline_records:
        if record.category != "in-corpus":
            continue
        assert not record.insufficient_evidence, f"{record.id}: in-corpus question refused"
        assert record.citations, f"{record.id}: in-corpus answer carried no citation"
        expected = set(must_cite[record.id])
        if not expected:
            continue
        cited_docs = set(record.cited_docs)
        assert expected & cited_docs, (
            f"{record.id}: expected a citation from {expected}, got {cited_docs}"
        )


def test_out_of_corpus_questions_never_fabricate_from_corpus(
    offline_records: list[EvalRecord],
) -> None:
    for record in offline_records:
        if record.category != "out-of-corpus":
            continue
        # Either an honest web fallback, or a refusal — never a kb-grounded answer.
        if record.insufficient_evidence:
            continue
        assert record.fallback_used, f"{record.id}: out-of-corpus answered from the corpus"
        assert all(source == "web" for source in record.citation_sources), (
            f"{record.id}: out-of-corpus citation not web-sourced: {record.citation_sources}"
        )


def test_adversarial_questions_are_never_confidently_confirmed(
    offline_records: list[EvalRecord], config: AxiomConfig
) -> None:
    high = config.confidence.thresholds.high
    for record in offline_records:
        if record.category != "adversarial":
            continue
        if record.confidence_score is None:
            continue  # refused — the honest outcome.
        assert not (record.confidence_score >= high and not record.flagged), (
            f"{record.id}: adversarial premise confirmed confidently "
            f"(score={record.confidence_score})"
        )


def test_citation_precision_meets_target(
    offline_records: list[EvalRecord], config: AxiomConfig
) -> None:
    overlap_min = config.grounding.support_keyword_overlap_min
    supported = 0
    total = 0
    for record in offline_records:
        result = citation_precision(
            record.answer, record.citations, support_keyword_overlap_min=overlap_min
        )
        supported += result.supported
        total += result.total
    aggregate = CitationPrecision(supported=supported, total=total)
    assert aggregate.escaped == 0, f"{aggregate.escaped} unsupported citations escaped the gate"
    assert aggregate.precision >= 0.95, f"citation precision {aggregate.precision:.3f} < 0.95"
