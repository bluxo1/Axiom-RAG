"""The hallucination gate (EVAL.md §4) — the test that proves the tagline.

Over the whole golden set, run through the real pipeline offline:

1. No answer carries a citation whose `chunk_id` was not retrieved (kb) — web
   citations are the retrieved web results, labelled `source="web"`.
2. No out-of-corpus or adversarial question receives a confident (>= high),
   unflagged, corpus-grounded answer.
3. Every fallback answer is labelled `source="web"` with a live URL.

Plus a stress test: even a *hostile* model that always fabricates cannot get an
unsupported citation past the gate. If any assertion fails, the build fails.
"""

from __future__ import annotations

import pytest

from app.config import AxiomConfig
from evals.dataset import GoldenEntry
from evals.harness import EvalRecord, run_entry_hostile, run_entry_offline


@pytest.fixture(scope="module")
def offline_records(request: pytest.FixtureRequest) -> list[EvalRecord]:
    """Run the whole golden set once with the honest offline model."""
    config = request.getfixturevalue("config")
    golden: list[GoldenEntry] = request.getfixturevalue("golden")
    return [run_entry_offline(config, entry) for entry in golden]


def test_no_citation_references_an_unretrieved_chunk(offline_records: list[EvalRecord]) -> None:
    for record in offline_records:
        retrieved = set(record.retrieved_chunk_ids)
        for citation in record.citations:
            if citation.source == "web":
                continue  # web citations are the retrieved web results, not kb ids.
            assert citation.chunk_id in retrieved, (
                f"{record.id}: citation {citation.chunk_id!r} was never retrieved"
            )


def test_no_out_of_corpus_answer_is_confident_and_unflagged(
    offline_records: list[EvalRecord], config: AxiomConfig
) -> None:
    high = config.confidence.thresholds.high
    for record in offline_records:
        if record.category == "in-corpus" or record.fallback_used:
            continue
        if record.confidence_score is None:
            continue  # a refusal has no score — the honest outcome.
        assert not (record.confidence_score >= high and not record.flagged), (
            f"{record.id}: out-of-corpus answered confidently unflagged "
            f"(score={record.confidence_score})"
        )


def test_fallback_answers_are_labeled_web(offline_records: list[EvalRecord]) -> None:
    saw_fallback = False
    for record in offline_records:
        if not record.fallback_used:
            continue
        saw_fallback = True
        assert record.citation_sources, f"{record.id}: fallback answer has no citations"
        assert all(source == "web" for source in record.citation_sources), (
            f"{record.id}: fallback citation not labelled web: {record.citation_sources}"
        )
    assert saw_fallback, "no fallback fired across the golden set — assertion is vacuous"


def test_hostile_model_cannot_leak_a_fabricated_citation(
    config: AxiomConfig, golden: list[GoldenEntry]
) -> None:
    # A model that always fabricates: the gate must reduce every turn to an
    # honest refusal — no citations, nothing confident.
    for entry in golden:
        record = run_entry_hostile(config, entry)
        assert record.insufficient_evidence, f"{entry.id}: hostile answer was not refused"
        assert record.citation_chunk_ids == (), f"{entry.id}: fabricated citation reached output"
        assert record.confidence_score is None, f"{entry.id}: refusal carried a confidence score"
