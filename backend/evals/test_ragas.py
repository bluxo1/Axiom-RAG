"""RAGAS faithfulness / answer-relevancy (EVAL.md §1, §3) — live-LLM only.

RAGAS scores need a judge model, so this whole module is skipped unless the run
opts in with `pytest evals/ --run-llm` (which needs `OPENAI_API_KEY` and spends
tokens). CI runs offline and never imports RAGAS — the import is lazy and lives
behind the skip, and `ragas` is an optional `eval` extra, so it is not installed
in the default environment.

Offline (the CI default), the pipeline's grounding guarantees are already
proven by test_hallucination.py and test_golden.py without a judge model; this
module adds the RAGAS numbers quoted in the eval report when run live.
"""

from __future__ import annotations

import pytest

from app.config import AxiomConfig
from evals.dataset import GoldenEntry

# Targets (EVAL.md §1): faithfulness >= 0.85, answer relevancy >= 0.80.
_FAITHFULNESS_TARGET = 0.85
_ANSWER_RELEVANCY_TARGET = 0.80


@pytest.fixture
def _require_llm(run_llm: bool) -> None:
    if not run_llm:
        pytest.skip("RAGAS scoring needs --run-llm (live judge model, spends tokens)")


@pytest.mark.usefixtures("_require_llm")
def test_ragas_faithfulness_and_relevancy_meet_targets(
    config: AxiomConfig, golden: list[GoldenEntry]
) -> None:
    # Lazy import: only reached on a --run-llm run, so CI never needs `ragas`.
    from evals.ragas_runner import score_golden_live

    scores = score_golden_live(config, golden)
    print(
        "\nLive RAGAS results\n"
        f"faithfulness: {scores.faithfulness:.3f} "
        f"(target >= {_FAITHFULNESS_TARGET:.2f})\n"
        f"answer relevancy: {scores.answer_relevancy:.3f} "
        f"(target >= {_ANSWER_RELEVANCY_TARGET:.2f})\n"
        f"scored entries: {scores.scored}\n"
        f"report: {scores.report_path}"
    )
    assert scores.faithfulness >= _FAITHFULNESS_TARGET, (
        f"faithfulness {scores.faithfulness:.3f} < {_FAITHFULNESS_TARGET}"
    )
    assert scores.answer_relevancy >= _ANSWER_RELEVANCY_TARGET, (
        f"answer relevancy {scores.answer_relevancy:.3f} < {_ANSWER_RELEVANCY_TARGET}"
    )
