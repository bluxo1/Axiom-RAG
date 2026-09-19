"""Eval report writer (EVAL.md §5).

Summarises a harness run into a dated Markdown table under `evals/reports/`,
the reproducible source for the numbers quoted in the README. Offline metrics
(citation precision, escape rate, category routing) are always available; the
RAGAS faithfulness / answer-relevancy columns are filled only on a `--run-llm`
run and left as `n/a (offline)` otherwise, so a report never implies a live
number was measured when it was not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evals.dataset import EVALS_DIR
from evals.harness import EvalRecord

REPORTS_DIR = EVALS_DIR / "reports"


@dataclass(frozen=True)
class ReportMetrics:
    """The numbers a report row carries."""

    total: int
    citation_precision: float
    escaped_unsupported: int
    confident_out_of_corpus: int
    fallback_rate: float
    faithfulness: float | None = None
    answer_relevancy: float | None = None


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a (offline)"


def render_report(metrics: ReportMetrics, *, mode: str, when: datetime | None = None) -> str:
    """Render the Markdown report body for one run."""
    stamp = (when or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            f"# Axiom eval report — {stamp}",
            "",
            f"- Mode: **{mode}**",
            f"- Golden entries: **{metrics.total}**",
            "",
            "| Metric | Value | Target |",
            "|--------|-------|--------|",
            f"| Citation precision | {metrics.citation_precision:.3f} | >= 0.95 |",
            f"| Unsupported-claim escape rate | {metrics.escaped_unsupported} | 0 |",
            f"| Confident out-of-corpus answers | {metrics.confident_out_of_corpus} | 0 |",
            f"| Fallback rate | {metrics.fallback_rate:.3f} | — |",
            f"| Faithfulness (RAGAS) | {_fmt(metrics.faithfulness)} | >= 0.85 |",
            f"| Answer relevancy (RAGAS) | {_fmt(metrics.answer_relevancy)} | >= 0.80 |",
            "",
        ]
    )


def write_report(body: str, *, when: datetime | None = None) -> Path:
    """Write a report to `evals/reports/<date>.md` and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = (when or datetime.now(UTC)).strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"{stamp}.md"
    path.write_text(body, encoding="utf-8")
    return path


def confident_out_of_corpus(records: Sequence[EvalRecord]) -> int:
    """Count out-of-corpus/adversarial turns answered confidently and unflagged.

    "Confident" = a corpus-grounded answer at or above the high threshold that is
    not flagged and not a web fallback. This is the number EVAL.md §4 asserts is
    zero.
    """
    count = 0
    for record in records:
        if record.category == "in-corpus":
            continue
        if record.fallback_used:
            continue  # an honest web answer is allowed, not a hallucination.
        if (
            record.confidence_score is not None
            and record.confidence_score >= 0.75
            and not record.flagged
        ):
            count += 1
    return count
