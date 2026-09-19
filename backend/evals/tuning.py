"""Threshold tuning on the golden set (Phases.md Phase 4, Architecture.md §3.5).

`config.yaml` ships the router thresholds (`high`/`low`) as *a-priori* values;
this module tunes them against the golden set and records before/after numbers
for the README (EVAL.md §5).

**Why a single run is enough.** A confidence *score* is threshold-independent —
`score_confidence` never reads a threshold, and the router only *buckets* an
already-computed score (`confidence/router.py`). Offline, the category mix makes
the buckets stable too: an in-corpus question grounds on the corpus and is
scored before routing; an out-of-corpus question refuses on the corpus and is
answered from the web fixture (fallback, regardless of threshold); an
adversarial question refuses (no score). So we run the golden set through the
real pipeline once, capture each outcome, and re-bucket it under every candidate
`(high, low)` pair analytically — exact, and fast enough to sweep densely.

**Objective** (Architecture.md §3.5, EVAL.md §4):

* hard constraint — zero out-of-corpus/adversarial answers routed *confident*
  (`>= high`, unflagged) off the corpus: the tagline. Offline this population is
  empty by construction, so the sweep also proves no threshold breaks it.
* maximize in-corpus recall — grounded in-corpus answers routed ANSWER (green),
  not needlessly flagged.
* keep a **noise margin** below the in-corpus score floor: production answers
  from a real model score lower than the noise-free offline double, so `high`
  must sit under the observed floor with headroom, not hug it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import AxiomConfig
from evals.dataset import GoldenEntry
from evals.harness import EvalRecord, run_entry_offline

# Headroom between the weakest genuine in-corpus score and `high`: a real model
# scores below the noise-free offline double, so green must not hug the floor.
NOISE_MARGIN = 0.15
# Round the recommendation to a tidy, reviewable step (config values are human-
# edited); never round *up* past the recall-preserving ceiling.
ROUND_STEP = 0.05


@dataclass(frozen=True)
class SweepRow:
    """Routing metrics for one candidate `(high, low)` pair over the golden set."""

    high: float
    low: float
    in_corpus_answer: int  # grounded in-corpus routed ANSWER (green)
    in_corpus_flagged: int  # grounded in-corpus routed FLAG (yellow) instead
    out_of_corpus_fallback: int  # out-of-corpus honestly web-answered
    escapes: int  # out-of-corpus/adversarial confident+unflagged off the corpus

    @property
    def in_corpus_total(self) -> int:
        return self.in_corpus_answer + self.in_corpus_flagged

    @property
    def in_corpus_recall(self) -> float:
        """Fraction of grounded in-corpus answers trusted (green). 1.0 if none."""
        return self.in_corpus_answer / self.in_corpus_total if self.in_corpus_total else 1.0


def collect_records(config: AxiomConfig, golden: Sequence[GoldenEntry]) -> list[EvalRecord]:
    """Run the whole golden set once through the offline pipeline."""
    return [run_entry_offline(config, entry) for entry in golden]


def _is_corpus_grounded(record: EvalRecord) -> bool:
    """True when the answer stands on corpus chunks (a real confidence score)."""
    return (
        record.confidence_score is not None
        and not record.fallback_used
        and not record.insufficient_evidence
    )


def evaluate_thresholds(records: Sequence[EvalRecord], *, high: float, low: float) -> SweepRow:
    """Re-bucket a single offline run under a candidate `(high, low)` pair."""
    in_answer = in_flag = oc_fallback = escapes = 0
    for record in records:
        if record.category == "in-corpus" and _is_corpus_grounded(record):
            assert record.confidence_score is not None  # narrowed by _is_corpus_grounded
            if record.confidence_score >= high:
                in_answer += 1
            else:
                in_flag += 1
            continue
        if record.fallback_used:
            oc_fallback += 1
            continue
        # Out-of-corpus/adversarial that somehow grounded on the corpus and would
        # be routed confident+unflagged: the escape the tagline forbids.
        if _is_corpus_grounded(record):
            assert record.confidence_score is not None
            if record.confidence_score >= high:
                escapes += 1
    return SweepRow(
        high=high,
        low=low,
        in_corpus_answer=in_answer,
        in_corpus_flagged=in_flag,
        out_of_corpus_fallback=oc_fallback,
        escapes=escapes,
    )


def sweep(
    records: Sequence[EvalRecord],
    *,
    highs: Sequence[float],
    low: float,
) -> list[SweepRow]:
    """Evaluate a grid of `high` candidates at a fixed `low`."""
    return [evaluate_thresholds(records, high=high, low=low) for high in highs]


@dataclass(frozen=True)
class Recommendation:
    """The tuned thresholds plus the numbers that justify them."""

    high: float
    low: float
    in_corpus_floor: float
    baseline: SweepRow
    tuned: SweepRow


def _in_corpus_scores(records: Sequence[EvalRecord]) -> list[float]:
    return [
        record.confidence_score
        for record in records
        if record.category == "in-corpus"
        and _is_corpus_grounded(record)
        and record.confidence_score is not None
    ]


def recommend(config: AxiomConfig, records: Sequence[EvalRecord]) -> Recommendation:
    """Derive `high` from the in-corpus score floor, keeping `low` as configured.

    `high = floor(in_corpus_min - NOISE_MARGIN)` down to `ROUND_STEP`: high enough
    that "green" means evidence at least as strong as our weakest golden answer
    minus a noise buffer, low enough to keep 100% in-corpus recall. `low` has no
    empirical pressure offline (no genuine answer scores near it), so it is left
    at its configured value.
    """
    scores = _in_corpus_scores(records)
    floor = min(scores) if scores else config.confidence.thresholds.high
    raw = floor - NOISE_MARGIN
    high = max(0.0, (raw // ROUND_STEP) * ROUND_STEP)
    high = round(high, 2)
    low = config.confidence.thresholds.low
    baseline = evaluate_thresholds(records, high=config.confidence.thresholds.high, low=low)
    tuned = evaluate_thresholds(records, high=high, low=low)
    return Recommendation(high=high, low=low, in_corpus_floor=floor, baseline=baseline, tuned=tuned)


# ─── Reporting (EVAL.md §5) ───────────────────────────────────────────────────

# The `high` grid the report sweeps, spanning either side of the in-corpus floor
# so the recall cliff is visible in the table.
SWEEP_HIGHS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def render_tuning_report(
    rec: Recommendation, rows: Sequence[SweepRow], *, when: datetime | None = None
) -> str:
    """Render the Markdown threshold-tuning report body."""
    stamp = (when or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")
    changed = (rec.high, rec.low) != (rec.baseline.high, rec.baseline.low)
    verdict = (
        f"tuned to **high={rec.high}, low={rec.low}**"
        if changed
        else f"**confirmed** the a-priori defaults (high={rec.high}, low={rec.low})"
    )
    lines = [
        f"# Axiom threshold tuning — {stamp}",
        "",
        f"Run over the golden set offline; the sweep {verdict}.",
        "",
        f"- In-corpus score floor: **{rec.in_corpus_floor:.4f}**",
        f"- Noise margin below the floor: **{NOISE_MARGIN}** "
        f"→ high = floor - margin, rounded down to {ROUND_STEP} = **{rec.high}**",
        f"- `low` has no empirical pressure offline (no genuine answer scores near "
        f"it) and is left at **{rec.low}**",
        "",
        "## Before / after",
        "",
        "| Thresholds | In-corpus green | In-corpus flagged | Web fallbacks | Escapes |",
        "|------------|-----------------|-------------------|---------------|---------|",
        _report_row("before (default)", rec.baseline),
        _report_row("after (tuned)", rec.tuned),
        "",
        f"## Sweep (fixed low={rec.low:.2f})",
        "",
        "| high | In-corpus green recall | Flagged | Escapes |",
        "|------|------------------------|---------|---------|",
    ]
    lines += [_sweep_row(row) for row in rows]
    lines += [
        "",
        "Escapes stay **0** across the whole grid: out-of-corpus questions refuse "
        "on the corpus and are web-answered, adversarial questions refuse — neither "
        "produces a corpus-grounded confident answer at any threshold. In-corpus "
        "recall holds at 100% until `high` crosses the score floor, then collapses.",
        "",
    ]
    return "\n".join(lines)


def _sweep_row(row: SweepRow) -> str:
    recall = f"{row.in_corpus_recall:.0%} ({row.in_corpus_answer}/{row.in_corpus_total})"
    return f"| {row.high:.2f} | {recall} | {row.in_corpus_flagged} | {row.escapes} |"


def _report_row(label: str, row: SweepRow) -> str:
    return (
        f"| {label} (high={row.high}, low={row.low}) | "
        f"{row.in_corpus_answer}/{row.in_corpus_total} | {row.in_corpus_flagged} | "
        f"{row.out_of_corpus_fallback} | {row.escapes} |"
    )


def write_tuning_report(body: str) -> Path:
    """Write the tuning report to `evals/reports/threshold-tuning.md`."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "threshold-tuning.md"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    """Run the sweep and write the report (`python -m evals.tuning`)."""
    import sys

    from app.config import Settings, load_knobs
    from evals.dataset import load_golden
    from tests.helpers import CONFIG_PATH

    config = AxiomConfig(Settings(), load_knobs(CONFIG_PATH))
    golden = load_golden()
    records = collect_records(config, golden)
    rec = recommend(config, records)
    rows = sweep(records, highs=SWEEP_HIGHS, low=rec.low)
    body = render_tuning_report(rec, rows)
    path = write_tuning_report(body)
    # The report body has non-ASCII (→, -); a Windows cp1252 console can't encode
    # it. Write bytes straight to the buffer so the CLI never dies on encoding.
    sys.stdout.buffer.write(body.encode("utf-8"))
    sys.stdout.buffer.write(f"\nWrote {path}\n".encode())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
