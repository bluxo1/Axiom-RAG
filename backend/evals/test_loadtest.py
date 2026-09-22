"""Load-test stats: pure functions, no HTTP (CLAUDE.md: offline tests only)."""

from __future__ import annotations

import pytest

from evals.loadtest import Sample, percentile, print_report, summarize

# ─── percentile ───────────────────────────────────────────────────────────────


def test_percentile_of_a_single_sample_is_itself() -> None:
    assert percentile([7.5], 95) == 7.5


def test_percentile_is_the_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0


def test_percentile_p95_of_one_hundred_samples() -> None:
    samples = [float(i) for i in range(1, 101)]
    assert percentile(samples, 95) == 95.0


def test_percentile_of_empty_set_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 95)


# ─── summarize ────────────────────────────────────────────────────────────────


def test_summarize_times_only_successful_requests() -> None:
    summary = summarize(
        [
            Sample(duration_s=1.0, status_code=200),
            Sample(duration_s=99.0, status_code=429, error="rate limited"),
            Sample(duration_s=3.0, status_code=200),
        ],
        p95_threshold_s=15.0,
    )

    assert summary.count == 3
    assert summary.errors == 1
    assert summary.min_s == 1.0
    assert summary.median_s == 2.0
    assert summary.p95_s == 3.0
    assert summary.max_s == 3.0


def test_all_errors_yields_a_failing_summary() -> None:
    summary = summarize(
        [Sample(duration_s=1.0, status_code=None, error="boom")], p95_threshold_s=15.0
    )

    assert summary.errors == 1
    assert summary.p95_ok is False


def test_p95_gate_passes_only_below_threshold() -> None:
    ok = summarize([Sample(duration_s=14.9, status_code=200)], p95_threshold_s=15.0)
    fail = summarize([Sample(duration_s=15.1, status_code=200)], p95_threshold_s=15.0)

    assert ok.p95_ok is True
    assert fail.p95_ok is False


def test_report_prints_the_verdict(capsys: pytest.CaptureFixture[str]) -> None:
    summary = summarize([Sample(duration_s=2.0, status_code=200)], p95_threshold_s=15.0)

    print_report(summary, p95_threshold_s=15.0)

    out = capsys.readouterr().out
    assert "PASS" in out
    assert "p95:" in out
