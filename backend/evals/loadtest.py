"""Latency validation under load (Phases.md Phase 4, EVAL.md §2).

Fires concurrent chat requests at a **running** backend and reports
full-answer latency percentiles against the spec gate:

    P95 full-answer latency < 15s  (Prompt.md §TARGETS, PRD.md §8)

The pipeline measured is the real one — retrieve → generate → verify →
confidence → route — so this needs the stack up and an OPENAI_API_KEY set
(it spends tokens; keep `--requests` modest):

    docker compose up --build --wait
    cd backend
    python -m evals.loadtest --requests 30 --concurrency 5

Each worker sends its own `X-Session-Id` so the per-session rate limiter
does not serialize the run into a rate-limit test. The exit code is the
gate: 0 when every request succeeded and P95 < threshold, 1 otherwise.

Stats are pure functions (`percentile`, `summarize`), unit-tested in
`tests/test_loadtest.py`; only the HTTP loop is live.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
# The gate being validated (EVAL.md: P95 full-answer < 15s).
DEFAULT_P95_THRESHOLD_S = 15.0
CHAT_PATH = "/api/v1/chat"
DOCUMENTS_PATH = "/api/v1/documents"
# Generous: a cold Render instance can take seconds just to wake.
REQUEST_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class Sample:
    """One request's outcome."""

    duration_s: float
    status_code: int | None  # None = transport-level failure (exception)
    error: str | None = None


@dataclass(frozen=True)
class Summary:
    """Aggregate over all samples; `p95_ok` is the Phase 4 gate."""

    count: int
    errors: int
    min_s: float
    median_s: float
    p95_s: float
    max_s: float
    p95_ok: bool


def percentile(samples: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of an ascending-sortable sequence.

    `percentile([x], 95) == x` and `percentile([1, 2, 3, 4], 50) == 2`,
    so a reported P95 really was exceeded by at most 5% of requests.
    """
    if not samples:
        raise ValueError("percentile of an empty sample set")
    ordered = sorted(samples)
    rank = max(1, math.ceil(len(ordered) * pct / 100))
    return ordered[rank - 1]


def summarize(samples: Sequence[Sample], *, p95_threshold_s: float) -> Summary:
    """Aggregate latency samples; only successful (HTTP 200) ones are timed in."""
    durations = [s.duration_s for s in samples if s.status_code == 200]
    errors = sum(1 for s in samples if s.status_code != 200)
    if not durations:
        return Summary(
            count=len(samples),
            errors=errors,
            min_s=0.0,
            median_s=0.0,
            p95_s=0.0,
            max_s=0.0,
            p95_ok=False,
        )
    p95 = percentile(durations, 95)
    return Summary(
        count=len(samples),
        errors=errors,
        min_s=min(durations),
        median_s=statistics.median(durations),
        p95_s=p95,
        max_s=max(durations),
        p95_ok=p95 < p95_threshold_s,
    )


async def _one_chat(client: httpx.AsyncClient, *, session_id: str, question: str) -> Sample:
    """One full-answer chat request, timed end to end."""
    started = time.perf_counter()
    try:
        response = await client.post(
            CHAT_PATH,
            json={"question": question},
            headers={"X-Session-Id": session_id},
        )
        duration = time.perf_counter() - started
        if response.status_code != 200:
            return Sample(
                duration_s=duration,
                status_code=response.status_code,
                error=response.text[:200],
            )
        return Sample(duration_s=duration, status_code=200)
    except httpx.HTTPError as exc:
        return Sample(
            duration_s=time.perf_counter() - started,
            status_code=None,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run(
    client: httpx.AsyncClient,
    *,
    requests: int,
    concurrency: int,
    question: str,
) -> list[Sample]:
    """Fire `requests` chats across `concurrency` workers (one session each)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int) -> Sample:
        async with semaphore:
            return await _one_chat(client, session_id=f"loadtest-{index}", question=question)

    return list(await asyncio.gather(*(worker(i) for i in range(requests))))


async def _ingest_if_asked(client: httpx.AsyncClient, document: Path | None) -> None:
    """Upload the corpus file the questions will be answered from, if given."""
    if document is None:
        return
    with document.open("rb") as handle:
        response = await client.post(DOCUMENTS_PATH, files={"file": (document.name, handle)})
    if response.status_code != 200:
        raise SystemExit(f"ingestion failed ({response.status_code}): {response.text[:300]}")


def print_report(summary: Summary, *, p95_threshold_s: float) -> None:
    """Human-readable report; the last line is the gate verdict."""
    print(f"requests: {summary.count}  errors: {summary.errors}")
    if summary.count == summary.errors:
        print("no successful request to time.")
        return
    print(f"min:    {summary.min_s:6.2f}s")
    print(f"median: {summary.median_s:6.2f}s")
    print(f"p95:    {summary.p95_s:6.2f}s")
    print(f"max:    {summary.max_s:6.2f}s")
    verdict = "PASS" if summary.p95_ok else "FAIL"
    print(f"P95 < {p95_threshold_s:.0f}s: {verdict}")


async def _main(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient(base_url=args.base_url, timeout=REQUEST_TIMEOUT_S) as client:
        await _ingest_if_asked(client, args.document)
        samples = await _run(
            client,
            requests=args.requests,
            concurrency=args.concurrency,
            question=args.question,
        )
    summary = summarize(samples, p95_threshold_s=args.p95_threshold)
    print_report(summary, p95_threshold_s=args.p95_threshold)
    if summary.errors:
        failed = [s for s in samples if s.status_code != 200][:5]
        for sample in failed:
            print(f"  [{sample.status_code}] {sample.error}")
    return 0 if summary.p95_ok and summary.errors == 0 else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--question",
        default="Summarize the key points of the ingested documents.",
    )
    parser.add_argument(
        "--document",
        type=Path,
        default=None,
        help="corpus file to ingest before the run (optional)",
    )
    parser.add_argument(
        "--p95-threshold",
        type=float,
        default=DEFAULT_P95_THRESHOLD_S,
        help="gate for the exit code, seconds (default 15 per EVAL.md)",
    )
    return parser.parse_args(argv)


def main() -> None:
    sys.exit(asyncio.run(_main(parse_args())))


if __name__ == "__main__":
    main()
