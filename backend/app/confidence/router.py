"""Threshold router (Architecture.md §3.5).

Maps a confidence score to one of three decisions against the config thresholds:

* `score >= high`        → **answer** normally
* `low <= score < high`  → **flag** (answer + visible low-confidence badge)
* `score <  low`         → **fallback** (try live web search)

`REFUSAL` is *not* returned here — it is decided by the chat service when there
is nothing to score (empty retrieval, grounding failed) or when a fallback
attempt also fails to ground. Keeping the router pure and total over a real
score keeps the routing rule itself trivial to test.
"""

from __future__ import annotations

from enum import StrEnum

from app.config import ConfidenceThresholds


class RouterDecision(StrEnum):
    """Persisted to `messages.router_decision` (Design.md §2.2)."""

    ANSWER = "answer"
    FLAG = "flag"
    FALLBACK = "fallback"
    REFUSAL = "refusal"


def route(score: float, thresholds: ConfidenceThresholds) -> RouterDecision:
    """Route a confidence score to answer / flag / fallback."""
    if score >= thresholds.high:
        return RouterDecision.ANSWER
    if score >= thresholds.low:
        return RouterDecision.FLAG
    return RouterDecision.FALLBACK
