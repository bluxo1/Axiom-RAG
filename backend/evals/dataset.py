"""Golden dataset loader (EVAL.md §2).

`golden.jsonl` holds 30-50 entries across three categories:

* **in-corpus** (60%) — the answer exists in the synthetic corpus; expect a
  grounded answer citing the `must_cite` document(s).
* **out-of-corpus** (20%) — the answer is not in the corpus; expect a web
  fallback or a flagged refusal, never a corpus-grounded fabrication.
* **adversarial** (20%) — a false-premise / trick question; expect a refusal or
  flag, never a confident confirmation of the false premise.

`must_cite` is expressed at **document-name** granularity, not raw `chunk_id`:
chunk ids are `sha1(doc_id:index)` over a per-ingest random `doc_id`
(Design.md §2.1), so they are not stable across runs. Document names are, which
keeps the golden set reproducible.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden.jsonl"
CORPUS_DIR = EVALS_DIR / "corpus"

Category = Literal["in-corpus", "out-of-corpus", "adversarial"]
_CATEGORIES: frozenset[str] = frozenset({"in-corpus", "out-of-corpus", "adversarial"})


@dataclass(frozen=True)
class GoldenEntry:
    """One golden Q&A row (EVAL.md §2)."""

    id: str
    question: str
    expected_answer: str
    must_cite: tuple[str, ...]
    category: Category


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenEntry]:
    """Parse `golden.jsonl` into typed entries, validating each row."""
    entries: list[GoldenEntry] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed fixture
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        entry = _entry_from(record, where=f"{path}:{line_number}")
        if entry.id in seen_ids:
            raise ValueError(f"{path}:{line_number} duplicate id {entry.id!r}")
        seen_ids.add(entry.id)
        entries.append(entry)
    if not entries:  # pragma: no cover - empty fixture
        raise ValueError(f"{path} contains no entries")
    return entries


def _entry_from(record: object, *, where: str) -> GoldenEntry:
    if not isinstance(record, dict):  # pragma: no cover - malformed fixture
        raise TypeError(f"{where} must be a JSON object")
    category = record.get("category")
    if category not in _CATEGORIES:
        raise ValueError(f"{where} has invalid category {category!r}")
    must_cite = record.get("must_cite", [])
    if not isinstance(must_cite, list) or not all(isinstance(name, str) for name in must_cite):
        raise ValueError(f"{where} must_cite must be a list of document names")
    return GoldenEntry(
        id=str(record["id"]),
        question=str(record["question"]),
        expected_answer=str(record.get("expected_answer", "")),
        must_cite=tuple(must_cite),
        category=category,
    )


def category_counts(entries: Sequence[GoldenEntry]) -> dict[str, int]:
    """Count entries per category (for the report header and mix assertions)."""
    counts = dict.fromkeys(sorted(_CATEGORIES), 0)
    for entry in entries:
        counts[entry.category] += 1
    return counts
