"""Recorded fixtures for the offline eval run (CLAUDE.md: no live calls in CI).

* `structured` builds a structured-answer JSON payload (Design.md §3.3) the way
  a model would emit it, for the offline LLM doubles.
* `WEB_FIXTURES` maps an out-of-corpus golden id to the web results a live
  search *would* have returned, so the fallback path produces a real
  web-sourced answer deterministically. The content is written to share
  keywords with the question, so the grounding support check accepts it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from app.search.provider import WebResult


def structured(claims: Sequence[tuple[str, Sequence[str]]], *, answer: str = "ok") -> str:
    """A structured-answer JSON string: each claim cites raw chunk-id markers."""
    cited: list[str] = []
    for _text, chunk_ids in claims:
        for chunk_id in chunk_ids:
            if chunk_id not in cited:
                cited.append(chunk_id)
    payload = {
        "status": "answered",
        "answer": answer,
        "claims": [{"text": text, "citation_ids": list(ids)} for text, ids in claims],
        "citations": [{"citation_id": c, "chunk_id": c} for c in cited],
    }
    return json.dumps(payload)


def _web(title: str, url: str, content: str) -> list[WebResult]:
    return [WebResult(title=title, url=url, content=content, relevance=0.9)]


# One plausible live result per out-of-corpus question. Content deliberately
# echoes the question's keywords so the keyword support check grounds it.
WEB_FIXTURES: dict[str, list[WebResult]] = {
    "oc-01": _web(
        "Canberra - Wikipedia",
        "https://en.wikipedia.org/wiki/Canberra",
        "Canberra is the capital city of Australia, located in the Australian Capital Territory.",
    ),
    "oc-02": _web(
        "FIFA World Cup final result",
        "https://www.fifa.com/worldcup",
        "Argentina won the most recent FIFA World Cup final, defeating France on penalties.",
    ),
    "oc-03": _web(
        "Boiling point of water",
        "https://en.wikipedia.org/wiki/Boiling_point",
        "The boiling point of water at sea level is one hundred degrees Celsius.",
    ),
    "oc-04": _web(
        "Eiffel Tower height",
        "https://en.wikipedia.org/wiki/Eiffel_Tower",
        "The Eiffel Tower stands about three hundred thirty metres tall including its antennas.",
    ),
    "oc-05": _web(
        "History of Python",
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "Python is a programming language released by Guido van Rossum in 1991.",
    ),
    "oc-06": _web(
        "Chemical symbol for gold",
        "https://en.wikipedia.org/wiki/Gold",
        "The chemical symbol for gold on the periodic table is Au.",
    ),
}
