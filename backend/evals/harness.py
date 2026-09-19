"""Eval harness: run the golden set through the real query pipeline.

The pipeline under test is the production one — `answer_question` (retrieve →
generate → ground → confidence → route → persist). Only the *providers* are
swapped for deterministic offline doubles (CLAUDE.md: no live calls in CI):

* a `HashingEmbedder` + `InMemoryVectorStore` with the real synthetic corpus
  ingested, so retrieval is genuine keyword-overlap retrieval;
* a **category-aware** LLM double that behaves like an honest model — it cites
  the best-matching retrieved chunk for in-corpus questions, answers
  out-of-corpus questions only from injected web results, and refuses the false
  premises of adversarial questions. The grounding hard gate, confidence scorer,
  and router are the *real* ones, so the harness measures pipeline behaviour,
  not the double.

`hostile_llm` is the opposite double — it always tries to fabricate — used by
the hallucination gate to prove the pipeline strips fabrications regardless of
what the model emits.

Live mode (`--run-llm`) swaps in the real OpenAI provider for RAGAS scoring; it
needs `OPENAI_API_KEY` and costs tokens.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.config import AxiomConfig
from app.db.session import Database
from app.llm.embeddings import HashingEmbedder
from app.llm.provider import INSUFFICIENT_EVIDENCE_JSON, LLMProvider
from app.rag.types import ChatResult, Citation
from app.search.provider import ScriptedWebSearch, WebResult
from app.services.chat import answer_question
from app.services.ingestion import ingest_upload
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore
from evals.dataset import CORPUS_DIR, GoldenEntry
from evals.fixtures import WEB_FIXTURES, structured

TEST_DATABASE_URL = "sqlite+pysqlite:///:memory:"

# A context entry: "[<id>] (<where>)\n<text>", id either a 40-hex kb chunk_id or
# "web#N". Matches the block up to the next entry or the end (prompts.build_context).
_CONTEXT_ENTRY = re.compile(r"\[([0-9a-f]{40}|web#\d+)\] \([^)]*\)\n(.+?)(?=\n\n\[|\Z)", re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")
_FABRICATED_CHUNK_ID = "f" * 40  # well-formed but never retrieved.


@dataclass(frozen=True)
class EvalRecord:
    """One golden entry's outcome after running the pipeline."""

    id: str
    category: str
    question: str
    answer: str
    insufficient_evidence: bool
    flagged: bool
    fallback_used: bool
    confidence_score: float | None
    confidence_level: str | None
    retrieved_chunk_ids: tuple[str, ...]
    citations: tuple[Citation, ...]
    citation_chunk_ids: tuple[str, ...]
    citation_sources: tuple[str, ...]
    cited_docs: tuple[str, ...]


def _context_entries(system_prompt: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in _CONTEXT_ENTRY.finditer(system_prompt)]


def _overlap(question: str, text: str) -> int:
    q = {w for w in _WORD.findall(question.lower()) if len(w) >= 3}
    t = {w for w in _WORD.findall(text.lower()) if len(w) >= 3}
    return len(q & t)


def _best_entry(question: str, entries: Sequence[tuple[str, str]]) -> tuple[str, str] | None:
    """The context entry whose text shares the most keywords with the question."""
    best: tuple[str, str] | None = None
    best_score = 0
    for chunk_id, text in entries:
        score = _overlap(question, text)
        if score > best_score:
            best_score, best = score, (chunk_id, text)
    return best


class HonestGoldenLLM:
    """A well-behaved model, parameterised by the golden entry's category.

    in-corpus: cite the best-matching retrieved chunk (grounded, confident).
    out-of-corpus: refuse on corpus context; answer from web context if present.
    adversarial: always refuse — never confirm a false premise.
    """

    def __init__(self, category: str) -> None:
        self._category = category

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        entries = _context_entries(system_prompt)
        is_web = any(chunk_id.startswith("web#") for chunk_id, _ in entries)

        if self._category == "adversarial":
            return INSUFFICIENT_EVIDENCE_JSON  # decline the false premise honestly.
        if self._category == "out-of-corpus" and not is_web:
            return INSUFFICIENT_EVIDENCE_JSON  # not in the corpus -> force fallback.

        best = _best_entry(user_prompt, entries)
        if best is None:
            return INSUFFICIENT_EVIDENCE_JSON
        chunk_id, text = best
        return structured([(text, [chunk_id])])


class HostileLLM:
    """Always tries to fabricate: an unsupported claim citing a fabricated id.

    Used to prove the grounding gate strips fabrications on every path — the
    output must never reach the user regardless of what the model emits.
    """

    def complete_structured(self, *, system_prompt: str, user_prompt: str) -> str:
        claim = "Fabricated claim with no support in any source."
        return structured([(claim, [_FABRICATED_CHUNK_ID])])


def build_runtime(config: AxiomConfig, llm: LLMProvider, *, web: list[WebResult]) -> Runtime:
    """An offline runtime with the synthetic corpus ingested and `llm` wired in."""
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    runtime = Runtime(
        config,
        database,
        embedder=HashingEmbedder(),
        llm=llm,
        vector_store=InMemoryVectorStore(),
        web_search=ScriptedWebSearch(web),
    )
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        ingest_upload(runtime, name=path.name, data=path.read_bytes())
    return runtime


def _record(entry: GoldenEntry, result: ChatResult) -> EvalRecord:
    confidence = result.confidence
    return EvalRecord(
        id=entry.id,
        category=entry.category,
        question=entry.question,
        answer=result.answer,
        insufficient_evidence=result.insufficient_evidence,
        flagged=result.flagged,
        fallback_used=result.fallback_used,
        confidence_score=confidence.score if confidence is not None else None,
        confidence_level=confidence.level if confidence is not None else None,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        citations=result.citations,
        citation_chunk_ids=tuple(c.chunk_id for c in result.citations),
        citation_sources=tuple(c.source for c in result.citations),
        cited_docs=tuple(c.doc for c in result.citations),
    )


def run_entry_offline(config: AxiomConfig, entry: GoldenEntry) -> EvalRecord:
    """Run one golden entry with the honest, category-aware offline model."""
    web = WEB_FIXTURES.get(entry.id, [])
    runtime = build_runtime(config, HonestGoldenLLM(entry.category), web=web)
    try:
        result = answer_question(runtime, question=entry.question)
    finally:
        runtime.db.dispose()
    return _record(entry, result)


def run_entry_hostile(config: AxiomConfig, entry: GoldenEntry) -> EvalRecord:
    """Run one golden entry with the fabricating model (gate stress test)."""
    runtime = build_runtime(config, HostileLLM(), web=[])
    try:
        result = answer_question(runtime, question=entry.question)
    finally:
        runtime.db.dispose()
    return _record(entry, result)
