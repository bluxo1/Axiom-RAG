"""Live RAGAS scoring (EVAL.md §1, §3) — imported only on a `--run-llm` run.

This module is never imported in CI: `test_ragas.py` imports it lazily inside a
test that is skipped unless `--run-llm` is passed, and `ragas` is an optional
`eval` extra. It builds a runtime with the *real* OpenAI providers, runs the
in-corpus golden questions through the production pipeline, and scores the
(question, answer, contexts, ground-truth) tuples with RAGAS faithfulness and
answer-relevancy.

Only in-corpus entries are scored: RAGAS faithfulness measures whether the
answer is grounded in the retrieved contexts, which is meaningful only when a
grounded answer is expected. Out-of-corpus/adversarial behaviour is asserted
offline by test_hallucination.py and test_golden.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import AxiomConfig
from app.db.session import Database
from app.services.chat import answer_question
from app.services.ingestion import ingest_upload
from app.services.runtime import Runtime
from evals.dataset import CORPUS_DIR, GoldenEntry
from evals.harness import TEST_DATABASE_URL, _context_entries


@dataclass(frozen=True)
class RagasScores:
    """Aggregate RAGAS metrics over the scored golden entries."""

    faithfulness: float
    answer_relevancy: float
    scored: int


def _build_live_runtime(config: AxiomConfig) -> Runtime:
    """A runtime with the real OpenAI + vector providers and the corpus ingested."""
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    runtime = Runtime(config, database)  # providers built lazily from config/.env.
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        ingest_upload(runtime, name=path.name, data=path.read_bytes())
    return runtime


def score_golden_live(
    config: AxiomConfig, golden: list[GoldenEntry], *, judge_model: str | None = None
) -> RagasScores:
    """Run in-corpus golden entries live and score them with RAGAS.

    The judge follows the app's own provider (EVAL.md §3): same key, same
    endpoint, same model family — so an OpenAI-compatible override
    (`OPENAI_API_BASE`, ADR-0003) applies to scoring too and the judge can
    never call a different vendor than the one being evaluated.
    """
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pydantic import SecretStr
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    settings = config.settings
    base_url = settings.openai_api_base
    api_key = settings.openai_api_key
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not configured; cannot score live.")
    secret_key = SecretStr(api_key.get_secret_value())
    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model_name=judge_model or config.generation.model,
            openai_api_key=secret_key,
            openai_api_base=base_url,
            request_timeout=60,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.embedding.model,
            openai_api_key=secret_key,
            openai_api_base=base_url,
        )
    )

    rows: list[dict[str, object]] = []
    runtime = _build_live_runtime(config)
    try:
        for entry in golden:
            if entry.category != "in-corpus":
                continue
            result = answer_question(runtime, question=entry.question)
            if result.insufficient_evidence:
                continue  # nothing to score: RAGAS needs an answer + contexts.
            contexts = [text for _chunk_id, text in _context_entries_for(runtime, entry)]
            rows.append(
                {
                    "question": entry.question,
                    "answer": result.answer,
                    "contexts": contexts,
                    "ground_truth": entry.expected_answer,
                }
            )
    finally:
        runtime.db.dispose()

    dataset = Dataset.from_list(rows)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=judge_embeddings,
    )
    return RagasScores(
        faithfulness=float(scores["faithfulness"]),
        answer_relevancy=float(scores["answer_relevancy"]),
        scored=len(rows),
    )


def _context_entries_for(runtime: Runtime, entry: GoldenEntry) -> list[tuple[str, str]]:
    """Retrieve and render the contexts the pipeline would ground this question on."""
    from app.rag.prompts import build_context
    from app.services.retrieval import retrieve

    retrieved = retrieve(runtime, query=entry.question)
    return _context_entries(build_context(retrieved))
