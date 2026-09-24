"""Resumable live RAGAS scoring over the trusted synthetic golden corpus.

Answers and judge metrics are checkpointed separately so a provider quota error
does not discard completed work. The checkpoint contains no provider secrets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.config import AxiomConfig, resolve_config_path
from app.db.session import Database
from app.services.chat import answer_question
from app.services.ingestion import ingest_upload
from app.services.runtime import Runtime
from app.vector.store import ChromaVectorStore
from evals.dataset import CORPUS_DIR, EVALS_DIR, GOLDEN_PATH, GoldenEntry
from evals.harness import TEST_DATABASE_URL, _context_entries

if TYPE_CHECKING:
    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM

_LOG = logging.getLogger(__name__)
_REPO_ROOT = EVALS_DIR.parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "ragas"
_REPORTS_DIR = EVALS_DIR / "reports"
_CACHE_VERSION = "1"
_METRICS = ("faithfulness", "answer_relevancy")


@dataclass(frozen=True)
class RagasScores:
    """Aggregate scores over every in-corpus entry and the evidence file."""

    faithfulness: float
    answer_relevancy: float
    scored: int
    report_path: Path


def _fingerprint(config: AxiomConfig, judge_model: str) -> str:
    """Only resume when the evaluation inputs and scoring code are unchanged."""
    digest = hashlib.sha256()
    paths = [
        GOLDEN_PATH,
        resolve_config_path(config.settings.config_path),
        Path(__file__),
        *sorted(CORPUS_DIR.glob("*.txt")),
        *sorted((_REPO_ROOT / "backend" / "app").rglob("*.py")),
    ]
    for path in paths:
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    options = {
        "cache_version": _CACHE_VERSION,
        "ragas_version": version("ragas"),
        "generation_model": config.generation.model,
        "embedding_model": config.embedding.model,
        "judge_model": judge_model,
        "provider_base": config.settings.openai_api_base,
    }
    digest.update(json.dumps(options, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _load_checkpoint(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "fingerprint": fingerprint,
            "started_at": datetime.now(UTC).isoformat(),
            "entries": {},
        }
    saved: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or saved.get("fingerprint") != fingerprint:
        raise ValueError(f"Invalid RAGAS checkpoint: {path}")
    if not isinstance(saved.get("entries"), dict):
        raise ValueError(f"Invalid RAGAS checkpoint entries: {path}")
    return saved


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _delete_collection(config: AxiomConfig, collection_name: str) -> None:
    try:
        import chromadb

        chromadb.HttpClient(
            host=config.settings.chroma_host, port=config.settings.chroma_port
        ).delete_collection(name=collection_name)
    except Exception:
        _LOG.warning("Could not remove temporary RAGAS collection %s", collection_name)


def _build_live_runtime(config: AxiomConfig, collection_name: str) -> Runtime:
    """Ingest into a fresh Chroma collection, separate from local app data."""
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    try:
        vector_store = ChromaVectorStore(
            host=config.settings.chroma_host,
            port=config.settings.chroma_port,
            collection_name=collection_name,
        )
        runtime = Runtime(config, database, vector_store=vector_store)
        for path in sorted(CORPUS_DIR.glob("*.txt")):
            ingest_upload(runtime, name=path.name, data=path.read_bytes())
    except BaseException:
        database.dispose()
        _delete_collection(config, collection_name)
        raise
    return runtime


def _score_metric(
    row: dict[str, Any],
    metric_name: str,
    judge: BaseRagasLLM,
    judge_embeddings: BaseRagasEmbeddings,
) -> float:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness
    from ragas.run_config import RunConfig

    metric = faithfulness if metric_name == "faithfulness" else answer_relevancy
    result = evaluate(
        Dataset.from_list([row]),
        metrics=[metric],
        llm=judge,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=1, max_retries=1),
        raise_exceptions=True,
        show_progress=False,
    )
    score = float(result[metric_name])
    if not math.isfinite(score):
        raise ValueError(f"RAGAS returned a non-finite {metric_name} score")
    return score


def _write_live_report(
    config: AxiomConfig,
    judge_model: str,
    golden: list[GoldenEntry],
    checkpoint: dict[str, Any],
    faithfulness: float,
    answer_relevancy: float,
) -> Path:
    now = datetime.now(UTC)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORTS_DIR / f"live-ragas-{now:%Y-%m-%d}.md"
    entries = checkpoint["entries"]
    lines = [
        f"# Live RAGAS report — {now:%Y-%m-%d %H:%M UTC}",
        "",
        f"- Started: {checkpoint['started_at']}",
        f"- Completed: {now.isoformat()}",
        f"- Generation model: `{config.generation.model}`",
        f"- Judge model: `{judge_model}`",
        f"- Embedding model: `{config.embedding.model}`",
        f"- In-corpus entries: {len(golden)}",
        "",
        "| Metric | Score | Target |",
        "| --- | ---: | ---: |",
        f"| Faithfulness | {faithfulness:.3f} | >= 0.85 |",
        f"| Answer relevancy | {answer_relevancy:.3f} | >= 0.80 |",
        "",
        "| Entry | Faithfulness | Answer relevancy |",
        "| --- | ---: | ---: |",
    ]
    for entry in golden:
        progress = entries[entry.id]
        lines.append(
            f"| {entry.id} | {progress['faithfulness']:.3f} | {progress['answer_relevancy']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def score_golden_live(
    config: AxiomConfig, golden: list[GoldenEntry], *, judge_model: str | None = None
) -> RagasScores:
    """Generate and judge each in-corpus answer, resuming saved progress."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pydantic import SecretStr
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    settings = config.settings
    api_key = settings.openai_api_key
    if api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not configured; cannot score live.")
    target_model = judge_model or config.generation.model
    in_corpus = [entry for entry in golden if entry.category == "in-corpus"]
    fingerprint = _fingerprint(config, target_model)
    cache_path = _CACHE_DIR / f"{fingerprint}.json"
    checkpoint = _load_checkpoint(cache_path, fingerprint)
    entries: dict[str, dict[str, Any]] = checkpoint["entries"]

    pending = [entry for entry in in_corpus if "row" not in entries.get(entry.id, {})]
    if pending:
        collection_name = f"ragas_{fingerprint[:12]}_{uuid4().hex[:8]}"
        runtime = _build_live_runtime(config, collection_name)
        try:
            for entry in pending:
                result = answer_question(runtime, question=entry.question)
                if result.insufficient_evidence:
                    raise RuntimeError(f"In-corpus entry {entry.id} had no answer to score")
                contexts = [text for _chunk_id, text in _context_entries_for(runtime, entry)]
                if not contexts:
                    raise RuntimeError(f"In-corpus entry {entry.id} had no contexts to score")
                entries.setdefault(entry.id, {})["row"] = {
                    "question": entry.question,
                    "answer": result.answer,
                    "contexts": contexts,
                    "ground_truth": entry.expected_answer,
                }
                _save_checkpoint(cache_path, checkpoint)
                print(f"RAGAS saved answer {entry.id}", flush=True)
        finally:
            runtime.db.dispose()
            _delete_collection(config, collection_name)

    secret_key = SecretStr(api_key.get_secret_value())
    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model_name=target_model,
            openai_api_key=secret_key,
            openai_api_base=settings.openai_api_base,
            request_timeout=60,
            max_retries=0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=config.embedding.model,
            openai_api_key=secret_key,
            openai_api_base=settings.openai_api_base,
            max_retries=0,
        )
    )
    for entry in in_corpus:
        progress = entries[entry.id]
        for metric_name in _METRICS:
            if metric_name in progress:
                continue
            progress[metric_name] = _score_metric(
                progress["row"], metric_name, judge, judge_embeddings
            )
            _save_checkpoint(cache_path, checkpoint)
            print(f"RAGAS saved {entry.id} {metric_name}", flush=True)

    faithfulness_score = fmean(entries[entry.id]["faithfulness"] for entry in in_corpus)
    relevancy_score = fmean(entries[entry.id]["answer_relevancy"] for entry in in_corpus)
    report_path = _write_live_report(
        config, target_model, in_corpus, checkpoint, faithfulness_score, relevancy_score
    )
    return RagasScores(
        faithfulness=faithfulness_score,
        answer_relevancy=relevancy_score,
        scored=len(in_corpus),
        report_path=report_path,
    )


def _context_entries_for(runtime: Runtime, entry: GoldenEntry) -> list[tuple[str, str]]:
    """Retrieve and render the contexts the pipeline would ground this question on."""
    from app.rag.prompts import build_context
    from app.services.retrieval import retrieve

    retrieved = retrieve(runtime, query=entry.question)
    return _context_entries(build_context(retrieved))
