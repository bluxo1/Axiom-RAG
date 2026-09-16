# ADR-0001: Embedding model choice

- **Status:** Accepted
- **Date:** 2026-09-16
- **Deciders:** Project author

## Context

Axiom needs embeddings for semantic retrieval. The plan uses OpenAI `text-embedding-3-small` (1536-dim) as the default and `sentence-transformers/all-MiniLM-L6-v2` (384-dim) as a local/dev alternative (Architecture §3.1, §5).

Embedding vectors from different models are **not compatible**: dimensions differ, and even at equal dimension the vector spaces are unrelated. A vector store collection built with one model cannot be searched with another.

## Decision

1. The active embedding model is declared once in `config.yaml` (`embedding.model`), never hardcoded.
2. Each vector-store collection is namespaced by model, e.g. `chunks_<model_shortname>`, so a model switch cannot silently mix incompatible vectors.
3. Switching the embedding model requires **full re-ingestion** of the corpus (re-embed every chunk). Ingestion is idempotent per doc, so this is a re-run of the ingestion job, not a migration of vectors.

## Consequences

- **Positive:** dev can run free/local (MiniLM) while prod uses OpenAI, without cross-contamination; swaps are explicit and reversible by re-ingestion.
- **Negative:** a model switch costs a full re-embed of the corpus (time + API spend on prod-sized corpora).
- **Mitigations:** embeddings cached per doc hash (Architecture §6) so unchanged docs are not re-embedded unnecessarily; eval regression runs (`EVAL.md`) must be repeated after any model switch before trusting old numbers.

## Alternatives considered

- **Single model only** — simpler, but blocks the free local dev path and vendor lock-in escape hatch.
- **Multi-model side-by-side retrieval** — doubles storage and adds rank-fusion complexity; deferred to post-v1 if ever needed.
