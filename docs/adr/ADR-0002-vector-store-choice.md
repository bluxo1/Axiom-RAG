# ADR-0002: Vector store choice (ChromaDB dev / pgvector prod)

- **Status:** Accepted
- **Date:** 2026-09-17
- **Deciders:** Project author

## Context

Axiom needs a vector store for chunk embeddings (Architecture §3.1, §5). The plan uses ChromaDB in dev and pgvector in prod. Unlike the embedding model (ADR-0001), the vector store is not forced by incompatible-vector concerns, but the dev/prod split still has failure modes: different deployment stories, different query semantics, and a data migration if dev corpus state is ever promoted.

## Decision

1. All vector-store access goes behind a single retrieval interface (`rag/` layer), never direct client calls in routes. Swapping the backend is a config change (`config.yaml: vector_store.backend`), not a code change.
2. Dev runs ChromaDB (embedded, zero-ops, ships in Docker Compose); prod runs pgvector on the same Postgres instance that already holds documents, sessions, and spend logs — one managed DB to operate (Neon).
3. The store contract is limited to what the retriever needs: upsert(chunk), query_topk(vector, k), delete_by_doc(doc_id). Anything richer (hybrid search, metadata filters) is a deliberate contract change, not a hidden dependency.
4. Dev corpora are disposable: there is no dev→prod data promotion path. Migration to prod = re-ingestion, mirroring the embedding-model re-ingestion rule (ADR-0001).

## Consequences

- **Positive:** dev has zero extra infrastructure; prod keeps a single relational store; the interface keeps both swappable (a third backend, e.g. Qdrant, is config-only).
- **Negative:** pgvector's index options (IVFFlat/HNSW) and distance functions must be kept consistent with the similarity metric used in dev or scores shift subtly across environments.
- **Mitigations:** the similarity metric (cosine) is pinned in `config.yaml` and asserted by an eval smoke test that compares top-k results across backends on a fixed corpus.
