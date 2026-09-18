"""Vector store behind a retrieval interface (ADR-0002).

The backend (ChromaDB in dev, pgvector in prod) is chosen by `config.yaml` and
never leaks into the pipeline: everything depends on the `VectorStore` Protocol.
Cosine similarity is pinned in config so dev and prod scores are comparable.
"""
