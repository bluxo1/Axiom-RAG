"""Retrieval-augmented generation: parsing, chunking, prompting, retrieval.

Rules.md §2: this package is pure and side-effect-free. Parsing, chunking, and
prompt construction are deterministic functions of their inputs; I/O (embedding
calls, the vector store, the database) lives in `app/llm`, `app/vector`, and
`app/services`.
"""
