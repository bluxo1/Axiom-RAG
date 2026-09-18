"""Application services: the side-effecting orchestration layer (Rules.md §2).

Pure RAG helpers live in `app/rag`; provider interfaces live in `app/llm` and
`app/vector`. This package wires them to the database and to configured
providers — ingestion (parse -> chunk -> embed -> store) and chat (retrieve ->
generate -> persist).
"""
