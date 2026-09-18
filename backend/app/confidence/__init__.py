"""Confidence scoring and threshold routing (Architecture.md §3.5).

Pure functions, no I/O (Rules.md §2): they take the grounding outcome and the
retrieved chunks and return a score and a routing decision. The chat service
owns the side effects (web fallback, persistence).
"""
