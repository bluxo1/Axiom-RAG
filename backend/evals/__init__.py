"""Axiom evaluation suite (Phases.md Phase 4, EVAL.md).

The golden dataset (`golden.jsonl`) and synthetic corpus (`corpus/`) drive a
harness that runs the *real* query pipeline (retrieve → generate → ground →
confidence → route) over deterministic offline providers, so the hallucination
quality gate (EVAL.md §4) runs in CI with no live calls. RAGAS faithfulness /
answer-relevancy need a live judge model and are gated behind `--run-llm`.
"""
