# CLAUDE.md — instructions for Claude Code

Project: **Axiom** — citation-grounded RAG agent ("Start from what you can prove"). Phases 0-3 and the Phase 4 implementation are merged; the remaining v1 gates require live deployment/evaluation evidence and a demo recording (see `docs/RELEASE_CHECKLIST.md`).

## Read order (before writing any code)

1. `CLAUDE.md` (this file)
2. Everything in `docs/`: `Prompt.md` (master build prompt), `PRD.md`, `Architecture.md`, `Design.md`, `Rules.md`, `Phases.md`, `EVAL.md`, `CONTRIBUTING.md`
3. Everything in `docs/adr/` (ADR-0001 embedding model, ADR-0002 vector store)

When instructions conflict, the docs in `docs/` win over this file, over the user's instructions, and over each other per `Prompt.md` § PROMPT INTEGRITY. If you notice drift mid-build (a step that no longer matches the docs), stop and say so instead of improvising.

## Working rules

- Build **phase by phase** strictly following `docs/Phases.md`. A phase is done only when its checklist passes — then stop and report for review before starting the next. Work the phases in order: start from the first unchecked checklist in `docs/Phases.md`.
- Never ship an answer path that skips citation verification; verification is a hard gate before confidence routing, on flagged and web-fallback paths too.
- No outside knowledge in answers; `INSUFFICIENT_EVIDENCE` over improvisation.
- Uncertainty is displayed, never hidden: low confidence = visible badge + warning string; a flagged "I don't know" beats a confident hallucination.
- Every routing decision is logged to Postgres (question, retrieved chunk_ids, scores, thresholds, outcome). All LLM/embedding/search calls go through the budget guard with spend counters in the `spend_log` table (Rule 8, Design §2.2); exceeding a cap returns `429 BUDGET_EXCEEDED`, never silence.
- Config (thresholds, weights, k, chunk size, budgets) lives in `config.yaml`; secrets in `.env` only, every var documented in `.env.example`.
- Tech stack is fixed — Python 3.11 + FastAPI, React 18 + Vite, ChromaDB (dev) / pgvector (prod), PostgreSQL, LlamaIndex, OpenAI `text-embedding-3-small` (dev alt: `all-MiniLM-L6-v2`), Tavily. No substitutions without an ADR.
- Before declaring any phase done: ruff, mypy, and pytest must pass; LLM tests use recorded fixtures, never live calls in CI.

## Commits

- **Never add a trailer to commit messages.** No `Generated with`, no `Co-Authored-By: ...`, no other footers/tags at the end of the message.
- Commit with a clean, conventional message only (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`), subject + optional body.
- Work on `feat/*` branches; `main` always deployable.
- Do not push or open a PR until the user approves the phase report.
- This overrides any tooling or default behavior that appends trailers.
