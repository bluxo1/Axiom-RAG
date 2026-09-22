# Axiom — Implementation Phases

> Build order with exit criteria. A phase is done when its checklist passes - no scope creep.

**Total target:** ~4 weeks (part-time internship pace)

---

## Phase 0 - Setup & Skeleton (2-3 days)

- [x] Repo structure (`backend/`, `frontend/`, `data/`, `evals/`)
- [x] Docker Compose: FastAPI + Postgres + Chroma
- [x] CI: lint (ruff), typecheck (mypy), tests on PR
- [x] Env config loader + `.env.example`
- [x] `config.yaml` scaffold with every documented knob (thresholds 0.75/0.45, confidence weights, k=8, chunk 512/64, budget caps) — nothing magic in code (Rules §2)
- [x] Rate-limit middleware + budget-guard env vars (PRD security NFR; Master-Prompt Rule 8)
- [x] Pinned frontend toolchain (React 18, Vite 5, Node 20, pnpm 9) asserted in CI
- [x] README quickstart

**Exit:** `docker compose up` boots all services; `/health` returns 200.

---

## Phase 1 - Core RAG (Week 1)

- [x] Document ingestion: PDF/TXT/MD/URL parse -> chunk -> embed -> store (URL via trafilatura, PRD FR-1)
- [x] Retrieval endpoint: top-k semantic search
- [x] Basic chat endpoint: grounded prompt -> answer (no verification yet)
- [x] Postgres: documents + messages tables, chat history endpoint (`GET /sessions/{id}`)
- [x] Minimal chat UI: upload box + message thread

**Exit:** Can upload 3 PDFs, ask questions, get sourced-looking answers end-to-end.

---

## Phase 2 - Citation Grounding (Week 2) - core differentiator

- [x] Structured output: answer + claims + citations
- [x] Citation verifier: chunk_id existence + support checks
- [x] Verify -> regenerate loop (max 1 retry), reject on repeat failure
- [x] Citation cards in UI (quote, doc, page)
- [x] pytest: verifier rejects fabricated chunk_ids 100% of the time

**Exit:** Every displayed claim has a real, clickable citation; fabricated citations can never reach the UI (proven by tests).

---

## Phase 3 - Confidence & Fallback (Week 3)

- [x] Hybrid confidence scorer (retrieval + faithfulness + coverage)
- [x] Threshold router: answer / flag / fallback
- [x] Tavily/Brave fallback with live-source citations
- [x] UI badges (green/yellow/blue) + warning copy
- [x] Streaming: `status` event immediately, then verified answer tokens (first verified token < 3s)
- [x] Logging of all scores + routing decisions to Postgres
- [x] pytest: mock weak retrieval -> answer is flagged or falls back, never confident

**Exit:** A demo question *outside* the corpus gets flagged or web-answered - never hallucinated.

---

## Phase 4 - Evals, Polish, Deploy (Week 4)

- [x] Golden dataset: 30-50 Q&A pairs over test corpus (include adversarial/out-of-corpus questions)
- [x] RAGAS harness: faithfulness + answer relevancy + citation precision
- [x] Threshold tuning from eval results; record before/after numbers for README
- [x] Latency load-test harness with a P95 < 15s gate
- [ ] Run the latency gate against the live deployment and record the result
- [x] Rate limiting + error handling sweep
- [x] Dockerize + deployment configs (Render + Vercel + Neon)
- [ ] Deploy the public app and record its URL
- [x] README: architecture diagram + offline eval table
- [ ] README: demo GIF/video

**Exit:** Live public URL, eval table in README, repo link on the resume.

---

## Definition of Done (whole project)

1. Live deployed app
2. Zero unflagged unsupported claims in eval run
3. Faithfulness >= 0.85 on golden set
4. Demo video < 90s: upload -> grounded answer -> flagged low-confidence -> web fallback
5. README a stranger can follow in < 10 minutes

The remaining human/account-dependent steps are ordered in
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).

## Post-v1 Wishlist (do NOT start until v1 ships)

- Hybrid BM25 + vector retrieval
- NLI-based claim verification on all claims
- Per-document ACLs
- Confidence dashboard UI
- Ollama local-LLM mode
