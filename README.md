# Axiom

> Start from what you can prove.

A citation-grounded RAG agent. Every claim is tied to a retrieved, verified source; low-confidence answers are visibly flagged; when retrieval fails, Axiom falls back to live web search instead of guessing.

**The measurable goal: zero unsupported claims reach the user unflagged.**

## How it works

Documents → chunks → embeddings → vector store. At query time: retrieve → generate with inline citations → **verify every citation (hard gate)** → score confidence → answer / flag / fall back to web search. Every routing decision is logged.

See [docs/Architecture.md](docs/Architecture.md) for the full design.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/Prompt.md](docs/Prompt.md) | Master build prompt (v1.1) |
| [docs/PRD.md](docs/PRD.md) | Product requirements |
| [docs/Architecture.md](docs/Architecture.md) | System design, pipeline, tech stack |
| [docs/Design.md](docs/Design.md) | API contracts, data models, prompts, UX |
| [docs/Rules.md](docs/Rules.md) | Non-negotiable project rules |
| [docs/Phases.md](docs/Phases.md) | Build order + exit criteria |
| [docs/EVAL.md](docs/EVAL.md) | Evaluation strategy + targets |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Workflow + merge checklist |
| [docs/adr/](docs/adr/) | Architecture decision records |

## Quickstart

### With Docker (the whole stack)

Boots FastAPI + Postgres + Chroma together. This is what the Phase 0 exit
criterion checks.

```bash
cp .env.example .env
docker compose up --build --wait
curl http://localhost:8000/health
```

`/health` returns `200` with `{"status":"ok","service":"Axiom",...}`. Tear down
with `docker compose down -v`.

### Backend only (local Python)

Requires Python 3.11.

```bash
cd backend
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Then `curl http://localhost:8000/health`. Run the quality gates the same way CI
does:

```bash
cd backend
ruff check . && ruff format --check . && mypy && pytest
```

### Frontend only (local Node)

Requires Node 20 and pnpm 9 (the versions are pinned and asserted).

```bash
cd frontend
corepack enable
pnpm install --frozen-lockfile
pnpm run assert:toolchain
pnpm run dev        # http://localhost:5173
```

`pnpm run build` produces the production bundle; CI runs the same build.

## Configuration

- **Knobs** (thresholds, weights, `k`, chunk size, budget defaults) live in
  [`config.yaml`](config.yaml). Nothing tunable is hardcoded.
- **Secrets and per-environment values** live in `.env` only. Copy
  [`.env.example`](.env.example) — every variable is documented there.
- Budget caps (`MAX_REQUEST_TOKENS`, `DAILY_TOKEN_CAP`, `MONTHLY_SPEND_USD`) set
  in `.env` override the `budget.*` defaults in `config.yaml`.

## Repository layout

```
backend/    FastAPI app, config loader, structured errors, rate limiting, tests
frontend/   React 18 + Vite 5 + Tailwind 3 (pinned), health-check shell
data/        Ingested corpora and the local vector store (gitignored, disposable)
evals/       Golden set and RAGAS reports (Phase 4)
docs/        Spec: PRD, Architecture, Design, Rules, Phases, ADRs
```

## Status

**Phase 0 (Setup & Skeleton) — complete.** Repo skeleton, docker-compose stack,
CI (ruff + mypy + pytest, frontend build, compose health check), config loader,
`config.yaml`, rate-limit middleware, and the pinned frontend toolchain are in
place. Next: Phase 1 (core RAG — ingestion, retrieval, chat). See
[docs/Phases.md](docs/Phases.md).
