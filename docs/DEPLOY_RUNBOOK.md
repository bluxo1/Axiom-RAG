# Axiom — Deploy Runbook

> Phases.md Phase 4: a live public URL. This is the stranger-followable path from
> a clean checkout to a running deploy in well under an hour.

## Topology

| Piece | Host | Source of truth |
|-------|------|-----------------|
| Backend (FastAPI) | **Render** web service, Docker | [`render.yaml`](../render.yaml), [`backend/Dockerfile`](../backend/Dockerfile) |
| Vector store (ChromaDB) | **Render** private service | [`render.yaml`](../render.yaml) |
| Postgres | **Neon** (managed) | `DATABASE_URL` secret |
| Frontend (React/Vite) | **Vercel** static build | [`frontend/vercel.json`](../frontend/vercel.json) |

The backend image bakes `config.yaml` (build context is the repo root), so it
boots standalone — no bind mount. Tables are created on first startup
(`app/main.py` lifespan), so there is no separate migration step.

> **Drift flag (Architecture.md §5 / ADR-0002).** Prod is *specified* to use
> pgvector, but only the Chroma backend is implemented today, so prod runs Chroma
> as a private service. ADR-0002 makes the vector backend a config change, so when
> pgvector lands: implement it in `app/vector/`, flip `vector_store.backend` to
> `pgvector`, drop the `axiom-chroma` service from `render.yaml`, and point it at
> the Neon `DATABASE_URL`. Until then, keep the Chroma service.

## Prerequisites

- GitHub repo connected to Render and Vercel.
- Accounts: [Render](https://render.com), [Vercel](https://vercel.com), [Neon](https://neon.tech).
- API keys: `OPENAI_API_KEY` (embeddings + generation), `TAVILY_API_KEY` (web fallback).

## 1. Postgres on Neon

1. Create a Neon project → copy the **pooled** connection string.
2. Rewrite it to the SQLAlchemy dialect the app expects (`app/config.py` default
   is `postgresql+psycopg://`):

   ```
   postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require
   ```

   Keep `sslmode=require` — Neon rejects plaintext. Hold this for step 2.

## 2. Backend + vector store on Render

1. **New → Blueprint**, pick this repo. Render reads [`render.yaml`](../render.yaml)
   and proposes `axiom-backend` (web) and `axiom-chroma` (private service).
2. Set the backend's secret env vars (declared `sync: false`, so Render prompts):
   - `DATABASE_URL` → the Neon string from step 1.
   - `OPENAI_API_KEY`, `TAVILY_API_KEY` → your keys.

   `CHROMA_HOST`/`CHROMA_PORT` wire to the Chroma service automatically; budget
   caps and `ENABLE_HSTS=true` come from the blueprint.
3. **Apply**. Render builds the image (baking `config.yaml`), starts Chroma with a
   1 GB persistent disk, and gates the backend on `GET /health`.
4. Copy the backend URL, e.g. `https://axiom-backend.onrender.com`.

Verify:

```bash
curl https://axiom-backend.onrender.com/health
```

Expect `200` with `{"status":"ok","service":"Axiom","environment":"prod",...}`.

## 3. Frontend on Vercel

1. **New Project** → this repo. Set **Root Directory = `frontend`** so Vercel picks
   up [`frontend/vercel.json`](../frontend/vercel.json) (Vite build, `dist` output,
   SPA rewrites) and the pinned pnpm from `package.json`.
2. Add a build-time environment variable:
   - `VITE_API_BASE_URL` = `https://axiom-backend.onrender.com/api/v1`
     (the backend URL from step 2 **plus the `/api/v1` prefix**).
3. **Deploy**, then copy the Vercel URL, e.g. `https://axiom.vercel.app`.

## 4. Allow the frontend origin (CORS)

CORS origins live in [`config.yaml`](../config.yaml) (`app.cors_origins`) and are
baked into the backend image, so the browser origin must be added and the backend
redeployed:

1. Add the Vercel origin to `app.cors_origins` in `config.yaml`:

   ```yaml
   app:
     cors_origins:
       - http://localhost:5173
       - http://127.0.0.1:5173
       - https://axiom.vercel.app   # prod frontend
   ```

2. Commit (config change → commit the doc-of-record together, per Rules.md §2) and
   redeploy `axiom-backend` on Render (**Manual Deploy → Deploy latest commit**).

## 5. Smoke test (the Definition of Done, live)

From the Vercel URL:

1. Upload a document → ask a question answered by it → **green** badge, citation
   cards with clickable sources.
2. Ask something outside the corpus → **blue** web-fallback badge, live URLs.
3. Ask a trick/false-premise question → **yellow** flag or an honest refusal —
   never a confident fabrication.

If any answer is confident *and* unsupported, stop and treat it as a release
blocker (EVAL.md §4): re-run `pytest backend/evals/` before shipping further.

## Operational notes

- **Budget guard (Rule 8).** `MAX_REQUEST_TOKENS` / `DAILY_TOKEN_CAP` /
  `MONTHLY_SPEND_USD` on the Render service cap spend; exceeding one returns
  `429 BUDGET_EXCEEDED`, never a silent bill. Raise them deliberately.
- **Secrets** live only in Render/Vercel dashboards and Neon — never in the repo
  (Rules.md §2, §5). `.env` is git-ignored; `.env.example` documents every var.
- **HSTS** is on in prod (`ENABLE_HSTS=true`) because Render terminates TLS; it
  stays off in plain-HTTP dev.
- **Free-tier cold starts.** Render's free/starter web service sleeps when idle,
  so the first request (and `first verified token < 3s`, Phase 3) can be slow
  until it wakes. Use a paid instance for latency SLAs.

## Rollback

- **Backend:** Render → the service → **Deploys** → **Rollback** to the last good
  deploy (or **Deploy** a specific commit).
- **Frontend:** Vercel → **Deployments** → **Promote** a previous deployment.
- **Config-only regression:** revert the `config.yaml` commit and redeploy the
  backend; knobs are versioned, so a bad threshold/weight change is a one-commit
  revert.

## Teardown

Delete the Render Blueprint (removes both services and the Chroma disk), delete
the Vercel project, and delete the Neon project. Nothing else persists.
