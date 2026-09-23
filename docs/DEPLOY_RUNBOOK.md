# Axiom — Deploy Runbook

> Phases.md Phase 4: a live public URL. This is the stranger-followable path from
> a clean checkout to a running deploy in well under an hour.

## Topology

| Piece | Host | Source of truth |
|-------|------|-----------------|
| Backend (FastAPI) | **Render** web service, Docker | [`render.yaml`](../render.yaml), [`backend/Dockerfile`](../backend/Dockerfile) |
| Postgres + vector store (pgvector) | **Neon** (managed) | `DATABASE_URL` secret, ADR-0002 |
| Frontend (React/Vite) | **Vercel** static build | [`frontend/vercel.json`](../frontend/vercel.json) |

The backend image bakes `config.yaml` (build context is the repo root), so it
boots standalone — no bind mount. Tables are created on first startup
(`app/main.py` lifespan), so there is no separate migration step.

Production sets `VECTOR_STORE_BACKEND=pgvector`; the app creates the `vector`
extension and its namespaced vector table on first vector access. Local
development keeps Chroma, so both ADR-0002 backends are exercised in Compose CI.

## Prerequisites

- GitHub repo connected to Render and Vercel.
- Accounts: [Render](https://render.com), [Vercel](https://vercel.com), [Neon](https://neon.tech).
- API keys: a Gemini API key from Google AI Studio (stored as
  `OPENAI_API_KEY` for the OpenAI-compatible client) and `TAVILY_API_KEY`
  (web fallback).

## 1. Postgres on Neon

1. Create a Neon project → copy the **pooled** connection string.
2. Rewrite it to the SQLAlchemy dialect the app expects (`app/config.py` default
   is `postgresql+psycopg://`):

   ```
   postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require
   ```

   Keep `sslmode=require` — Neon rejects plaintext. Hold this for step 2.

## 2. Backend on Render

1. **New → Blueprint**, pick this repo. Render reads [`render.yaml`](../render.yaml)
   and proposes the `axiom-backend` web service.
2. Set the backend's secret env vars (declared `sync: false`, so Render prompts):
   - `DATABASE_URL` → the Neon string from step 1.
   - `OPENAI_API_KEY` → your Gemini API key from Google AI Studio.
   - `TAVILY_API_KEY` → your Tavily key.

   `OPENAI_API_BASE` is already set by the blueprint to Google's
   OpenAI-compatible endpoint, matching the committed Gemini models.
   `VECTOR_STORE_BACKEND=pgvector`, budget caps, and `ENABLE_HSTS=true` come
   from the blueprint.
3. **Apply**. Render builds the image (baking `config.yaml`) and gates the
   backend on `GET /health`. The first ingestion enables pgvector in Neon and
   creates the vector table.
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

On the Render backend, add the environment variable `CORS_ORIGINS` with the
exact Vercel origin (no path), for example:

```text
https://axiom.vercel.app
```

Use a comma-separated list if the app has more than one stable browser origin.
Save the variable and let Render redeploy the backend. This deployment-specific
override is validated at startup and avoids a source-code commit just to connect
the frontend. Blank/unset keeps the local origins from `config.yaml`.

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

- **Public demo data.** The API intentionally has no login so the demo stays
  open. CORS only restricts browser origins; it is not authentication. Every
  visitor shares the same document corpus and can upload, list, or delete its
  documents. Never ingest confidential material.
- **Request limits.** API bodies must include `Content-Length`; JSON requests are
  limited to 1 MiB. Document uploads are limited to the configured file size
  (20 MiB by default) plus bounded multipart overhead, and the file itself is
  read only up to the configured limit plus one byte before rejection.
- **URL ingestion.** Only HTTP/HTTPS on ports 80/443 is accepted. Each redirect
  is revalidated, both DNS address families must resolve to public IPs, and
  those addresses are pinned to the outbound connection. DNS lookups have a
  finite timeout, redirects are limited to four, response bodies to 5 MiB, and
  the total DNS plus fetch time to 20 seconds.
- **Rate-limit scope.** Production uses Render's `CF-Connecting-IP` value when
  it is a valid single IP; development uses the network peer address. Caller
  supplied session IDs and `X-Forwarded-For` do not choose a fresh bucket. New
  addresses share a bounded overflow bucket rather than resetting existing
  limits. Buckets are process-local, so run one worker/instance for the
  configured cap to apply as intended; a shared Redis limiter is needed before
  scaling out.
- **Chroma network isolation.** Chroma is local-development only and its Compose
  port binds to `127.0.0.1`. Keep it that way: the Chroma Python package is used
  only as an HTTP client to the native Rust server image. The current
  `PYSEC-2026-311` and `PYSEC-2026-3813` through `PYSEC-2026-3815` advisories
  affect Chroma's Python FastAPI server and have no patched PyPI release. Do not
  replace the native image with the Python server or publish port 8000. The CI
  audit documents these scoped exceptions and still fails on any new advisory.
  Upgrade promptly when Chroma ships a fixed release.
- **Transitive audit scope.** `PYSEC-2026-2447` requires an attacker who can
  already write DiskCache's local cache directory; Axiom does not expose that
  directory. `PYSEC-2026-3740` affects NLTK model-artifact APIs that Axiom does
  not call, and the installed NLTK 3.10.3 is the advisory's fixed version.
- **Production image audit (2026-09-22).** The default Docker stage excludes
  Chroma and Docker Scout reports zero critical findings. Its three remaining
  high findings (`CVE-2026-81726` in transitive NLTK and `CVE-2026-85091` /
  `CVE-2026-82560` in base-image zlib / Perl) have no fixed release. Axiom does
  not call NLTK's download or archive-extraction APIs. Rebuild and re-audit the
  image regularly, and upgrade as soon as fixes are published.
- **Budget guard (Rule 8).** `MAX_REQUEST_TOKENS` / `DAILY_TOKEN_CAP` /
  `MONTHLY_SPEND_USD` on the Render service cap spend; exceeding one returns
  `429 BUDGET_EXCEEDED`, never a silent bill. Raise them deliberately.
- **Secrets** live only in Render/Vercel dashboards and Neon — never in the repo
  (Rules.md §2, §5). `.env` is git-ignored; `.env.example` documents every var.
- **HSTS** is on in prod (`ENABLE_HSTS=true`) because Render terminates TLS; it
  stays off in plain-HTTP dev.
- **Free-tier cold starts.** Render's free web service sleeps after 15 minutes
  without inbound traffic, so the first request (and `first verified token
  < 3s`, Phase 3) can be slow until it wakes. Free instances are for demos and
  previews, not latency SLAs.

## Rollback

- **Backend:** Render → the service → **Deploys** → **Rollback** to the last good
  deploy (or **Deploy** a specific commit).
- **Frontend:** Vercel → **Deployments** → **Promote** a previous deployment.
- **Config-only regression:** revert the `config.yaml` commit and redeploy the
  backend; knobs are versioned, so a bad threshold/weight change is a one-commit
  revert.

## Teardown

Delete the Render Blueprint, the Vercel project, and the Neon project. Nothing
else persists.
