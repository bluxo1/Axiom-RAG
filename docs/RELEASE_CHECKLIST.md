# Axiom — v1 Release Checklist

The public app is deployed and its backend health check returns HTTP 200. The
remaining v1 evidence is the functional browser smoke test, live RAGAS scores,
live latency result, and demo video. Public-API security hardening was merged in
[PR #10](https://github.com/bluxo1/Axiom-RAG/pull/10). Confirm Render has
deployed the merged revision before completing the live smoke tests below.

Never paste secrets into GitHub, screenshots, terminal transcripts, or this
repository. Store them only in the provider dashboards and the ignored `.env`.

## 1. Deploy the public app

Current deployment: [frontend](https://axiom-rag.vercel.app/) and
[backend /health](https://axiom-backend-7n5c.onrender.com/health). The health
endpoint was verified live; finish the three functional browser smoke tests
below.

Follow [`DEPLOY_RUNBOOK.md`](DEPLOY_RUNBOOK.md) in order:

1. Create Neon Postgres and retain the pooled `DATABASE_URL`.
2. Apply `render.yaml` as a Render Blueprint. Supply `DATABASE_URL`, the Gemini
   key as `OPENAI_API_KEY`, and `TAVILY_API_KEY`. The blueprint selects the
   production pgvector backend in Neon; no separate vector service is needed.
3. Confirm `https://<backend>/health` returns HTTP 200.
4. Import `frontend/` into Vercel and set `VITE_API_BASE_URL` to
   `https://<backend>/api/v1`.
5. Set `CORS_ORIGINS=https://<frontend>` on the Render backend and redeploy.
6. Complete the three browser smoke tests in the deploy runbook.

Record both public URLs in the README and set the frontend URL as the GitHub
repository website.

## 2. Capture live RAGAS evidence

This step uses the configured Gemini key and spends tokens. From a current
checkout of `main`, start the local Docker stack, then run:

```powershell
docker compose up -d --build --wait
Set-Location backend
python -m pip install -e ".[dev,eval]"
pytest evals/ -v -s --run-llm
```

The command prints faithfulness and answer-relevancy scores and fails if they
miss 0.85 and 0.80 respectively. Copy the results into a dated file under
`backend/evals/reports/` and replace the `--run-llm` placeholders in README.

## 3. Capture live latency evidence

With the public backend warm, run this from `backend/` (it performs 30 paid chat
requests and uploads the small Zephyr test document):

```powershell
$env:AXIOM_BACKEND_URL = "https://<backend>"
python -m evals.loadtest `
  --base-url $env:AXIOM_BACKEND_URL `
  --document evals/corpus/zephyr-api.txt `
  --question "What status code is returned when the rate limit is exceeded?" `
  --requests 30 `
  --concurrency 5
```

The command exits successfully only when all requests return 200 and P95
full-answer latency is below 15 seconds. Record its output in the dated report.
Warm the Render service before measuring; free-tier cold starts are not a steady
state latency measurement.

## 4. Record the demo

Record a video shorter than 90 seconds showing:

1. Upload a document.
2. Ask an in-corpus question and open its verified citation.
3. Ask a weak/ambiguous question and show the visible low-confidence warning.
4. Ask an out-of-corpus question and show the web-fallback badge and live URL.

Add the GIF/video link near the top of README. Do not expose dashboard tabs,
environment variables, API keys, or connection strings while recording.

## 5. Close Phase 4

After the four steps above:

- Mark the remaining Phase 4 boxes in [`Phases.md`](Phases.md).
- Change README status from “final live release evidence is pending” to “v1
  complete.”
- Run the normal backend, frontend, and Compose gates.
- Commit the evidence/docs on a `feat/*` branch and merge it through CI.

At that point every item in the project Definition of Done is evidenced.
