# ADR-0003: OpenAI-compatible endpoint override (OPENAI_API_BASE)

- **Status:** Accepted
- **Date:** 2026-09-22
- **Deciders:** Project author

## Context

The stack fixes OpenAI as the generation/embedding provider (CLAUDE.md), but the
API key available during Phase 4 hit `credit_balance_exhausted`, and OpenAI
billing requires a card that the project author did not have. Google's Gemini
API offers a free tier with an **OpenAI-compatible endpoint**
(`https://generativelanguage.googleapis.com/v1beta/openai/`) that speaks the
same `/chat/completions` and `/embeddings` protocol.

Meanwhile the only OpenAI entry points in the codebase are two LlamaIndex
clients (`OpenAILLM`, `OpenAIEmbedder`), both of which accept an `api_base`.

## Decision

1. A single env var, **`OPENAI_API_BASE`**, overrides the endpoint for every
   OpenAI-protocol client in the app: generation, embeddings, and the RAGAS
   judge (EVAL.md §3) alike. Unset/blank means the real `api.openai.com`.
2. The override is **env-only** (`.env` / Render dashboard), never committed
   config: endpoints are deployment-specific (Rules.md §2, §5).
3. `config.yaml` remains the source of truth for **which model** runs
   (`generation.model`, `embedding.model`); the env var only changes **where**
   the OpenAI protocol is served from. Switching to Gemini therefore also edits
   `config.yaml`'s model names to a Gemini-supported slug (e.g.
   `gemini-2.0-flash`, `text-embedding-004`).
4. Collection namespacing (ADR-0001) keeps incompatible embedding vectors
   apart automatically: the effective collection is
   `{prefix}_{embedding.model slug}`, so an endpoint/model switch forces a
   fresh collection and full re-ingestion rather than silent mixing.

## Consequences

- **Positive:** the whole pipeline (chat, load test, RAGAS scoring) runs on a
  free tier with zero vendor-protocol code; returning to OpenAI is a one-line
  `.env` change (clear `OPENAI_API_BASE`, restore model names).
- **Negative:** the budget guard's `price_per_million_usd` estimates assume
  OpenAI list prices; on Gemini free tier they overestimate spend, which is
  safe (caps trip early, never late).
- **Risks:** Gemini's JSON-mode semantics differ subtly from OpenAI's. The
  structured-output schema is validated downstream (`rag/generation.py`), and
  the verify/regenerate loop rejects malformed payloads, so a divergence shows
  up as a refusal/retry, never as an unverified claim. The RAGAS judge wrapper
  uses LangChain's `ChatOpenAI` with the same `base_url`, so scoring cannot
  silently call a different vendor than the one evaluated.

## Alternatives considered

- **OpenAI debit-card top-up** — blocked on author's payment access at the
  time; remains the production default (`OPENAI_API_BASE` unset).
- **Groq provider** (already in `GenerationSection`) — free tier exists but
  covers generation only, not embeddings, and would need a second provider
  branch in `Runtime`.
- **Local `sentence-transformers` embedder** (ADR-0001 alternative) — free but
  no chat model; does not unblock generation or RAGAS scoring.
