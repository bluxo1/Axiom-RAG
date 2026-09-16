# Axiom - Project Rules

> The constitution. If code breaks these rules, the code is wrong - not the rules.

---

## 1. Grounding Rules (non-negotiable)

1. **No citation, no claim.** A factual claim without a verified citation never reaches the user. Verification is a hard gate that runs *before* confidence routing on every path — flagged and web-fallback answers included.
2. **Citations must exist.** A `chunk_id` not present in the retrieved context is a bug, whatever the LLM said.
3. **No outside knowledge.** The LLM answers from context only. If context is insufficient -> `INSUFFICIENT_EVIDENCE`, not improvisation.
4. **Uncertainty is displayed, never hidden.** Low confidence = visible badge + warning string. No exceptions.
5. **Fallback is honest.** Web-sourced answers are labeled as such, with live URLs.
6. **"I don't know" is a valid answer.** A flagged "I don't know" beats a confident hallucination.
7. **Every routing decision is logged.** Question, retrieved chunks, scores, threshold, outcome - persisted for tuning and demos.

## 2. Code Rules

- Python 3.11+, typed end-to-end (mypy strict on new modules)
- ruff for lint/format; no lint warnings merged
- All I/O at the edges: pure functions in `rag/`, `confidence/`; side effects in `api/`
- LLM calls behind an interface (`llm/provider.py`) so providers are swappable
- No secrets in code - env vars only; `.env.example` documents every var
- Config (thresholds, weights, k) lives in `config.yaml`, not scattered in code

## 3. Testing Rules

- Every phase has at least one test that would have caught its worst failure:
  - verifier rejects fabricated chunk_ids
  - weak retrieval -> flag/fallback (never confident)
  - fallback labels sources as web
- Golden eval set is versioned; eval scores must not regress between PRs
- LLM-dependent tests use recorded fixtures, never live calls in CI

## 4. Git & Commit Rules

- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`
- One PR = one phase task; PR description includes what/how-tested/screenshot-if-UI
- `main` always deployable; work happens on `feat/*` branches

## 5. Documentation Rules

- README quickstart must work from a clean clone (tested each phase)
- Every env var documented in `.env.example` with a comment
- ADRs live in `docs/adr/` for decisions that are hard to reverse (vector DB choice, embedding model choice, threshold strategy)
- Public claims (eval numbers, benchmarks) must be reproducible from `evals/`

## 6. Anti-Patterns (banned)

- Passing raw LLM output to the UI without verification
- `except: pass`
- "Temporary" hardcoded values that survive a phase exit
- Adding a feature not in Phases.md without updating Phases.md first
- Shipping a UI state that shows confidence without a backend score behind it
