# Axiom — Design Document

> API contracts, data models, and prompt/UX design.

---

## 1. API Design

Base URL: `/api/v1`

### 1.1 Documents
```
POST   /documents            multipart upload (PDF/TXT/MD)
POST   /documents/url        ingest from URL {url}
GET    /documents            list ingested docs
DELETE /documents/{doc_id}
```
Response: `{doc_id, name, chunks: n, status: "ready" | "processing" | "failed"}`

### 1.2 Chat
```
POST /chat
{
  "question": "string",
  "session_id": "uuid (optional)"
}
```

Response (structured):
```json
{
  "answer": "The policy covers ... [c1] however pre-existing ... [c2]",
  "citations": [
    {"id": "c1", "chunk_id": "doc9#14", "doc": "policy.pdf", "page": 3,
     "quote": "coverage includes ...", "source": "kb"},
    {"id": "c2", "chunk_id": "web#2", "doc": "live web", "url": "https://...",
     "quote": "...", "source": "web"}
  ],
  "confidence": {
    "score": 0.81,
    "level": "high",
    "breakdown": {"retrieval": 0.9, "faithfulness": 0.8, "coverage": 0.75}
  },
  "flagged": false,
  "fallback_used": false,
  "session_id": "uuid"
}
```

Citation id convention: the LLM emits raw `[chunk_id]` markers (e.g. `[doc9#14]`); the verifier rewrites every *verified* marker to a sequential display id (`[c1]`, `[c2]`, ...) in both `answer` and `citations[].id`. The API always exposes display ids; the `id` ↔ `chunk_id` mapping above is how the UI resolves a marker back to its source card.

Flagged variant: same schema with `"level": "low"`, `flagged: true`, plus `"warning": "Sources are weak - verify before relying on this."`

### 1.3 Meta
```
GET /health          liveness
GET /sessions/{id}   chat history
GET /metrics         flagged-rate, fallback-rate, spend counters (Rule 8; for the demo dashboard)
```

## 2. Data Models

### 2.1 Chunk (vector store payload)
```json
{
  "chunk_id": "sha1(doc_id:index)",
  "doc_id": "uuid",
  "doc_name": "string",
  "page": 3,
  "text": "string",
  "embedding": ["..."],
  "metadata": {"ingested_at": "iso8601", "source": "kb"}
}
```

### 2.2 Postgres Tables
```sql
documents(doc_id PK, name, status, created_at)
chunks(chunk_id PK, doc_id FK, page, text)
sessions(session_id PK, created_at)
messages(msg_id PK, session_id FK, question, answer_json,
         confidence, confidence_breakdown jsonb, router_decision,
         retrieved_chunk_ids jsonb, flagged, fallback_used,
         latency_ms, created_at)
-- router_decision: 'answer' | 'flag' | 'fallback' | 'refusal'
-- (Rules.md #7: every routing decision persisted with retrieved chunks + scores)

spend_log(spend_id PK, created_at, provider, kind,   -- kind: 'llm' | 'embedding' | 'search'
          tokens, estimated_cost_usd)
-- (Prompt.md Rule 8: per-call spend counters; daily/monthly caps checked against this table;
--  GET /metrics aggregates it. Exceeding a cap -> 429 BUDGET_EXCEEDED.)
```

## 3. Prompt Design

### 3.1 Grounded Generation Prompt (system)
```
You are Axiom, an assistant that answers ONLY from the provided context.

Rules:
1. Every factual claim MUST end with a citation marker [chunk_id] from the context.
2. NEVER cite a chunk_id that is not present in the context.
3. If the context does not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE
4. Do not use outside knowledge.

Context:
{chunks}
```

> Structured-output mode: rule 3 becomes `status: "insufficient_evidence"` in the JSON schema (§3.3) instead of a bare string. Raw `[chunk_id]` markers emitted by the model are rewritten to sequential display ids (`[c1]`, ...) by the verifier before the response leaves the API.

### 3.2 Faithfulness Self-Check Prompt
```
For each (claim, cited_chunk) pair, reply VALID or INVALID with a one-line reason.

Claim: {claim}
Cited text: {chunk}
```

### 3.3 Structured Output
Enforced via `response_format` / tool calling:
```
{status: "answered" | "insufficient_evidence",
 answer: str,
 claims: [{text: str, citation_ids: [str]}],
 citations: [{citation_id: str, chunk_id: str}]}
```
`citation_ids` use raw chunk-based markers at generation time; the verifier (Architecture §3.4) rewrites verified citations to display ids `c1, c2, ...` in the final API response.

## 4. Citation UX Design

- Inline markers in the answer, e.g. `...covered by the policy [1]`
- Clicking a marker opens a source card: document name, page, quoted text, retrieval score
- Confidence badge on every answer:
  - GREEN - Grounded (score >= 0.75)
  - YELLOW - Low confidence (0.45-0.75) - "verify before relying"
  - BLUE - Web sources - "answered via live search"
- If `INSUFFICIENT_EVIDENCE` and fallback fails: an honest "I don't know" card

## 5. Error Handling

| Case | Behavior |
|------|----------|
| No docs ingested | 400 + friendly message, CTA to upload |
| LLM timeout | 504, retry once, then structured error |
| Verification fails after retry | Return flagged answer with warning |
| Search API down | Flag + explain fallback unavailable |
| Malformed LLM JSON | Auto-repair pass (re-ask to fix JSON), else 502 |

## 6. Rate Limiting & Costs
- Per-session token bucket (dev: generous; prod: configurable)
- Budget guard on every LLM/embedding/search call (Rule 8): per-request token limit (`MAX_REQUEST_TOKENS`), daily token cap (`DAILY_TOKEN_CAP`), monthly USD cap (`MONTHLY_SPEND_USD`) — env-configured, wired through `config.yaml`. Each call appends to the `spend_log` table; exceeding a cap returns `429 BUDGET_EXCEEDED`, never a silent surprise bill.
- Cache answers by (question_hash, doc_set_hash) to avoid repeat spend; `doc_set_hash` = sha1 over the sorted `(doc_id, status)` pairs of the currently ingested corpus (invalidated by any ingest/delete)
