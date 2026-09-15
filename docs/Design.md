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

Flagged variant: same schema with `"level": "low"`, `flagged: true`, plus `"warning": "Sources are weak - verify before relying on this."`

### 1.3 Meta
```
GET /health          liveness
GET /sessions/{id}   chat history
GET /metrics         flagged-rate, fallback-rate (for the demo dashboard)
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
         confidence, flagged, fallback_used, latency_ms, created_at)
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

### 3.2 Faithfulness Self-Check Prompt
```
For each (claim, cited_chunk) pair, reply VALID or INVALID with a one-line reason.

Claim: {claim}
Cited text: {chunk}
```

### 3.3 Structured Output
Enforced via `response_format` / tool calling:
`{answer: str, claims: [{text: str, citation_ids: [str]}]}`

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
- Cache answers by (question_hash, doc_set_hash) to avoid repeat spend
