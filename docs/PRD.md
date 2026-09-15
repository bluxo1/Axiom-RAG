# Axiom — Product Requirements Document

> Start from what you can prove.

**Version:** 1.0 | **Status:** Draft | **Last updated:** 2026-09-16

---

## 1. Overview

Axiom is a retrieval-augmented generation (RAG) agent that answers questions **only with evidence**. Every claim is tied to a retrieved source, low-confidence answers are visibly flagged, and when retrieval fails, Axiom falls back to live web search instead of guessing.

**One-liner:** A RAG agent with citation grounding, confidence scoring, and live-search fallback.

---

## 2. Problem Statement

LLM chatbots hallucinate confidently. Existing RAG demos retrieve context but:
- rarely cite sources verifiably,
- never admit uncertainty,
- silently fall back to parametric memory (a polite term for making things up).

For any use case where a wrong answer has a cost (legal, medical, academic, enterprise), this is a dealbreaker.

---

## 3. Goals

| # | Goal | Success measure |
|---|------|-----------------|
| G1 | Every answer claim is grounded in a retrieved source | 100% of claims carry a verifiable citation |
| G2 | Unsupported/low-confidence answers are flagged, not hidden | UI badge + structured `confidence` field |
| G3 | Retrieval failure triggers web-search fallback | Fallback events logged and labeled |
| G4 | Measurable hallucination reduction | RAGAS faithfulness ≥ target on golden set |
| G5 | Internship-portfolio quality | Deployed live app + public repo + demo video |

## 4. Non-Goals (v1)

- Multi-user SaaS with billing
- Fine-tuning custom models
- Voice/multimodal input
- Mobile apps
- Multi-step agentic tool chains beyond the fallback

## 5. Target Users

1. **Primary:** internship recruiters / technical reviewers evaluating the author
2. **Secondary:** anyone who wants grounded answers over a private document set (students, researchers)

## 6. User Stories

- **US-1:** As a user, I upload PDFs and ask questions so that I get answers with inline citations I can click to verify.
- **US-2:** As a user, when sources are weak, I see a visible low-confidence warning instead of a confident wrong answer.
- **US-3:** As a user, when my documents don't contain the answer, Axiom searches the web and cites live sources.
- **US-4:** As a reviewer, I can run an eval suite that reports faithfulness and citation accuracy.

## 7. Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Document ingestion (PDF, TXT, MD, URL) with chunking + embedding | P0 |
| FR-2 | Semantic retrieval over the vector store | P0 |
| FR-3 | Answer generation with structured citations `[chunk_id]` | P0 |
| FR-4 | Citation verification (every citation must exist in retrieved context) | P0 |
| FR-5 | Confidence score per answer (hybrid: retrieval + faithfulness + coverage) | P0 |
| FR-6 | Confidence flagging below threshold in API + UI | P0 |
| FR-7 | Web-search fallback (Tavily/Brave) when retrieval is empty/weak | P0 |
| FR-8 | Chat history persisted in Postgres | P1 |
| FR-9 | Streaming responses | P1 |
| FR-10 | Eval harness (RAGAS) + golden dataset | P1 |

## 8. Non-Functional Requirements

- **Latency:** first token < 3s (streaming); full answer < 15s for 4k-token contexts
- **Cost:** dev runs on free tiers / local models; API spend capped via env-configured limits
- **Reliability:** structured errors; never a silent 500
- **Observability:** query, retrieved docs, scores, and fallback events logged to Postgres
- **Security:** API keys in env vars only; no secrets in repo; rate-limited endpoints

## 9. Success Metrics

| Metric | Target |
|--------|--------|
| Citation precision (citations that support their claim) | ≥ 95% |
| RAGAS faithfulness on golden set | ≥ 0.85 |
| Unsupported claims reaching the user unflagged | **0** |
| Fallback precision (fallbacks that were truly necessary) | ≥ 90% |

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| LLM cites documents not in context | Verification layer rejects/repairs before responding |
| Confidence threshold mis-tuned | Log everything; tune on golden set |
| Free-tier rate limits (embeddings/search) | Caching, batching, local fallback models |
| Scope creep into full SaaS | Non-goals enforced; phases have exit criteria |

## 11. Out of Scope

See §4. Anything not listed there goes to the v2 wishlist, not the build.
