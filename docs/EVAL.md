# Axiom - Evaluation Strategy

> Axiom's core claim is measurable. This file defines how we prove it.

---

## 1. Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| **Citation precision** | % of citations that actually support their claim | >= 95% |
| **Faithfulness (RAGAS)** | Is the answer entailed by retrieved context? | >= 0.85 |
| **Answer relevancy (RAGAS)** | Does the answer address the question? | >= 0.80 |
| **Unsupported-claim escape rate** | Claims reaching the user without verified support | **0** |
| **Fallback precision** | % of fallbacks that were truly necessary (manual review) | >= 90% |
| **P95 latency** | 95th percentile time-to-full-answer | < 15s |

## 2. Golden Dataset

`evals/golden.jsonl` - 30-50 entries:

```json
{"question": "...", "expected_answer": "...", "must_cite": ["doc#chunk"], "category": "in-corpus | out-of-corpus | adversarial"}
```

Categories:
- **in-corpus** (60%): answer exists in test docs
- **out-of-corpus** (20%): answer NOT in docs -> expect fallback or flagged refusal
- **adversarial** (20%): misleading/trick questions -> expect flag or refusal, never confident fabrication

## 3. Running Evals

```bash
pytest evals/ -v --run-llm   # live LLM eval (costs tokens)
pytest evals/ -v             # recorded-fixture mode (CI default)
```

## 4. Hallucination Test (the money test)

`evals/test_hallucination.py` asserts, over the adversarial set:

1. No answer contains a citation whose chunk_id was not retrieved
2. No out-of-corpus question receives a confident (>= 0.75) unflagged answer
3. Fallback events are labeled `source: "web"`

If any assertion fails, the build fails. This is the test that proves the tagline.

## 5. Reporting

Results written to `evals/reports/<date>.md`, summarized in README:

| Date | Faithfulness | Citation precision | Escape rate |
|------|-------------|--------------------|-------------|
