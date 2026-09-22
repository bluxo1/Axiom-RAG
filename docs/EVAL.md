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

> **Note:** RAGAS ships faithfulness and answer relevancy out of the box, but
> **not** citation precision. Citation precision is a **custom scorer** in
> `evals/`, built on the grounding verifier's support check.

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
cd backend
pip install -e ".[dev,eval]" # once; live-only dependencies are optional
pytest evals/ -v             # recorded-fixture mode (CI default)
pytest evals/ -v -s --run-llm # live scores printed; costs tokens
```

The optional RAGAS extra is never installed in production or CI. Current
RAGAS advisories `PYSEC-2026-3046` and `PYSEC-2026-3047` affect multimodal
contexts that dereference caller-provided URLs or file paths. Axiom's live
runner passes only trusted text contexts built from `evals/corpus/`; do not
adapt it to score untrusted multimodal input until RAGAS publishes a fix.

## 4. Hallucination Test (the money test)

`evals/test_hallucination.py` asserts, over the adversarial set:

1. No answer contains a citation whose chunk_id was not retrieved
2. No out-of-corpus question receives a confident (>= 0.75) unflagged answer
3. Fallback events are labeled `source: "web"`

If any assertion fails, the build fails. This is the test that proves the tagline.

## 5. Reporting

Record release results in `evals/reports/<date>.md` and summarize them in the
README:

| Date | Faithfulness | Citation precision | Escape rate |
|------|-------------|--------------------|-------------|
