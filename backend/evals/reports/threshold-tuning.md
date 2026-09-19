# Axiom threshold tuning — 2026-09-19 00:01 UTC

Run over the golden set offline; the sweep **confirmed** the a-priori defaults (high=0.75, low=0.45).

- In-corpus score floor: **0.9048**
- Noise margin below the floor: **0.15** → high = floor - margin, rounded down to 0.05 = **0.75**
- `low` has no empirical pressure offline (no genuine answer scores near it) and is left at **0.45**

## Before / after

| Thresholds | In-corpus green | In-corpus flagged | Web fallbacks | Escapes |
|------------|-----------------|-------------------|---------------|---------|
| before (default) (high=0.75, low=0.45) | 18/18 | 0 | 6 | 0 |
| after (tuned) (high=0.75, low=0.45) | 18/18 | 0 | 6 | 0 |

## Sweep (fixed low=0.45)

| high | In-corpus green recall | Flagged | Escapes |
|------|------------------------|---------|---------|
| 0.60 | 100% (18/18) | 0 | 0 |
| 0.65 | 100% (18/18) | 0 | 0 |
| 0.70 | 100% (18/18) | 0 | 0 |
| 0.75 | 100% (18/18) | 0 | 0 |
| 0.80 | 100% (18/18) | 0 | 0 |
| 0.85 | 100% (18/18) | 0 | 0 |
| 0.90 | 100% (18/18) | 0 | 0 |
| 0.95 | 17% (3/18) | 15 | 0 |

Escapes stay **0** across the whole grid: out-of-corpus questions refuse on the corpus and are web-answered, adversarial questions refuse — neither produces a corpus-grounded confident answer at any threshold. In-corpus recall holds at 100% until `high` crosses the score floor, then collapses.
