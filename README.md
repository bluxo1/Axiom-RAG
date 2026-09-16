# Axiom

> Start from what you can prove.

A citation-grounded RAG agent. Every claim is tied to a retrieved, verified source; low-confidence answers are visibly flagged; when retrieval fails, Axiom falls back to live web search instead of guessing.

**The measurable goal: zero unsupported claims reach the user unflagged.**

## How it works

Documents → chunks → embeddings → vector store. At query time: retrieve → generate with inline citations → **verify every citation (hard gate)** → score confidence → answer / flag / fall back to web search. Every routing decision is logged.

See [docs/Architecture.md](docs/Architecture.md) for the full design.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/Prompt.md](docs/Prompt.md) | Master build prompt (v1.1) |
| [docs/PRD.md](docs/PRD.md) | Product requirements |
| [docs/Architecture.md](docs/Architecture.md) | System design, pipeline, tech stack |
| [docs/Design.md](docs/Design.md) | API contracts, data models, prompts, UX |
| [docs/Rules.md](docs/Rules.md) | Non-negotiable project rules |
| [docs/Phases.md](docs/Phases.md) | Build order + exit criteria |
| [docs/EVAL.md](docs/EVAL.md) | Evaluation strategy + targets |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Workflow + merge checklist |
| [docs/adr/](docs/adr/) | Architecture decision records |

## Status

Spec complete — implementation starting at **Phase 0**. Quickstart will land here as part of Phase 0's exit criteria.
