# Contributing to Axiom

Thanks for helping make Axiom trustworthy. The grounding rules in `Rules.md` apply to every contribution.

## Getting Started

1. Fork + clone
2. `cp .env.example .env` and fill in keys
3. `docker compose up`
4. `pytest` - everything must pass

## Workflow

- Pick a task from unchecked boxes in `Phases.md`
- Branch: `feat/<short-name>`
- Conventional commits
- PR must include: what changed, how tested, updated docs if behavior changed

## Definition of a Mergeable PR

- [ ] Tests pass (including the hallucination suite)
- [ ] No new mypy/ruff warnings
- [ ] README quickstart still works if setup changed
- [ ] Follows all rules in `Rules.md`

## Found a bug in grounding?

File it with the label `grounding-bug` - these take priority over everything else.
