"""Model providers behind narrow interfaces (Rules.md §2).

Every LLM and embedding call goes through a Protocol here, so the pipeline never
imports a vendor SDK directly and tests inject deterministic fakes instead of
making live calls (CLAUDE.md: recorded fixtures, never live calls in CI). The
active provider is chosen by `config.yaml` (`embedding.provider`,
`generation.provider`).
"""
