"""Relational persistence (PostgreSQL; SQLite in tests).

Design.md §2.2 defines the schema. Phase 1 creates the tables it writes:
`documents`, `chunks`, `sessions`, `messages`. The `spend_log` table and the
confidence/routing columns on `messages` arrive with the budget guard and the
router in Phases 3, alongside the code that populates them.
"""
