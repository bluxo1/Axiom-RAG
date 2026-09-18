"""Logging setup.

Observability is a requirement, not a nicety (PRD.md §8): routing decisions and
errors must be traceable. Phase 0 sets the format; Phase 3 adds the Postgres
decision log.
"""

from __future__ import annotations

import logging

from app.config import LogLevel

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
_HANDLER_NAME = "axiom-console"


def configure_logging(level: LogLevel) -> None:
    """Attach one stderr handler at `level`.

    Idempotent, and deliberately additive: it never removes handlers installed
    by a host process (uvicorn, pytest), because silently dropping someone
    else's log sink is how failures go unnoticed.
    """
    root = logging.getLogger()
    root.setLevel(level)

    for existing in root.handlers:
        if existing.get_name() == _HANDLER_NAME:
            existing.setLevel(level)
            return

    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
