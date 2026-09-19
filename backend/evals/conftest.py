"""Eval fixtures and the `--run-llm` switch (EVAL.md §3).

`pytest evals/` runs offline against recorded fixtures (the CI default).
`pytest evals/ --run-llm` opts into live LLM calls for RAGAS scoring, which
needs `OPENAI_API_KEY` and spends tokens — never run in CI.
"""

from __future__ import annotations

import pytest

from app.config import AxiomConfig, Knobs, Settings, load_knobs
from evals.dataset import GoldenEntry, load_golden

_REPO_ROOT_CONFIG = "config.yaml"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run live-LLM evals (RAGAS). Needs OPENAI_API_KEY and spends tokens.",
    )


@pytest.fixture(scope="session")
def run_llm(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--run-llm"))


@pytest.fixture(scope="session")
def knobs() -> Knobs:
    from tests.helpers import CONFIG_PATH

    return load_knobs(CONFIG_PATH)


@pytest.fixture(scope="session")
def config(knobs: Knobs) -> AxiomConfig:
    return AxiomConfig(Settings(), knobs)


@pytest.fixture(scope="session")
def golden() -> list[GoldenEntry]:
    return load_golden()
