"""Typed configuration: `config.yaml` knobs plus `.env` secrets.

Split of responsibility (Rules.md §2, §5):

* **`config.yaml`** — every tunable knob: thresholds, confidence weights, k,
  chunk size, budget caps. Committed, reviewable, never magic in code.
* **`.env`** — secrets and deployment-specific endpoints only. Documented in
  `.env.example`.

Budget caps appear in both on purpose (Prompt.md Rule 8, Design.md §6): YAML
holds the defaults, environment variables override them so an operator can
tighten a cap without editing committed config.

Both sources are validated on load with `extra="forbid"`, so a typo in a key
fails at startup instead of silently falling back to a default.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "test", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class ConfigError(RuntimeError):
    """Configuration is missing, unreadable, or invalid.

    Raised at startup rather than tolerated: a half-configured grounding
    pipeline is worse than a process that refuses to boot.
    """


class StrictModel(BaseModel):
    """Immutable model that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ─── config.yaml sections ─────────────────────────────────────────────────────


class AppSection(StrictModel):
    """Service identity and HTTP surface (Design.md §1)."""

    name: str = Field(min_length=1)
    api_prefix: str
    cors_origins: tuple[str, ...]

    @field_validator("api_prefix")
    @classmethod
    def _check_prefix(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("api_prefix must start with '/'")
        if len(value) > 1 and value.endswith("/"):
            raise ValueError("api_prefix must not end with '/'")
        return value


class ChunkingSection(StrictModel):
    """Recursive splitting parameters (Architecture.md §3.1)."""

    size_tokens: PositiveInt
    overlap_tokens: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _overlap_fits(self) -> Self:
        if self.overlap_tokens >= self.size_tokens:
            raise ValueError("chunking.overlap_tokens must be smaller than size_tokens")
        return self


class EmbeddingSection(StrictModel):
    """Active embedding model (ADR-0001).

    Declared once, here. Switching it requires full re-ingestion, so the
    collection name is namespaced by model to keep incompatible vectors apart.
    `price_per_million_usd` feeds the Rule 8 spend estimate (Design.md §6):
    a conservative list-price number used to trip caps early, not invoicing.
    """

    provider: Literal["openai", "sentence-transformers"]
    model: str = Field(min_length=1)
    dimensions: PositiveInt
    batch_size: PositiveInt
    price_per_million_usd: NonNegativeFloat

    @property
    def slug(self) -> str:
        """Filesystem/collection-safe form of the model name.

        `text-embedding-3-small` -> `text_embedding_3_small`. Used to namespace
        vector-store collections per ADR-0001 §Decision 2.
        """
        return re.sub(r"[^a-z0-9]+", "_", self.model.lower()).strip("_")


class VectorStoreSection(StrictModel):
    """Vector store backend (ADR-0002).

    Backend is a config change, not a code change. Cosine is pinned so dev
    (Chroma) and prod (pgvector) produce comparable scores.
    """

    backend: Literal["chroma", "pgvector"]
    similarity: Literal["cosine", "l2", "ip"]
    collection_prefix: str = Field(min_length=1)


class IngestionSection(StrictModel):
    """Ingestion limits (PRD.md §8 security: bounded, rate-limited inputs).

    Uploads are streamed and rejected with 413 before any parsing or
    embedding work once they exceed `max_upload_mb`.
    """

    max_upload_mb: PositiveInt


class RetrievalSection(StrictModel):
    """Top-k semantic search (Architecture.md §3.2)."""

    top_k: PositiveInt


class GenerationSection(StrictModel):
    """LLM generation and its failure budgets (Design.md §5)."""

    provider: Literal["openai", "groq"]
    model: str = Field(min_length=1)
    temperature: Annotated[float, Field(ge=0.0, le=2.0)]
    timeout_seconds: PositiveFloat
    max_repair_attempts: Annotated[int, Field(ge=0)]
    max_timeout_retries: Annotated[int, Field(ge=0)]
    # Rule 8 spend accounting: per-1M-token estimate (blended input + output
    # list price, deliberately the larger of the two) used by the budget guard.
    price_per_million_usd: NonNegativeFloat


class GroundingSection(StrictModel):
    """Citation verification, the hard gate (Architecture.md §3.4)."""

    max_regenerations: Annotated[int, Field(ge=0)]
    support_keyword_overlap_min: UnitFloat


class ConfidenceWeights(StrictModel):
    """Hybrid confidence weights (Architecture.md §3.5). Must sum to 1.0."""

    retrieval: UnitFloat
    faithfulness: UnitFloat
    coverage: UnitFloat

    @model_validator(mode="after")
    def _sums_to_one(self) -> Self:
        total = self.retrieval + self.faithfulness + self.coverage
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"confidence.weights must sum to 1.0, got {total:.6f}")
        return self


class ConfidenceThresholds(StrictModel):
    """Router thresholds (Architecture.md §3.5).

    `>= high` answers, `[low, high)` answers with a visible low-confidence
    badge, `< low` falls back to web search.
    """

    high: UnitFloat
    low: UnitFloat

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low >= self.high:
            raise ValueError("confidence.thresholds.low must be below high")
        return self


class ConfidenceSection(StrictModel):
    weights: ConfidenceWeights
    thresholds: ConfidenceThresholds


class FallbackSection(StrictModel):
    """Live web-search fallback (Architecture.md §3.6)."""

    enabled: bool
    provider: Literal["tavily", "brave"]
    max_results: PositiveInt
    timeout_seconds: PositiveFloat
    # Rule 8 spend accounting: web search is billed per call, not per token, so
    # the guard records this flat estimate against the monthly USD cap.
    price_per_search_usd: NonNegativeFloat


class BudgetSection(StrictModel):
    """Spend caps (Prompt.md Rule 8, Design.md §6).

    Defaults live in `config.yaml`; `MAX_REQUEST_TOKENS`, `DAILY_TOKEN_CAP`, and
    `MONTHLY_SPEND_USD` override them. Exceeding a cap returns
    429 `BUDGET_EXCEEDED`, never a silent surprise bill.
    """

    max_request_tokens: PositiveInt
    daily_token_cap: PositiveInt
    monthly_spend_usd: PositiveFloat

    def with_env_overrides(self, settings: Settings) -> BudgetSection:
        """Return a copy with any environment-provided cap applied."""
        overrides: dict[str, int | float] = {}
        if settings.max_request_tokens is not None:
            overrides["max_request_tokens"] = settings.max_request_tokens
        if settings.daily_token_cap is not None:
            overrides["daily_token_cap"] = settings.daily_token_cap
        if settings.monthly_spend_usd is not None:
            overrides["monthly_spend_usd"] = settings.monthly_spend_usd
        if not overrides:
            return self
        # Re-validate rather than model_copy: an env var must clear the same
        # bar as a committed value (e.g. DAILY_TOKEN_CAP=0 is a misconfiguration).
        return BudgetSection.model_validate({**self.model_dump(), **overrides})


class RateLimitSection(StrictModel):
    """Per-session token bucket (PRD.md §8 security, Design.md §6).

    Dev is generous; the Phase 4 hardening sweep tightens prod.
    """

    enabled: bool
    capacity: PositiveFloat
    refill_per_second: PositiveFloat
    exempt_paths: tuple[str, ...]
    max_tracked_identities: PositiveInt


class CacheSection(StrictModel):
    """Answer cache keyed by (question_hash, doc_set_hash) (Design.md §6)."""

    enabled: bool
    ttl_seconds: PositiveInt
    max_entries: PositiveInt


class Knobs(StrictModel):
    """The whole of `config.yaml`, validated."""

    app: AppSection
    chunking: ChunkingSection
    ingestion: IngestionSection
    embedding: EmbeddingSection
    vector_store: VectorStoreSection
    retrieval: RetrievalSection
    generation: GenerationSection
    grounding: GroundingSection
    confidence: ConfidenceSection
    fallback: FallbackSection
    budget: BudgetSection
    rate_limit: RateLimitSection
    cache: CacheSection


# ─── .env settings ────────────────────────────────────────────────────────────


def _blank_to_none(value: Any) -> Any:  # noqa: ANN401 - pre-validation hook
    """Treat an empty or whitespace-only env var as unset.

    `.env.example` ships keys with empty values so every variable is visible;
    an empty value must mean "not configured", not "configured as empty".
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


class Settings(BaseSettings):
    """Environment-provided secrets, endpoints, and budget overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # `.env` also carries POSTGRES_* and VITE_* for compose and the browser;
        # those are not backend settings.
        extra="ignore",
    )

    axiom_env: Environment = "dev"
    log_level: LogLevel = "INFO"
    config_path: Path = Path("config.yaml")

    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None

    database_url: str = "postgresql+psycopg://axiom:axiom@localhost:5432/axiom"
    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8001, gt=0, le=65535)

    # Emit HSTS (Strict-Transport-Security). Meaningful only behind TLS, so it
    # defaults off for plain-HTTP dev and the deploy turns it on (see .env.example).
    enable_hsts: bool = False

    # Unset (or blank) means "use the config.yaml default".
    max_request_tokens: int | None = None
    daily_token_cap: int | None = None
    monthly_spend_usd: float | None = None

    _blank_is_unset = field_validator(
        "openai_api_key",
        "groq_api_key",
        "tavily_api_key",
        "brave_api_key",
        "max_request_tokens",
        "daily_token_cap",
        "monthly_spend_usd",
        mode="before",
    )(_blank_to_none)


# ─── Composition ──────────────────────────────────────────────────────────────


class AxiomConfig:
    """Effective configuration: YAML knobs with environment overrides applied.

    Read this instead of reaching for `os.environ` or re-parsing YAML.
    """

    def __init__(self, settings: Settings, knobs: Knobs) -> None:
        self.settings = settings
        self.app = knobs.app
        self.chunking = knobs.chunking
        self.ingestion = knobs.ingestion
        self.embedding = knobs.embedding
        self.vector_store = knobs.vector_store
        self.retrieval = knobs.retrieval
        self.generation = knobs.generation
        self.grounding = knobs.grounding
        self.confidence = knobs.confidence
        self.fallback = knobs.fallback
        self.budget = knobs.budget.with_env_overrides(settings)
        self.rate_limit = knobs.rate_limit
        self.cache = knobs.cache

    @property
    def collection_name(self) -> str:
        """Vector-store collection for the active embedding model (ADR-0001)."""
        return f"{self.vector_store.collection_prefix}_{self.embedding.slug}"


def _repo_root() -> Path:
    """Repository root, derived from this file's location.

    `backend/app/config.py` -> `backend/app` -> `backend` -> repo root.
    """
    return Path(__file__).resolve().parents[2]


def resolve_config_path(config_path: Path) -> Path:
    """Locate `config.yaml`.

    An absolute path is used as given. A relative path is tried against the
    working directory first, then the repository root, so the same value works
    whether a process starts in `backend/` or at the root. Containers set an
    absolute `CONFIG_PATH` and skip the search entirely.
    """
    if config_path.is_absolute():
        if not config_path.is_file():
            raise ConfigError(f"config file not found: {config_path}")
        return config_path

    candidates = [Path.cwd() / config_path, _repo_root() / config_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise ConfigError(f"config file not found: tried {tried}")


def load_knobs(path: Path) -> Knobs:
    """Parse and validate a `config.yaml`, raising `ConfigError` on any problem."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    try:
        return Knobs.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {path}:\n{exc}") from exc


def build_config() -> AxiomConfig:
    """Load settings and knobs from disk. Uncached; prefer `get_config`."""
    settings = Settings()
    return AxiomConfig(settings, load_knobs(resolve_config_path(settings.config_path)))


@lru_cache(maxsize=1)
def get_config() -> AxiomConfig:
    """Process-wide configuration, loaded once.

    Usable directly as a FastAPI dependency. Tests call
    `get_config.cache_clear()` to force a reload.
    """
    return build_config()
