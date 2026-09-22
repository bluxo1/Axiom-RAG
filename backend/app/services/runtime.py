"""Runtime container: the configured providers an app instance uses.

Held on `app.state.runtime`, like config and the database. Providers are built
lazily and cached, so `create_app` never requires OpenAI or a live Chroma to be
reachable just to serve `/health`; the cost (a missing key, an unreachable
store) is paid on the first `/documents` or `/chat` call and surfaces as a
structured 503, not a startup crash.

Every provider handed out here is wrapped in the Rule 8 budget guard
(`app.core.budget`), including test-injected fakes — spend capping is a
property of the runtime, not of one provider implementation.

Tests construct a `Runtime` with explicit fakes, bypassing the builders.
"""

from __future__ import annotations

from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from app.config import AxiomConfig
from app.core.budget import BudgetGuard
from app.core.errors import AxiomError, ErrorCode
from app.db.session import Database
from app.llm.embeddings import Embedder, OpenAIEmbedder
from app.llm.guarded import GuardedEmbedder, GuardedLLM, GuardedWebSearch
from app.llm.provider import LLMProvider, OpenAILLM
from app.search.provider import BraveWebSearch, TavilyWebSearch, WebSearchProvider
from app.vector.store import ChromaVectorStore, PgVectorStore, VectorStore


def _unavailable(message: str) -> AxiomError:
    return AxiomError(
        ErrorCode.PROVIDER_UNAVAILABLE,
        message,
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
    )


class Runtime:
    """Owns the database and the configured model/vector providers."""

    def __init__(
        self,
        config: AxiomConfig,
        database: Database,
        *,
        embedder: Embedder | None = None,
        llm: LLMProvider | None = None,
        vector_store: VectorStore | None = None,
        web_search: WebSearchProvider | None = None,
    ) -> None:
        self.config = config
        self.db = database
        self.budget = BudgetGuard(config, database)
        # Injected fakes get the guard too: Rule 8 applies to every path.
        self._embedder: Embedder | None = (
            GuardedEmbedder(embedder, self.budget, model=config.embedding.model)
            if embedder is not None
            else None
        )
        self._llm: LLMProvider | None = (
            GuardedLLM(llm, self.budget, model=config.generation.model) if llm else None
        )
        self._vector_store = vector_store
        self._web_search: WebSearchProvider | None = (
            GuardedWebSearch(
                web_search,
                self.budget,
                provider=config.fallback.provider,
                price_usd=config.fallback.price_per_search_usd,
            )
            if web_search is not None
            else None
        )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._build_embedder()
        return self._embedder

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = self._build_vector_store()
        return self._vector_store

    @property
    def web_search(self) -> WebSearchProvider:
        if self._web_search is None:
            self._web_search = self._build_web_search()
        return self._web_search

    def _build_embedder(self) -> Embedder:
        section = self.config.embedding
        if section.provider == "openai":
            settings = self.config.settings
            key = settings.openai_api_key
            if key is None:
                raise _unavailable("OpenAI API key is not configured (set OPENAI_API_KEY).")
            return GuardedEmbedder(
                OpenAIEmbedder(
                    api_key=key.get_secret_value(),
                    model=section.model,
                    dimensions=section.dimensions,
                    batch_size=section.batch_size,
                    api_base=settings.openai_api_base,
                ),
                self.budget,
                model=section.model,
            )
        raise _unavailable(
            f"embedding provider '{section.provider}' has no Phase 1 implementation."
        )

    def _build_llm(self) -> LLMProvider:
        section = self.config.generation
        if section.provider == "openai":
            settings = self.config.settings
            key = settings.openai_api_key
            if key is None:
                raise _unavailable("OpenAI API key is not configured (set OPENAI_API_KEY).")
            return GuardedLLM(
                OpenAILLM(
                    api_key=key.get_secret_value(),
                    model=section.model,
                    temperature=section.temperature,
                    timeout_seconds=section.timeout_seconds,
                    api_base=settings.openai_api_base,
                ),
                self.budget,
                model=section.model,
            )
        raise _unavailable(
            f"generation provider '{section.provider}' has no Phase 1 implementation."
        )

    def _build_vector_store(self) -> VectorStore:
        if self.config.vector_store.backend == "pgvector":
            try:
                return PgVectorStore(
                    engine=self.db.engine,
                    collection_name=self.config.collection_name,
                    dimensions=self.config.embedding.dimensions,
                )
            except Exception as exc:
                raise _unavailable(f"pgvector store is unavailable: {exc}") from exc
        if self.config.vector_store.backend == "chroma":
            settings = self.config.settings
            try:
                return ChromaVectorStore(
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                    collection_name=self.config.collection_name,
                )
            except Exception as exc:  # chromadb raises broadly on connect failure
                raise _unavailable(f"vector store is unreachable: {exc}") from exc
        raise _unavailable(
            f"vector backend '{self.config.vector_store.backend}' has no Phase 1 implementation."
        )

    def _build_web_search(self) -> WebSearchProvider:
        section = self.config.fallback
        settings = self.config.settings
        if section.provider == "tavily":
            if settings.tavily_api_key is None:
                raise _unavailable("Tavily API key is not configured (set TAVILY_API_KEY).")
            inner: WebSearchProvider = TavilyWebSearch(
                api_key=settings.tavily_api_key.get_secret_value(),
                timeout_seconds=section.timeout_seconds,
            )
        else:
            if settings.brave_api_key is None:
                raise _unavailable("Brave API key is not configured (set BRAVE_API_KEY).")
            inner = BraveWebSearch(
                api_key=settings.brave_api_key.get_secret_value(),
                timeout_seconds=section.timeout_seconds,
            )
        return GuardedWebSearch(
            inner,
            self.budget,
            provider=section.provider,
            price_usd=section.price_per_search_usd,
        )
