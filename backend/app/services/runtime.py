"""Runtime container: the configured providers an app instance uses.

Held on `app.state.runtime`, like config and the database. Providers are built
lazily and cached, so `create_app` never requires OpenAI or a live Chroma to be
reachable just to serve `/health`; the cost (a missing key, an unreachable
store) is paid on the first `/documents` or `/chat` call and surfaces as a
structured 503, not a startup crash.

Tests construct a `Runtime` with explicit fakes, bypassing the builders.
"""

from __future__ import annotations

from starlette.status import HTTP_503_SERVICE_UNAVAILABLE

from app.config import AxiomConfig
from app.core.errors import AxiomError, ErrorCode
from app.db.session import Database
from app.llm.embeddings import Embedder, OpenAIEmbedder
from app.llm.provider import LLMProvider, OpenAILLM
from app.vector.store import ChromaVectorStore, VectorStore


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
    ) -> None:
        self.config = config
        self.db = database
        self._embedder = embedder
        self._llm = llm
        self._vector_store = vector_store

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

    def _build_embedder(self) -> Embedder:
        section = self.config.embedding
        if section.provider == "openai":
            key = self.config.settings.openai_api_key
            if key is None:
                raise _unavailable("OpenAI API key is not configured (set OPENAI_API_KEY).")
            return OpenAIEmbedder(
                api_key=key.get_secret_value(),
                model=section.model,
                dimensions=section.dimensions,
                batch_size=section.batch_size,
            )
        raise _unavailable(
            f"embedding provider '{section.provider}' has no Phase 1 implementation."
        )

    def _build_llm(self) -> LLMProvider:
        section = self.config.generation
        if section.provider == "openai":
            key = self.config.settings.openai_api_key
            if key is None:
                raise _unavailable("OpenAI API key is not configured (set OPENAI_API_KEY).")
            return OpenAILLM(
                api_key=key.get_secret_value(),
                model=section.model,
                temperature=section.temperature,
                timeout_seconds=section.timeout_seconds,
            )
        raise _unavailable(
            f"generation provider '{section.provider}' has no Phase 1 implementation."
        )

    def _build_vector_store(self) -> VectorStore:
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
