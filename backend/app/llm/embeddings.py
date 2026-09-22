"""Embedding providers (ADR-0001).

The active model is declared in `config.yaml` and must not drift: switching it
requires full re-ingestion, which is why the vector-store collection is
namespaced by model slug (`config.collection_name`).

`Embedder` is the interface the pipeline depends on. `OpenAIEmbedder` is the
real provider; `HashingEmbedder` is a deterministic, dependency-free fake for
tests — same text always yields the same vector, and shared tokens raise cosine
similarity, so retrieval tests are meaningful without a live key.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. All vectors from one embedder share a dimension."""

    @property
    def dimensions(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder for tests.

    Not for production: no semantics beyond token overlap. It exists so the
    pipeline can be exercised end-to-end with no network and no API key.
    """

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha1(token.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimensions
            vector[bucket] += 1.0
        return _l2_normalize(vector)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedder:
    """OpenAI embeddings via LlamaIndex (Architecture.md §5)."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        batch_size: int,
        api_base: str | None = None,
    ) -> None:
        from llama_index.embeddings.openai import OpenAIEmbedding

        self._dimensions = dimensions
        self._client = OpenAIEmbedding(
            model=model,
            api_key=api_key,
            api_base=api_base,
            dimensions=dimensions,
            embed_batch_size=batch_size,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return cast("list[list[float]]", self._client.get_text_embedding_batch(list(texts)))

    def embed_query(self, text: str) -> list[float]:
        return cast("list[float]", self._client.get_query_embedding(text))
