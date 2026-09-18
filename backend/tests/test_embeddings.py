"""HashingEmbedder: determinism and token-overlap similarity (test double)."""

from __future__ import annotations

import math

import pytest

from app.llm.embeddings import HashingEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def test_same_text_embeds_identically() -> None:
    embedder = HashingEmbedder()
    assert embedder.embed_query("hello world") == embedder.embed_query("hello world")


def test_shared_tokens_raise_similarity() -> None:
    embedder = HashingEmbedder()
    base = embedder.embed_query("insurance policy coverage")
    overlapping = embedder.embed_query("policy coverage details")
    disjoint = embedder.embed_query("banana helicopter mountain")

    assert _cosine(base, overlapping) > _cosine(base, disjoint)


def test_vectors_have_the_declared_dimension_and_are_normalized() -> None:
    embedder = HashingEmbedder(dimensions=32)
    vector = embedder.embed_query("some words here")

    assert embedder.dimensions == 32
    assert len(vector) == 32
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)


def test_batch_matches_single() -> None:
    embedder = HashingEmbedder()
    texts = ["alpha beta", "gamma"]
    batch = embedder.embed_texts(texts)

    assert batch == [embedder.embed_query(text) for text in texts]


def test_zero_dimensions_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions must be positive"):
        HashingEmbedder(0)
