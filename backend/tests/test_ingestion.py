"""Ingestion pipeline and document endpoints (Phases.md Phase 1, Design.md §1.1)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from app.config import AxiomConfig
from app.db.models import Document
from app.db.session import Database
from app.llm.provider import ScriptedLLM
from app.services.ingestion import ingest_upload
from app.services.runtime import Runtime
from app.vector.store import InMemoryVectorStore
from tests.helpers import TEST_DATABASE_URL

DOCS = "/api/v1/documents"


def _upload(client: TestClient, name: str, body: bytes) -> Response:
    return client.post(DOCS, files={"file": (name, body, "text/plain")})


def test_upload_ingests_and_reports_ready(client: TestClient) -> None:
    response = _upload(client, "notes.md", b"Axiom starts from what you can prove. " * 20)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["chunks"] >= 1
    assert payload["name"] == "notes.md"


def test_uploaded_document_is_listed_then_deleted(client: TestClient, runtime: Runtime) -> None:
    uploaded = _upload(client, "notes.txt", b"grounded retrieval augmented generation").json()
    doc_id = uploaded["doc_id"]

    listed = client.get(DOCS).json()["documents"]
    assert [d["doc_id"] for d in listed] == [doc_id]

    assert client.delete(f"{DOCS}/{doc_id}").status_code == 204
    assert client.get(DOCS).json()["documents"] == []
    # Vectors are gone too, not just the row.
    assert runtime.vector_store.query([1.0] * 64, k=10) == []


def test_delete_unknown_document_is_404(client: TestClient) -> None:
    response = client.delete(f"{DOCS}/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_unsupported_type_is_a_friendly_400(client: TestClient) -> None:
    response = client.post(DOCS, files={"file": ("data.zip", b"blob", "application/zip")})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_empty_document_is_rejected(client: TestClient) -> None:
    assert _upload(client, "empty.txt", b"   ").status_code == 400


# ─── Failure marks the document 'failed' (service level) ──────────────────────


class _FailingEmbedder:
    dimensions = 64

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedding backend down")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding backend down")


def test_ingestion_failure_marks_document_failed(config: AxiomConfig) -> None:
    database = Database(TEST_DATABASE_URL)
    database.create_all()
    runtime = Runtime(
        config,
        database,
        embedder=_FailingEmbedder(),
        llm=ScriptedLLM(),
        vector_store=InMemoryVectorStore(),
    )
    try:
        with pytest.raises(RuntimeError, match="embedding backend down"):
            ingest_upload(runtime, name="notes.txt", data=b"some ingestible text here")

        with database.session() as session:
            documents = session.query(Document).all()
            assert len(documents) == 1
            assert documents[0].status == "failed"
    finally:
        database.dispose()
