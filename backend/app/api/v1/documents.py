"""Document endpoints (Design.md §1.1)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_413_CONTENT_TOO_LARGE,
)

from app.api.deps import RuntimeDep
from app.api.v1.schemas import DocumentList, DocumentSummary, UrlIngestRequest
from app.core.errors import AxiomError, ErrorCode
from app.rag.parsing import ParsingError, UploadTooLargeError
from app.services.documents import delete_document, list_documents
from app.services.ingestion import IngestResult, ingest_from_url, ingest_upload

router = APIRouter(prefix="/documents", tags=["documents"])


def _summary(result: IngestResult) -> DocumentSummary:
    return DocumentSummary(
        doc_id=result.doc_id, name=result.name, chunks=result.chunks, status=result.status
    )


def _bad_request(exc: ParsingError) -> AxiomError:
    return AxiomError(ErrorCode.VALIDATION_ERROR, str(exc), status_code=HTTP_400_BAD_REQUEST)


def _too_large(exc: UploadTooLargeError) -> AxiomError:
    return AxiomError(ErrorCode.VALIDATION_ERROR, str(exc), status_code=HTTP_413_CONTENT_TOO_LARGE)


@router.post("", response_model=DocumentSummary, summary="Upload a document (PDF/TXT/MD)")
def upload_document(runtime: RuntimeDep, file: Annotated[UploadFile, File()]) -> DocumentSummary:
    limit = runtime.config.ingestion.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    try:
        result = ingest_upload(runtime, name=file.filename or "upload", data=data)
    except UploadTooLargeError as exc:
        raise _too_large(exc) from exc
    except ParsingError as exc:
        raise _bad_request(exc) from exc
    return _summary(result)


@router.post("/url", response_model=DocumentSummary, summary="Ingest a document from a URL")
def ingest_url(runtime: RuntimeDep, body: UrlIngestRequest) -> DocumentSummary:
    try:
        result = ingest_from_url(runtime, url=body.url)
    except ParsingError as exc:
        raise _bad_request(exc) from exc
    return _summary(result)


@router.get("", response_model=DocumentList, summary="List ingested documents")
def list_ingested(runtime: RuntimeDep) -> DocumentList:
    return DocumentList(documents=[_summary(result) for result in list_documents(runtime)])


@router.delete("/{doc_id}", status_code=204, summary="Delete a document and its chunks")
def delete(runtime: RuntimeDep, doc_id: str) -> None:
    if not delete_document(runtime, doc_id):
        raise AxiomError(
            ErrorCode.NOT_FOUND, f"document '{doc_id}' not found.", status_code=HTTP_404_NOT_FOUND
        )
