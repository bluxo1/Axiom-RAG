"""Structured errors.

Prompt.md §API CONTRACT: *structured errors only — never a silent 500.* Every
error leaving the API uses one envelope, so a client can branch on
`error.code` instead of parsing prose:

```json
{"error": {"code": "RATE_LIMITED", "message": "...", "details": {...}}}
```
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request
from starlette.responses import Response
from starlette.status import (
    HTTP_404_NOT_FOUND,
    HTTP_405_METHOD_NOT_ALLOWED,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    """Stable, machine-readable error codes.

    Codes are added as the phase that raises them lands, so every member here
    is reachable. `BUDGET_EXCEEDED` arrives with the Rule 8 budget guard, which
    is active from Phase 1 because that is when LLM/embedding spend begins.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    RATE_LIMITED = "RATE_LIMITED"
    # A configured spend cap would be crossed (Rule 8): 429, retry later when
    # the daily/monthly window resets. Distinct from RATE_LIMITED (request rate).
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    # A required provider (LLM/embeddings/vector store) is not configured or is
    # unreachable — a 503, distinct from a bug (500). Raised from Phase 1's
    # ingestion and chat paths.
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """The only error shape the API emits."""

    error: ErrorBody


class AxiomError(Exception):
    """An error with a client-facing code and status.

    Anything raised inside a request that is *not* an `AxiomError` still gets
    the same envelope via the catch-all handler — it just gets logged as a bug.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status_code: int = HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def error_response(
    *,
    status_code: int,
    code: ErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build the error envelope.

    Middleware uses this directly: exceptions raised inside middleware bypass
    the app's exception handlers, so the response is constructed there instead.
    """
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


_STATUS_TO_CODE: dict[int, ErrorCode] = {
    HTTP_404_NOT_FOUND: ErrorCode.NOT_FOUND,
    HTTP_405_METHOD_NOT_ALLOWED: ErrorCode.METHOD_NOT_ALLOWED,
    HTTP_422_UNPROCESSABLE_CONTENT: ErrorCode.VALIDATION_ERROR,
}


async def _handle_axiom_error(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, AxiomError):  # pragma: no cover - registered per type
        raise exc
    logger.warning(
        "axiom_error code=%s status=%s path=%s", exc.code, exc.status_code, request.url.path
    )
    return error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def _handle_validation_error(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        raise exc
    logger.info("validation_error path=%s", request.url.path)
    return error_response(
        status_code=HTTP_422_UNPROCESSABLE_CONTENT,
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed.",
        details={"errors": exc.errors()},
    )


async def _handle_http_exception(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover
        raise exc
    code = _STATUS_TO_CODE.get(exc.status_code)
    if code is None:
        code = (
            ErrorCode.INTERNAL_ERROR
            if exc.status_code >= HTTP_500_INTERNAL_SERVER_ERROR
            else ErrorCode.VALIDATION_ERROR
        )
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    logger.info("http_error status=%s path=%s", exc.status_code, request.url.path)
    return error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
        headers=exc.headers,
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> Response:
    """Catch-all so an unhandled exception never becomes a bare 500 body.

    The traceback is logged (Rules.md §6 bans swallowing errors); the client
    gets the envelope without internals.
    """
    logger.exception("unhandled_error path=%s", request.url.path, exc_info=exc)
    return error_response(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        code=ErrorCode.INTERNAL_ERROR,
        message="Internal server error.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every error path to the structured envelope."""
    app.add_exception_handler(AxiomError, _handle_axiom_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected_error)
