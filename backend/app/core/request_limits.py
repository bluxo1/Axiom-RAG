"""Reject oversized API bodies before parsing them into memory or temp files."""

from __future__ import annotations

from collections.abc import Iterable

from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.errors import ErrorCode, error_response

_BODY_METHODS = {"POST", "PUT", "PATCH"}
_DEFAULT_API_BODY_LIMIT = 1024 * 1024
_MULTIPART_OVERHEAD_LIMIT = 256 * 1024


class RequestSizeLimitMiddleware:
    """Apply strict Content-Length limits before the framework parses request bodies.

    Requests with bodies must declare their length. This rejects chunked uploads,
    which cannot be bounded before Starlette's multipart parser spools them.
    The upload route gets the configured file size plus bounded multipart overhead;
    other API bodies are capped at 1 MiB.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_prefix: str,
        upload_path: str,
        max_upload_bytes: int,
    ) -> None:
        self.app = app
        self.api_prefix = api_prefix.rstrip("/")
        self.upload_path = upload_path
        self.max_upload_body_bytes = max_upload_bytes + _MULTIPART_OVERHEAD_LIMIT

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope["type"] != "http" or not (
            path == self.api_prefix or path.startswith(f"{self.api_prefix}/")
        ):
            await self.app(scope, receive, send)
            return

        method = scope["method"].upper()
        if method not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return

        raw_lengths: Iterable[bytes] = (
            value for name, value in scope.get("headers", []) if name.lower() == b"content-length"
        )
        lengths = list(raw_lengths)
        limit = (
            self.max_upload_body_bytes
            if method == "POST" and scope["path"] == self.upload_path
            else _DEFAULT_API_BODY_LIMIT
        )
        if not lengths:
            await self._reject(
                scope,
                receive,
                send,
                status_code=411,
                message="Content-Length is required for API request bodies.",
            )
            return
        if len(lengths) != 1 or not lengths[0].isdigit() or len(lengths[0]) > 20:
            await self._reject(
                scope,
                receive,
                send,
                status_code=400,
                message="Invalid Content-Length header.",
            )
            return
        content_length = int(lengths[0])
        if content_length > limit:
            await self._reject(
                scope,
                receive,
                send,
                status_code=413,
                message=f"Request body exceeds the {limit}-byte limit.",
            )
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status_code: int,
        message: str,
    ) -> None:
        response = error_response(
            status_code=status_code,
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
        )
        await response(scope, receive, send)
