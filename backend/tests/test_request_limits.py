"""Body-size limits run before FastAPI parses request bodies."""

from __future__ import annotations

import asyncio

from starlette.types import Message, Receive, Scope, Send

from app.core.request_limits import RequestSizeLimitMiddleware


def _scope(path: str, *, method: str = "POST", content_length: bytes | None) -> Scope:
    headers = [] if content_length is None else [(b"content-length", content_length)]
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }


def test_upload_body_over_limit_is_rejected_before_app_is_called() -> None:
    called = False
    messages: list[Message] = []

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal called
        called = True

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    middleware = RequestSizeLimitMiddleware(
        app, api_prefix="/api/v1", upload_path="/api/v1/documents", max_upload_bytes=10
    )
    asyncio.run(
        middleware(
            _scope("/api/v1/documents", content_length=str(10 + 256 * 1024 + 1).encode()),
            receive,
            send,
        )
    )

    assert not called
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 413


def test_json_body_over_one_mib_is_rejected_before_app_is_called() -> None:
    called = False
    messages: list[Message] = []

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal called
        called = True

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    middleware = RequestSizeLimitMiddleware(
        app, api_prefix="/api/v1", upload_path="/api/v1/documents", max_upload_bytes=10
    )
    asyncio.run(
        middleware(
            _scope("/api/v1/chat", content_length=str(1024 * 1024 + 1).encode()),
            receive,
            send,
        )
    )

    assert not called
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 413


def test_api_body_without_content_length_is_rejected() -> None:
    called = False
    messages: list[Message] = []

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal called
        called = True

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    middleware = RequestSizeLimitMiddleware(
        app, api_prefix="/api/v1", upload_path="/api/v1/documents", max_upload_bytes=10
    )
    asyncio.run(middleware(_scope("/api/v1/chat", content_length=None), receive, send))

    assert not called
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 411


def test_upload_request_allows_configured_file_size_plus_multipart_overhead() -> None:
    called = False
    messages: list[Message] = []

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        nonlocal called
        called = True
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    middleware = RequestSizeLimitMiddleware(
        app, api_prefix="/api/v1", upload_path="/api/v1/documents", max_upload_bytes=10
    )
    asyncio.run(
        middleware(
            _scope("/api/v1/documents", content_length=str(10 + 256 * 1024).encode()),
            receive,
            send,
        )
    )

    assert called
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 204
