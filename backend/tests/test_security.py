"""Security behaviors (PRD.md §8): SSRF-guarded URL ingestion, upload caps.

The SSRF guard is enforced *before* any network activity, so these tests make
no requests — refused targets are refused on validation alone.
"""

from __future__ import annotations

import socket
from socket import AddressFamily, SocketKind

import pytest
from fastapi.testclient import TestClient

from app.config import Knobs
from app.rag.parsing import (
    ParsingError,
    UploadTooLargeError,
    _fetch_safe_url,
    enforce_upload_limit,
    ensure_safe_url,
)

DOCS = "/api/v1/documents"


# ─── ensure_safe_url: refused targets ─────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/page",  # loopback IPv4
        "http://localhost/page",  # loopback by name
        "http://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
        "http://10.0.0.5/internal",  # private range
        "http://192.168.1.10/router",  # private range
        "http://172.16.0.1/admin",  # private range
        "http://[::1]/page",  # loopback IPv6
        "http://[fe80::1]/page",  # link-local IPv6
        "http://[::ffff:127.0.0.1]/",  # IPv4-mapped loopback
        "file:///etc/passwd",  # not http(s)
        "ftp://example.com/file",  # not http(s)
        "http://",  # no host
        "not a url at all",  # no scheme
    ],
)
def test_unsafe_urls_are_refused(url: str) -> None:
    with pytest.raises(ParsingError):
        ensure_safe_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/page",
        "http://[::1]/page",
        "http://169.254.169.254/",
        "http://10.0.0.5/",
        "http://192.168.1.10/",
    ],
)
def test_private_literals_are_refused_without_dns(url: str) -> None:
    """IP-literal refusals must not depend on DNS resolution at all."""

    def _exploding_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        raise AssertionError("DNS must not be consulted for IP literals")

    original = socket.getaddrinfo
    socket.getaddrinfo = _exploding_getaddrinfo  # type: ignore[assignment]
    try:
        with pytest.raises(ParsingError):
            ensure_safe_url(url)
    finally:
        socket.getaddrinfo = original


# ─── ensure_safe_url: name resolution ─────────────────────────────────────────


def test_hostname_resolving_to_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public-looking name that resolves into a private range is refused."""

    def _fake_getaddrinfo(host: str, *args: object, **kwargs: object) -> list[tuple[object, ...]]:
        return [(AddressFamily.AF_INET, SocketKind.SOCK_STREAM, 6, "", ("10.0.0.9", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    with pytest.raises(ParsingError, match="non-public address"):
        ensure_safe_url("https://internal.example.com/docs")


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com/",
        "https://example.com:8443/",
        "http://example.com:443/",
    ],
)
def test_urls_with_credentials_or_nonstandard_ports_are_refused(url: str) -> None:
    with pytest.raises(ParsingError):
        ensure_safe_url(url)


def test_redirect_to_private_address_is_refused_before_second_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_getaddrinfo(host: str, *_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        if host == "public.example":
            return [(AddressFamily.AF_INET, SocketKind.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        raise AssertionError("private redirect must be rejected before DNS lookup")

    def _fake_request(
        url: str, address: str, *, deadline: float
    ) -> tuple[int, dict[str, str], bytes]:
        assert deadline > 0
        calls.append((url, address))
        return 302, {"location": "http://127.0.0.1/admin"}, b""

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    monkeypatch.setattr("app.rag.parsing._request_pinned", _fake_request)

    with pytest.raises(ParsingError):
        _fetch_safe_url("https://public.example/start")

    assert calls == [("https://public.example/start", "93.184.216.34")]


def test_redirect_destination_is_resolved_and_pinned_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_hosts: list[str] = []
    fetched: list[tuple[str, str]] = []
    deadlines: list[float] = []

    def _fake_getaddrinfo(host: str, *_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        resolved_hosts.append(host)
        address = "93.184.216.34" if host == "first.example" else "1.1.1.1"
        return [(AddressFamily.AF_INET, SocketKind.SOCK_STREAM, 6, "", (address, 0))]

    def _fake_request(
        url: str, address: str, *, deadline: float
    ) -> tuple[int, dict[str, str], bytes]:
        deadlines.append(deadline)
        fetched.append((url, address))
        if len(fetched) == 1:
            return 302, {"location": "https://second.example/final"}, b""
        return 200, {"content-length": "4"}, b"page"

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    monkeypatch.setattr("app.rag.parsing._request_pinned", _fake_request)

    final_url, body = _fetch_safe_url("https://first.example/start")

    assert resolved_hosts == ["first.example", "second.example"]
    assert fetched == [
        ("https://first.example/start", "93.184.216.34"),
        ("https://second.example/final", "1.1.1.1"),
    ]
    assert final_url == "https://second.example/final"
    assert body == b"page"
    assert len(set(deadlines)) == 1


def test_hostname_resolving_to_public_address_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_getaddrinfo(host: str, *args: object, **kwargs: object) -> list[tuple[object, ...]]:
        return [(AddressFamily.AF_INET, SocketKind.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    assert ensure_safe_url("https://example.com/docs") == "https://example.com/docs"


def test_unresolvable_hostname_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_getaddrinfo(
        host: str, *args: object, **kwargs: object
    ) -> list[tuple[object, ...]]:
        raise socket.gaierror("name resolution failed")

    monkeypatch.setattr(socket, "getaddrinfo", _failing_getaddrinfo)

    with pytest.raises(ParsingError):
        ensure_safe_url("https://no-such-host.invalid/")


def test_public_ipv6_literal_is_accepted() -> None:
    # A global IPv6 literal: validated without DNS (literals parse directly).
    url = "http://[2606:2800:220:1:248:1893:25c8:1946]/"
    assert ensure_safe_url(url) == url


# ─── upload size cap ──────────────────────────────────────────────────────────


def test_upload_within_limit_passes() -> None:
    enforce_upload_limit(b"x" * 1024, max_upload_mb=1)


def test_upload_over_limit_is_rejected() -> None:
    with pytest.raises(UploadTooLargeError, match="limit is 1 MB"):
        enforce_upload_limit(b"x" * (1024 * 1024 + 1), max_upload_mb=1)


def test_oversized_upload_returns_413(client: TestClient) -> None:
    # Committed config caps uploads at 20 MB; exceed it by one byte.
    payload = b"x" * (20 * 1024 * 1024 + 1)
    response = client.post(DOCS, files={"file": ("big.txt", payload, "text/plain")})

    assert response.status_code == 413
    body = response.json()["error"]
    assert body["code"] == "VALIDATION_ERROR"
    assert "limit" in body["message"]


def test_cors_is_exact_and_does_not_enable_credentials(client: TestClient) -> None:
    response = client.options(
        "/api/v1/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in response.headers


# ─── config wiring ────────────────────────────────────────────────────────────


def test_ingestion_limits_are_declared_in_config(knobs: Knobs) -> None:
    assert knobs.ingestion.max_upload_mb >= 1
    assert knobs.embedding.price_per_million_usd >= 0
    assert knobs.generation.price_per_million_usd >= 0
