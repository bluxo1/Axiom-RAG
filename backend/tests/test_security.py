"""Security behaviors (PRD.md §8): SSRF-guarded URL ingestion, upload caps.

The SSRF guard is enforced *before* any network activity, so these tests make
no requests — refused targets are refused on validation alone.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import dns.resolver
import pytest
from fastapi.testclient import TestClient

from app.config import Knobs
from app.rag.parsing import (
    ParsingError,
    UploadTooLargeError,
    _fetch_safe_url,
    _resolve_public_addresses,
    enforce_upload_limit,
    ensure_safe_url,
)

DOCS = "/api/v1/documents"


class _FakeDnsAnswer:
    def __init__(self, addresses: tuple[str, ...]) -> None:
        self._records = tuple(SimpleNamespace(address=address) for address in addresses)
        self.rrset = self._records or None

    def __iter__(self) -> Iterator[SimpleNamespace]:
        return iter(self._records)


def _install_dns(
    monkeypatch: pytest.MonkeyPatch,
    addresses: dict[tuple[str, str], tuple[str, ...]],
    seen_hosts: list[str] | None = None,
) -> None:
    def resolve(
        _resolver: dns.resolver.Resolver,
        host: str,
        record_type: str,
        **kwargs: object,
    ) -> _FakeDnsAnswer:
        lifetime = kwargs.get("lifetime")
        assert isinstance(lifetime, float)
        assert 0 < lifetime <= 4.0
        assert kwargs.get("search") is False
        if seen_hosts is not None:
            seen_hosts.append(host)
        return _FakeDnsAnswer(addresses.get((host, record_type), ()))

    monkeypatch.setattr(dns.resolver.Resolver, "resolve", resolve)


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
def test_private_literals_are_refused_without_dns(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IP-literal refusals must not depend on DNS resolution at all."""

    def _exploding_resolve(*_args: object, **_kwargs: object) -> _FakeDnsAnswer:
        raise AssertionError("DNS must not be consulted for IP literals")

    monkeypatch.setattr(dns.resolver.Resolver, "resolve", _exploding_resolve)
    with pytest.raises(ParsingError):
        ensure_safe_url(url)


# ─── ensure_safe_url: name resolution ─────────────────────────────────────────


def test_hostname_resolving_to_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public-looking name that resolves into a private range is refused."""
    _install_dns(monkeypatch, {("internal.example.com", "A"): ("10.0.0.9",)})

    with pytest.raises(ParsingError, match="non-public address"):
        ensure_safe_url("https://internal.example.com/docs")


def test_private_ipv6_answer_is_refused_when_ipv4_answer_is_public(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_dns(
        monkeypatch,
        {
            ("mixed.example.com", "A"): ("93.184.216.34",),
            ("mixed.example.com", "AAAA"): ("fd00::1",),
        },
    )

    with pytest.raises(ParsingError, match="non-public address"):
        ensure_safe_url("https://mixed.example.com/")


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


def test_overlong_url_is_refused_before_dns() -> None:
    with pytest.raises(ParsingError, match="2048-character limit"):
        ensure_safe_url("https://example.com/" + "a" * 2048)


def test_redirect_to_private_address_is_refused_before_second_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    _install_dns(monkeypatch, {("public.example", "A"): ("93.184.216.34",)})

    def _fake_request(
        url: str, address: str, *, deadline: float
    ) -> tuple[int, dict[str, str], bytes]:
        assert deadline > 0
        calls.append((url, address))
        return 302, {"location": "http://127.0.0.1/admin"}, b""

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

    _install_dns(
        monkeypatch,
        {
            ("first.example", "A"): ("93.184.216.34",),
            ("second.example", "A"): ("1.1.1.1",),
        },
        resolved_hosts,
    )

    def _fake_request(
        url: str, address: str, *, deadline: float
    ) -> tuple[int, dict[str, str], bytes]:
        deadlines.append(deadline)
        fetched.append((url, address))
        if len(fetched) == 1:
            return 302, {"location": "https://second.example/final"}, b""
        return 200, {"content-length": "4"}, b"page"

    monkeypatch.setattr("app.rag.parsing._request_pinned", _fake_request)

    final_url, body = _fetch_safe_url("https://first.example/start")

    assert resolved_hosts == [
        "first.example",
        "first.example",
        "second.example",
        "second.example",
    ]
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
    _install_dns(monkeypatch, {("example.com", "A"): ("93.184.216.34",)})

    assert ensure_safe_url("https://example.com/docs") == "https://example.com/docs"


def test_unresolvable_hostname_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_dns(monkeypatch, {})

    with pytest.raises(ParsingError):
        ensure_safe_url("https://no-such-host.invalid/")


def test_expired_fetch_deadline_stops_before_dns_lookup() -> None:
    with pytest.raises(ParsingError, match="URL fetch timed out"):
        _resolve_public_addresses("example.com", deadline=0.0)


def test_unicode_hostname_is_normalized_before_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    ascii_host = "bücher.example".encode("idna").decode("ascii")
    _install_dns(monkeypatch, {(ascii_host, "A"): ("93.184.216.34",)})

    assert ensure_safe_url("https://bücher.example/") == "https://bücher.example/"


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
