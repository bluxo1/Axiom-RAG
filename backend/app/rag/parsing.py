"""Source parsing: PDF/TXT/MD/URL -> extracted text (Architecture.md §3.1).

pypdf for PDFs (per-page, so page numbers survive into citations), trafilatura
for URL main-content extraction, plain decode for TXT/MD. Heavy parsers are
imported lazily so unit tests for the dispatch and text paths do not require
them.

Uploads are bounded before parsing. URL fetches are SSRF-guarded: only HTTP(S)
on standard ports with public, pinned DNS answers is fetched, with each redirect
validated and response size/time bounded (PRD.md §8 security).

Raises `ParsingError` on an unsupported type, an oversized upload, an unsafe
URL, or when extraction yields nothing; the API layer maps these to 400/413
with friendly messages (Design.md §5).
"""

from __future__ import annotations

import http.client
import io
import ipaddress
import socket
import ssl
import time
from pathlib import PurePosixPath
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

import dns.exception
import dns.resolver

from app.rag.types import ParsedDocument, ParsedPage

# Extensions we accept as uploads (Design.md §1.1: PDF/TXT/MD).
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".text"}
_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = _TEXT_EXTENSIONS | _PDF_EXTENSIONS
_URL_FETCH_TIMEOUT_SECONDS = 10.0
_URL_FETCH_TOTAL_TIMEOUT_SECONDS = 20.0
_URL_DNS_TIMEOUT_SECONDS = 4.0
_URL_FETCH_MAX_BYTES = 5 * 1024 * 1024
_URL_FETCH_MAX_REDIRECTS = 4
_URL_MAX_LENGTH = 2048
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class ParsingError(ValueError):
    """A source could not be parsed into usable text."""


class UploadTooLargeError(ParsingError):
    """An upload exceeds `ingestion.max_upload_mb` (mapped to 413)."""


class UnsafeUrlError(ParsingError):
    """A URL is not fetchable under the SSRF policy (mapped to 400)."""


def enforce_upload_limit(data: bytes, *, max_upload_mb: int) -> None:
    """Reject an upload bigger than the configured cap before any parsing."""
    limit_bytes = max_upload_mb * 1024 * 1024
    if len(data) > limit_bytes:
        raise UploadTooLargeError(
            f"file is {len(data) / (1024 * 1024):.1f} MB; the limit is {max_upload_mb} MB."
        )


def _extension(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def _resolve_public_addresses(host: str, *, deadline: float) -> tuple[str, ...]:
    """Resolve A and AAAA records within the URL fetch deadline.

    dnspython applies a finite lifetime to each system-configured DNS lookup,
    unlike ``socket.getaddrinfo`` which has no timeout argument. Both address
    families are checked; a hostname is rejected if any answer is non-public.
    """
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise UnsafeUrlError(f"host {host!r} is not a public address")
        return (str(literal),)

    normalized_host = host.rstrip(".").casefold()
    if normalized_host == "localhost" or normalized_host.endswith(".localhost"):
        raise UnsafeUrlError("localhost names are not fetchable")

    addresses: list[str] = []
    try:
        resolver = dns.resolver.Resolver(configure=True)
        for record_type in ("A", "AAAA"):
            remaining = min(_URL_DNS_TIMEOUT_SECONDS, deadline - time.monotonic())
            if remaining <= 0:
                raise ParsingError("URL fetch timed out")
            answer = resolver.resolve(
                host,
                record_type,
                lifetime=remaining,
                search=False,
                raise_on_no_answer=False,
            )
            if answer.rrset is None:
                continue
            for record in answer:
                address = ipaddress.ip_address(record.address)
                if not address.is_global:
                    raise UnsafeUrlError("URL host resolves to a non-public address")
                normalized_address = str(address)
                if normalized_address not in addresses:
                    addresses.append(normalized_address)
    except ParsingError:
        raise
    except dns.exception.DNSException as exc:
        raise UnsafeUrlError("URL host could not be resolved within the DNS time limit") from exc
    except UnsafeUrlError:
        raise
    except ValueError as exc:
        raise UnsafeUrlError("URL host returned an invalid address") from exc

    if not addresses:
        raise UnsafeUrlError("URL host has no public addresses")
    return tuple(addresses)


def _parse_safe_url(url: str) -> tuple[SplitResult, str, int]:
    """Parse and validate URL syntax without performing DNS resolution."""
    if len(url) > _URL_MAX_LENGTH:
        raise UnsafeUrlError(f"URL exceeds the {_URL_MAX_LENGTH}-character limit")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError("invalid URL") from exc

    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("unsupported URL scheme")
    host = parsed.hostname
    if not host or parsed.username is not None or parsed.password is not None:
        raise UnsafeUrlError("URL must contain a host and must not contain credentials")
    try:
        normalized_ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            host = host.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise UnsafeUrlError("URL host is invalid") from exc
    else:
        host = str(normalized_ip)
    expected_port = 443 if parsed.scheme == "https" else 80
    if port is not None and port != expected_port:
        raise UnsafeUrlError("only the standard HTTP and HTTPS ports are allowed")
    if any(ord(character) < 32 for character in url):
        raise UnsafeUrlError("URL contains control characters")
    return parsed, host, expected_port


def ensure_safe_url(url: str) -> str:
    """Validate one fetch hop: HTTP(S), no userinfo, public host, standard port."""
    _, host, _ = _parse_safe_url(url)
    _resolve_public_addresses(host, deadline=time.monotonic() + _URL_DNS_TIMEOUT_SECONDS)
    return url


def _url_label(url: str) -> str:
    """A source name without fragments or query strings that may contain tokens."""
    parsed = urlsplit(url)
    host = parsed.hostname or "unknown host"
    port = parsed.port
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, parsed.path[:180], "", ""))


def _request_pinned(
    url: str, address: str, *, deadline: float
) -> tuple[int, dict[str, str], bytes]:
    """Fetch one URL over a socket connected to its already-validated IP."""
    parsed = urlsplit(url)
    host = parsed.hostname
    if host is None:  # _parse_safe_url validates this; retain a defensive check.
        raise UnsafeUrlError("URL has no host")
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UnsafeUrlError("URL host is invalid") from exc
    port = 443 if parsed.scheme == "https" else 80
    path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    host_header = f"[{ascii_host}]" if ":" in ascii_host else ascii_host

    remaining_time = deadline - time.monotonic()
    if remaining_time <= 0:
        raise ParsingError("URL fetch timed out")
    timeout = min(_URL_FETCH_TIMEOUT_SECONDS, remaining_time)
    connection: http.client.HTTPConnection
    if parsed.scheme == "https":
        connection = http.client.HTTPSConnection(ascii_host, port, timeout=timeout)
    else:
        connection = http.client.HTTPConnection(ascii_host, port, timeout=timeout)
    try:
        sock = socket.create_connection((address, port), timeout=timeout)
        if parsed.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=ascii_host)
        connection.sock = sock
        connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        connection.putheader("Host", host_header)
        connection.putheader("User-Agent", "Axiom-URL-Importer/1.0")
        connection.putheader("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1")
        connection.putheader("Accept-Encoding", "identity")
        connection.putheader("Connection", "close")
        connection.endheaders()
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > _URL_FETCH_MAX_BYTES:
                    raise ParsingError("URL response exceeds the 5 MB fetch limit")
            except ValueError:
                raise ParsingError("URL response has an invalid content length") from None

        chunks: list[bytes] = []
        size = 0
        while True:
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise ParsingError("URL fetch timed out")
            if connection.sock is not None:
                connection.sock.settimeout(min(_URL_FETCH_TIMEOUT_SECONDS, remaining_time))
            chunk = response.read(min(64 * 1024, _URL_FETCH_MAX_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > _URL_FETCH_MAX_BYTES:
                raise ParsingError("URL response exceeds the 5 MB fetch limit")
        return response.status, headers, b"".join(chunks)
    except (OSError, ssl.SSLError, http.client.HTTPException, TimeoutError) as exc:
        raise ParsingError("could not fetch URL within the network limits") from exc
    finally:
        connection.close()


def _fetch_safe_url(url: str) -> tuple[str, bytes]:
    """Fetch a bounded public URL, pinning each DNS result and checking redirects."""
    current_url = url
    deadline = time.monotonic() + _URL_FETCH_TOTAL_TIMEOUT_SECONDS
    for redirect_count in range(_URL_FETCH_MAX_REDIRECTS + 1):
        _, host, _ = _parse_safe_url(current_url)
        addresses = _resolve_public_addresses(host, deadline=deadline)
        last_error: ParsingError | None = None
        response: tuple[int, dict[str, str], bytes] | None = None
        for address in addresses:
            try:
                response = _request_pinned(current_url, address, deadline=deadline)
                break
            except ParsingError as exc:
                last_error = exc
        if response is None:
            raise last_error or ParsingError("could not fetch URL")

        status, headers, body = response
        if status in _REDIRECT_STATUSES:
            location = headers.get("location")
            if not location:
                raise ParsingError("URL redirect did not include a destination")
            if redirect_count == _URL_FETCH_MAX_REDIRECTS:
                raise ParsingError("URL redirected too many times")
            current_url = urljoin(current_url, location)
            continue
        if status < 200 or status >= 300:
            raise ParsingError(f"URL server returned HTTP {status}")
        return current_url, body
    raise ParsingError("URL redirected too many times")


def parse_upload(name: str, data: bytes) -> ParsedDocument:
    """Dispatch an uploaded file to the right parser by extension."""
    ext = _extension(name)
    if ext in _PDF_EXTENSIONS:
        return parse_pdf(data, name)
    if ext in _TEXT_EXTENSIONS:
        return parse_text(data, name)
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ParsingError(f"unsupported file type '{ext or name}'. Supported: {supported}")


def parse_text(data: bytes, name: str) -> ParsedDocument:
    """Decode a plain-text or Markdown upload as UTF-8 (lenient)."""
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        raise ParsingError(f"'{name}' contains no text")
    return ParsedDocument(name=name, pages=(ParsedPage(text=text, page=None),))


def parse_pdf(data: bytes, name: str) -> ParsedDocument:
    """Extract text per page with pypdf, keeping 1-based page numbers."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = tuple(
            ParsedPage(text=(page.extract_text() or "").strip(), page=number)
            for number, page in enumerate(reader.pages, start=1)
        )
    except (PdfReadError, OSError, ValueError) as exc:
        raise ParsingError(f"could not read PDF '{name}': {exc}") from exc

    non_empty = tuple(page for page in pages if page.text)
    if not non_empty:
        raise ParsingError(
            f"'{name}' has no extractable text (it may be a scanned image; OCR is out of scope)"
        )
    return ParsedDocument(name=name, pages=non_empty)


def parse_url(url: str) -> ParsedDocument:
    """Fetch a URL and extract its main content with trafilatura.

    The URL is SSRF-checked first (`ensure_safe_url`); private, loopback, and
    metadata targets are refused before any request is made.
    """
    import trafilatura

    final_url, downloaded = _fetch_safe_url(url)
    text = trafilatura.extract(
        downloaded.decode("utf-8", errors="replace"),
        include_comments=False,
        include_tables=True,
    )
    if not text or not text.strip():
        raise ParsingError("no main content extracted from URL")
    return ParsedDocument(
        name=_url_label(final_url), pages=(ParsedPage(text=text.strip(), page=None),)
    )
