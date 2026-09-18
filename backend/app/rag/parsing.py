"""Source parsing: PDF/TXT/MD/URL -> extracted text (Architecture.md §3.1).

pypdf for PDFs (per-page, so page numbers survive into citations), trafilatura
for URL main-content extraction, plain decode for TXT/MD. Heavy parsers are
imported lazily so unit tests for the dispatch and text paths do not require
them.

Uploads are size-capped (`ingestion.max_upload_mb`) and URL fetches are
SSRF-guarded: only http(s) with a public, resolved host is fetched, so
`POST /documents/url` cannot be pointed at cloud metadata endpoints or the
internal network (PRD.md §8 security).

Raises `ParsingError` on an unsupported type, an oversized upload, an unsafe
URL, or when extraction yields nothing; the API layer maps these to 400/413
with friendly messages (Design.md §5).
"""

from __future__ import annotations

import io
import ipaddress
import socket
from pathlib import PurePosixPath
from urllib.parse import urlparse

from app.rag.types import ParsedDocument, ParsedPage

# Extensions we accept as uploads (Design.md §1.1: PDF/TXT/MD).
_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".text"}
_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = _TEXT_EXTENSIONS | _PDF_EXTENSIONS


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


def _is_publicly_routable(host: str) -> bool:
    """True when every resolved address for `host` is public and global.

    Resolution happens here — not inside the fetcher — so the check and the
    fetch can't be separated by a later refactor. DNS rebinding (resolving
    public, then re-resolving private at fetch time) is out of scope for a
    same-process check; the strict resolution here still blocks the direct
    forms (metadata IPs, localhost, LAN names, odd schemes).
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_loopback or address.is_link_local:
            return False
    return True


def ensure_safe_url(url: str) -> str:
    """Validate a fetchable URL: https/http only, host public and resolved.

    Returns the validated URL; raises `UnsafeUrlError` otherwise. Blocks the
    SSRF classics: cloud metadata endpoints (169.254.169.254), loopback,
    private ranges, and non-http schemes.
    """
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise UnsafeUrlError(f"invalid URL: {url!r}") from exc

    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError(f"unsupported URL scheme {parsed.scheme or '(none)'!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError(f"URL has no host: {url!r}")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None  # a hostname, not an IP literal — resolved below
    if literal is not None and (
        not literal.is_global or literal.is_loopback or literal.is_link_local
    ):
        raise UnsafeUrlError(f"host {host!r} is not a public address")

    if not _is_publicly_routable(host):
        raise UnsafeUrlError(f"host {host!r} does not resolve to a public address")
    return url


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

    url = ensure_safe_url(url)
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise ParsingError(f"could not fetch '{url}'")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
    if not text or not text.strip():
        raise ParsingError(f"no main content extracted from '{url}'")
    return ParsedDocument(name=url, pages=(ParsedPage(text=text.strip(), page=None),))
