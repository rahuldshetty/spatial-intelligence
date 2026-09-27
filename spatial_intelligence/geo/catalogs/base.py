"""Shared catalog plumbing: HTTPS JSON fetches, STAC links, and query scoring.

Every provider module in this package builds on these helpers, so the safety
limits (HTTPS only, a 20 MiB response cap) and the keyword matching used for
natural-language searches are stated exactly once.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

from ...contracts.errors import ToolInputError

#: Hard cap on one catalog JSON document.
MAX_JSON_BYTES = 20 * 1024 * 1024
#: Words that carry no signal in a natural-language catalog query.
SEARCH_STOP_WORDS = frozenset(
    {"data", "dataset", "imagery", "image", "event", "load", "open"}
)

_USER_AGENT = "spatial-intelligence/0.1 catalog-client"


def _request(url: str, *, payload: dict | None = None) -> urllib.request.Request:
    """Build an HTTPS request for a catalog call, rejecting other schemes."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ToolInputError("catalog URLs must use HTTPS")
    headers = {"User-Agent": _USER_AGENT}
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    return urllib.request.Request(url, data=data, headers=headers)


def _read_json(request: urllib.request.Request, timeout: float) -> dict:
    """Run one request and decode a size-capped JSON object from the response."""
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_JSON_BYTES:
            raise ToolInputError(f"catalog response is too large: {length} bytes")
        raw = response.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ToolInputError("catalog response exceeded the 20 MiB safety limit")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ToolInputError("catalog response must be a JSON object")
    return value


def bbox_intersects(item_bbox: Any, bounds: list[float] | None) -> bool:
    """Whether an item's bbox overlaps the query bounds (unknown bbox: yes)."""
    if bounds is None or not isinstance(item_bbox, list) or len(item_bbox) < 4:
        return True
    try:
        west, south, east, north = (float(value) for value in item_bbox[:4])
    except (TypeError, ValueError):
        # A provider can publish a null or a string where a number belongs; that
        # item is not worth failing a whole multi-page search over.
        return True
    query_west, query_south, query_east, query_north = bounds
    return (
        west <= query_east
        and east >= query_west
        and south <= query_north
        and north >= query_south
    )


def fetch_json(url: str, *, timeout: float = 30.0) -> dict:
    """Fetch one catalog JSON object over HTTPS, capped at 20 MiB.

    The transport is resolved at call time (``urllib.request.urlopen``), so a
    test or a future transport can substitute it without touching callers.
    """
    return _read_json(_request(url), timeout)


def post_json(url: str, payload: dict, *, timeout: float = 30.0) -> dict:
    """POST a JSON body and read a JSON object back, under the same caps.

    STAC item search is a POST, so the HTTPS-only rule and the 20 MiB limit are
    shared with :func:`fetch_json` rather than restated.
    """
    return _read_json(_request(url, payload=payload), timeout)


def links(document: dict, rel: str) -> list[dict]:
    """Return the document's ``rel`` links that carry an href."""
    return [
        link
        for link in document.get("links", [])
        if isinstance(link, dict) and link.get("rel") == rel and link.get("href")
    ]


def event_id(url: str, link: dict) -> str:
    """Return a STAC link's event id: its title, or the parent directory name."""
    href = urljoin(url, str(link["href"]))
    fallback = Path(urlparse(href).path).parent.name or href
    return str(link.get("title") or fallback)


def search_terms(text: str) -> set[str]:
    """Normalize a title or query into comparable keyword terms.

    Plurals and gerunds are folded so "flooding" matches "flood", and the
    generic words in :data:`SEARCH_STOP_WORDS` are ignored.
    """
    raw_terms = set(re.findall(r"[a-z0-9]+", text.lower())) - SEARCH_STOP_WORDS
    terms = set()
    for term in raw_terms:
        if len(term) > 5 and term.endswith("ing"):
            term = term[:-3]
        elif len(term) > 4 and term.endswith("s"):
            term = term[:-1]
        terms.add(term)
    return terms


__all__ = [
    "bbox_intersects",
    "MAX_JSON_BYTES",
    "SEARCH_STOP_WORDS",
    "event_id",
    "fetch_json",
    "links",
    "post_json",
    "search_terms",
]
