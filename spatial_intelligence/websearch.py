"""Web search: the local implementation behind the agent's ``web_search`` tool.

The agent's ``WebSearch`` capability (see ``agent/builder.py``) prefers the
provider's native web search and falls back to :func:`web_search` on the models
that have none — which is every model reachable through the custom
OpenAI-compatible endpoint this harness ships with (DeepSeek, llama.cpp, vLLM).

The fallback is a plain HTTPS POST to DuckDuckGo's HTML endpoint: no API key, no
extra dependency, and the same shape as the catalog clients (HTTPS only, a size
cap, :class:`ToolInputError` for a rejected request). The HTML is parsed with
``html.parser`` rather than a regex because titles and snippets carry nested
``<b>`` tags and HTML entities that a text scan would mangle.

The function is the tool: pydantic-ai wraps a bare callable as a ``Tool``, so its
name is the model-facing tool name and its docstring is the model-facing
description. Do not rename it without renaming it in ``tools/build.py``.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from html.parser import HTMLParser

from .contracts.errors import ToolInputError

#: DuckDuckGo's server-rendered HTML endpoint (no key, no JavaScript).
SEARCH_URL = "https://html.duckduckgo.com/html/"
#: Hard cap on one search response.
MAX_SEARCH_BYTES = 5 * 1024 * 1024
#: Seconds before one search request is abandoned.
SEARCH_TIMEOUT_SECONDS = 20.0
#: Most results one search may return.
MAX_RESULTS = 25
#: The result anchor and snippet anchor classes DuckDuckGo uses.
RESULT_LINK_CLASS = "result__a"
RESULT_SNIPPET_CLASS = "result__snippet"
#: Host of the redirect wrapper DuckDuckGo wraps result URLs in.
_REDIRECT_HOST = "duckduckgo.com"
#: A browser user agent; a bot-looking request is answered with a challenge page.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _destination(href: str) -> str:
    """Return the real URL behind a DuckDuckGo result link.

    DuckDuckGo wraps some results in ``//duckduckgo.com/l/?uddg=<url>&rut=...``;
    the destination is the (already percent-decoded) ``uddg`` parameter.
    """
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith(_REDIRECT_HOST):
        through = urllib.parse.parse_qs(parsed.query).get("uddg")
        if through:
            return through[0]
    return href


class _ResultParser(HTMLParser):
    """Collect ``{title, url, snippet}`` from one DuckDuckGo results page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._href: str = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        href = attributes.get("href") or ""
        if RESULT_LINK_CLASS in classes and href:
            self._capture, self._href, self._text = "title", href, []
        elif RESULT_SNIPPET_CLASS in classes:
            self._capture, self._text = "snippet", []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._capture is None:
            return
        text = " ".join("".join(self._text).split())
        if self._capture == "title":
            self.results.append(
                {"title": text, "url": _destination(self._href), "snippet": ""}
            )
        elif text and self.results:
            self.results[-1]["snippet"] = text
        self._capture, self._href, self._text = None, "", []

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._text.append(data)


def parse_results(html: str) -> list[dict[str, str]]:
    """Parse a DuckDuckGo HTML results page into ``{title, url, snippet}`` rows.

    Split out from :func:`web_search` so the parser is testable against a saved
    page without a network call.
    """
    parser = _ResultParser()
    parser.feed(html)
    parser.close()
    return parser.results


def _clamp(limit: int) -> int:
    """Return ``limit`` clamped to ``[1, MAX_RESULTS]``, rejecting non-numbers."""
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise ToolInputError(f"max_results must be an integer, not {limit!r}") from None
    return max(1, min(value, MAX_RESULTS))


def _fetch(query: str, timeout: float) -> str:
    """POST ``query`` to DuckDuckGo and return the HTML, under a size cap."""
    body = urllib.parse.urlencode({"q": query}).encode("utf-8")
    request = urllib.request.Request(
        SEARCH_URL,
        data=body,
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/html",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_SEARCH_BYTES:
            raise ToolInputError(f"search response is too large: {length} bytes")
        raw = response.read(MAX_SEARCH_BYTES + 1)
    if len(raw) > MAX_SEARCH_BYTES:
        raise ToolInputError("search response exceeded the 5 MiB safety limit")
    return raw.decode("utf-8", "replace")


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web and return result titles, URLs, and snippets.

    Use this for current or external information no dataset tool covers: library
    and API documentation, a data provider's terms or endpoints, what an error
    message means, or a place name. Treat the results as leads, not facts: open
    the promising URL with ``download`` before relying on it. Keep
    ``max_results`` small (3-5) unless the user asked for a survey.
    """
    text = (query or "").strip()
    if not text:
        raise ToolInputError("web_search needs a non-empty query")
    limit = _clamp(max_results)
    results = parse_results(_fetch(text, SEARCH_TIMEOUT_SECONDS))
    if not results:
        raise ToolInputError(
            "DuckDuckGo returned no results (no matches, or the request was "
            "rate-limited); rephrase the query or try again shortly"
        )
    return results[:limit]


__all__ = ["MAX_RESULTS", "SEARCH_URL", "parse_results", "web_search"]
