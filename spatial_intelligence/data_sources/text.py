"""Reusable plain-text cleaning for provider metadata.

Catalogs are inconsistent: ArcGIS item descriptions routinely arrive as HTML
fragments with entities, images, and links, while other providers return plain
text.  The UI and the model should see readable text, so every dataset's title,
description, attribution, and license pass through one cleaner at the source
boundary.
"""

from __future__ import annotations

import html
import re
from dataclasses import replace
from html.parser import HTMLParser

from .base import Dataset

_WHITESPACE = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5",
        "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "td", "th", "tr", "ul",
    }
)
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})


class _Extractor(HTMLParser):
    """Collect text nodes, separating block elements and skipping scripts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def clean_text(value: object, *, limit: int | None = None) -> str:
    """Return readable plain text from plain text or an HTML fragment.

    Tags are stripped, common entities decoded, scripts/styles dropped, and
    whitespace collapsed. ``limit`` truncates after cleaning, never before it.
    """
    raw = "" if value is None else str(value)
    if not raw:
        return ""
    if "<" in raw or "&" in raw:
        parser = _Extractor()
        try:
            parser.feed(raw)
            parser.close()
            raw = parser.text()
        except Exception:
            # A malformed fragment must not fail the catalog result; fall back
            # to a conservative tag strip and still decode entities below.
            raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = _CONTROL.sub(" ", raw)
    raw = _WHITESPACE.sub(" ", raw).strip()
    if limit is not None and limit > 0 and len(raw) > limit:
        raw = raw[:limit].rstrip()
    return raw


def clean_dataset(dataset: Dataset) -> Dataset:
    """Return a dataset whose user-facing text fields are safe to display."""
    return replace(
        dataset,
        title=clean_text(dataset.title, limit=240) or dataset.id,
        description=clean_text(dataset.description, limit=800),
        attribution=clean_text(dataset.attribution, limit=240) or None,
        license=clean_text(dataset.license, limit=240) or None,
    )


__all__ = ["clean_dataset", "clean_text"]
