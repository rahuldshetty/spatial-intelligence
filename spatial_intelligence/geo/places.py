"""Place-name lookup for the wizard's AOI field and the agent's geocode tool.

Nominatim's public search endpoint, queried with a User-Agent it asks clients to
send and a small response cap. It returns a JSON *array*, which is why this does
not reuse the catalog client (that one insists on an object).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from ..contracts.errors import ToolInputError

NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "spatial-intelligence/0.1 (place lookup)"
MAX_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 20.0
#: Places whose bounds are huge are still answered; the AOI field shows the box.
MAX_RESULTS = 5


def geocode(place: str) -> dict:
    """Resolve a place name to ``{label, bbox, center, type}``.

    ``bbox`` is ``[west, south, east, north]`` in degrees, the same order every
    tool in this harness takes.
    """
    query = " ".join(str(place or "").split())
    if not query:
        raise ToolInputError("place must not be empty")
    url = NOMINATIM_SEARCH + "?" + urllib.parse.urlencode(
        {"q": query, "format": "jsonv2", "limit": MAX_RESULTS}
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_BYTES)
    except Exception as exc:  # urllib raises a wide family here
        raise ToolInputError(f"place lookup failed for {query!r}: {exc}") from exc
    try:
        results = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise ToolInputError("place lookup did not return JSON") from exc
    if not isinstance(results, list) or not results:
        raise ToolInputError(f"no place found for {query!r}")
    found = results[0]
    corners = found.get("boundingbox")
    if not isinstance(corners, list) or len(corners) != 4:
        raise ToolInputError(f"place lookup returned no bounds for {query!r}")
    try:
        south, north, west, east = (float(corner) for corner in corners)
    except (TypeError, ValueError) as exc:
        raise ToolInputError(f"place lookup returned bad bounds for {query!r}") from exc
    center = None
    if found.get("lon") is not None and found.get("lat") is not None:
        center = [float(found["lon"]), float(found["lat"])]
    return {
        "place": query,
        "label": found.get("display_name") or query,
        "bbox": [west, south, east, north],
        "center": center,
        "type": found.get("type"),
    }


__all__ = ["geocode"]
