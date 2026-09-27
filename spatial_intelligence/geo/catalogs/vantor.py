"""Vantor Open Data: the disaster-event STAC catalog GeoLibre's plugin reads.

The catalog is a STAC root whose ``child`` links are per-event collections; each
collection links one STAC item per scene. Only HTTPS GeoTIFF/COG assets are
offered, because the map renders them as COG layers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

from ...contracts.errors import ToolInputError
from .base import event_id as event_id_of
from .base import bbox_intersects, fetch_json, links, search_terms
from .scenes import SceneCache, scene_key

#: STAC root of the Vantor Open Data event catalog.
VANTOR_CATALOG = "https://vantor-opendata.s3.amazonaws.com/events/catalog.json"
#: Most collection items fetched for one event.
MAX_EVENT_ITEMS = 500
#: Worker threads used to fetch collection items.
MAX_FETCH_WORKERS = 6
#: Accepted acquisition phases.
PHASES = frozenset({"all", "pre", "post"})


def search_vantor_events(query: str = "", limit: int = 20) -> list[dict]:
    """Search Vantor Open Data disaster events by title.

    This uses the Vantor STAC catalog consumed by GeoLibre's Vantor Open Data
    plugin. Call ``search_vantor_imagery`` with one returned event id.
    """
    catalog = fetch_json(VANTOR_CATALOG)
    query_terms = search_terms(query)
    events: list[tuple[int, dict]] = []
    for link in links(catalog, "child"):
        event = event_id_of(VANTOR_CATALOG, link)
        title = str(link.get("title") or event)
        score = len(query_terms & search_terms(title))
        if query_terms and score < len(query_terms):
            continue
        events.append(
            (
                score,
                {
                    "id": event,
                    "title": title,
                    "provider": "Vantor Open Data",
                },
            )
        )
    events.sort(key=lambda item: (-item[0], item[1]["title"]))
    return [event for _, event in events[: max(1, min(limit, 100))]]


def search_vantor_imagery(
    event_id: str,
    bounds: list[float] | None = None,
    phase: str = "all",
    limit: int = 30,
    *,
    cache: SceneCache,
) -> list[dict]:
    """List renderable Vantor scenes for an event, optionally filtered by AOI.

    ``bounds`` is ``[west, south, east, north]`` and ``phase`` is ``all``,
    ``pre``, or ``post``. Results contain stable scene keys, thumbnails, dates,
    resolution, cloud cover, and source COG URLs.
    """
    if bounds is not None and (
        len(bounds) != 4 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]
    ):
        raise ToolInputError("bounds must be [west, south, east, north]")
    if phase not in PHASES:
        raise ToolInputError("phase must be all, pre, or post")
    catalog = fetch_json(VANTOR_CATALOG)
    selected = None
    for link in links(catalog, "child"):
        candidate = event_id_of(VANTOR_CATALOG, link)
        if candidate == event_id or str(link.get("title") or "") == event_id:
            selected = urljoin(VANTOR_CATALOG, str(link["href"]))
            break
    if selected is None:
        raise ToolInputError(f"Vantor event not found: {event_id!r}")

    collection = fetch_json(selected)
    item_urls = [urljoin(selected, str(link["href"])) for link in links(collection, "item")]
    # Bound work even when a catalog unexpectedly contains thousands of items.
    item_urls = item_urls[:MAX_EVENT_ITEMS]
    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(MAX_FETCH_WORKERS, max(1, len(item_urls)))) as pool:
        futures = {pool.submit(fetch_json, url): url for url in item_urls}
        for future in as_completed(futures):
            try:
                items.append(future.result())
            except Exception:
                # One unreadable item must not hide its siblings.
                continue

    scenes = []
    for item in items:
        props = item.get("properties") or {}
        item_phase = str(props.get("phase") or "").lower().replace("-event", "")
        if phase != "all" and item_phase != phase:
            continue
        if not bbox_intersects(item.get("bbox"), bounds):
            continue
        cog_url = _https_asset(item, "visual")
        if not cog_url:
            continue
        item_id = str(item.get("id") or "unknown")
        thumbnail = (item.get("assets") or {}).get("thumbnail") or {}
        scene = {
            "scene_key": scene_key("vantor", item_id, cog_url),
            "id": item_id,
            "title": str(props.get("title") or item_id),
            "provider": "Vantor Open Data",
            "event": event_id,
            "datetime": props.get("datetime"),
            "phase": item_phase or None,
            "sensor": props.get("vehicle_name") or props.get("constellation"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "gsd": props.get("pan_gsd") or props.get("multispectral_gsd"),
            "bbox": item.get("bbox"),
            "thumbnail_url": thumbnail.get("href") if isinstance(thumbnail, dict) else None,
            "asset_url": cog_url,
            "render": "cog",
        }
        scenes.append(cache.remember(scene))
    scenes.sort(key=lambda scene: str(scene.get("datetime") or ""), reverse=True)
    return scenes[: max(1, min(limit, 100))]


def _https_asset(item: dict, preferred: str | None = None) -> str | None:
    """Return the item's HTTPS GeoTIFF asset, preferring ``preferred``."""
    assets = item.get("assets") or {}
    if preferred and isinstance(assets.get(preferred), dict):
        href = assets[preferred].get("href")
        if isinstance(href, str) and href.startswith("https://"):
            return href
    for asset in assets.values():
        if not isinstance(asset, dict):
            continue
        href = asset.get("href")
        media_type = str(asset.get("type") or "").lower()
        if (
            isinstance(href, str)
            and href.startswith("https://")
            and ("tiff" in media_type or "geotiff" in media_type)
        ):
            return href
    return None


__all__ = ["PHASES", "VANTOR_CATALOG", "search_vantor_events", "search_vantor_imagery"]
