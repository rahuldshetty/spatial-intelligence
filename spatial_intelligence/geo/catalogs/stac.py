"""STAC search against the public catalogs GeoLibre's STAC panel browses.

Two hosted catalogs are built in. AWS **Earth Search** indexes AWS Open Data
buckets (``sentinel-2-l2a``, ``landsat-c2-l2``, ``naip``, ``sentinel-1-grd``,
``cop-dem-glo-*``); Sentinel-2 publishes HTTPS asset URLs, the rest publish
``s3://`` URIs, which are translated here to the bucket's public HTTPS address.
That translation only reaches buckets readable anonymously: Earth Search's
Landsat Collection 2 and NAIP assets sit in requester-pays buckets and answer
403, so those collections belong on **Microsoft Planetary Computer**, whose
Azure blobs need a per-collection SAS token. Any other STAC API is reachable by
URL and is signed only when its host is Planetary Computer's.

That token is a short-lived credential, so it is never part of a scene: the stored
``asset_url`` and per-band hrefs are the catalog's own ones, the scene key is
derived from those, and a token is added when a URL is about to be used. The search
result the model sees carries signed URLs, because its usual next step is ``clip``
on a remote asset.

Scenes are normalized to the shape the Vantor and OpenAerialMap clients already
produce, so the download and map tools stay provider-agnostic: one ``scene_key``
per asset URL is the only thing the model has to carry between the search and
the layer.

Two behaviors are worth knowing, because they are the provider's, not ours:

* A cloud filter must be applied here rather than in the search body. Sending
  ``query: {"eo:cloud_cover": {"lt": n}}`` to a collection without that property
  (a DEM, for instance) does not error — it silently matches nothing, which
  reads as "no data" for a collection that has plenty.
* Static collections carry the dataset's own datetime, so a recent date window
  excludes them entirely. Callers omit the dates for those.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import quote, urlparse

from ...contracts.errors import ToolInputError
from .base import bbox_intersects, fetch_json, links, post_json, search_terms
from .scenes import SceneCache, scene_key

#: AWS Earth Search (Element 84): public HTTPS COG assets, no credentials.
EARTH_SEARCH_API = "https://earth-search.aws.element84.com/v1"
#: Microsoft Planetary Computer: Azure blob assets behind a SAS token.
PLANETARY_COMPUTER_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
#: The host whose assets need signing.
PLANETARY_COMPUTER_HOST = "planetarycomputer.microsoft.com"
#: The per-collection SAS token endpoint.
PLANETARY_COMPUTER_SAS_API = f"https://{PLANETARY_COMPUTER_HOST}/api/sas/v1/token"
#: Catalog used when the caller names none.
DEFAULT_CATALOG = "earth-search"

#: Collections requested per listing page.
MAX_COLLECTIONS_PER_PAGE = 1000
#: Listing pages followed in one call.
MAX_COLLECTION_PAGES = 3
#: Item pages followed in one search.
MAX_SEARCH_PAGES = 3
#: Most items requested per item page.
MAX_PAGE_LIMIT = 100
#: Most scenes one search returns.
MAX_LIMIT = 100
#: Most band assets remembered for one scene.
MAX_BAND_ASSETS = 24
#: Seconds trimmed from a token's expiry before it is fetched again.
SAS_EXPIRY_SKEW_SECONDS = 60.0
#: Token lifetime assumed when the response carries no usable expiry.
SAS_FALLBACK_TTL_SECONDS = 45 * 60.0
#: Accepted ``sort`` values.
SORTS = ("date", "cloud")
#: Asset names preferred as a scene's display raster, best first.
DISPLAY_ASSETS = ("visual", "data", "image", "cog_default")

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class Catalog:
    """One STAC API: where it is, what it is called, and whether it signs."""

    key: str
    label: str
    api: str
    description: str
    signed: bool = False
    collections: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        """Return the model-facing description of this catalog."""
        return {
            "catalog": self.key,
            "name": self.label,
            "api": self.api,
            "signed": self.signed,
            "description": self.description,
            "common_collections": list(self.collections),
        }


#: The process-wide token cache, created on first use (see ``_shared_signer``).
_SHARED_SIGNER: "SasTokenCache | None" = None

#: The built-in catalogs. ``common_collections`` is a short, verified hint list
#: (not the catalog's contents): ``list_collections`` is the authority.
CATALOGS: Mapping[str, Catalog] = {
    "earth-search": Catalog(
        key="earth-search",
        label="Earth Search",
        api=EARTH_SEARCH_API,
        description=(
            "AWS Open Data catalog by Element 84. Sentinel-2 publishes HTTPS "
            "Cloud-Optimized GeoTIFFs directly; other collections publish s3:// "
            "URIs, which this build reads over public HTTPS. Landsat Collection 2 "
            "and NAIP live in requester-pays buckets, so search those on "
            "planetary-computer instead."
        ),
        # Only collections whose assets this module can actually fetch: an id here
        # is one the model will try, so Landsat C2 and NAIP stay out (their buckets
        # answer 403) and are named in the description above instead.
        collections=(
            "sentinel-2-l2a",
            "sentinel-1-grd",
            "cop-dem-glo-30",
            "cop-dem-glo-90",
        ),
    ),
    "planetary-computer": Catalog(
        key="planetary-computer",
        label="Planetary Computer",
        api=PLANETARY_COMPUTER_API,
        description=(
            "Microsoft's STAC catalog (Sentinel, Landsat, HLS, land cover). Asset "
            "URLs are signed with a per-collection SAS token when they are used."
        ),
        signed=True,
        # 3dep-lidar-copc is deliberately absent: its items publish LASzip point
        # clouds, so a search there finds items and builds no scene from any.
        collections=(
            "sentinel-2-l2a",
            "landsat-c2-l2",
            "hls2-l30",
            "io-lulc-annual-v02",
        ),
    ),
}


def list_catalogs() -> list[dict]:
    """Return every built-in STAC catalog, with its common collection ids."""
    return [catalog.as_dict() for catalog in CATALOGS.values()]


def resolve_catalog(reference: str) -> Catalog:
    """Resolve a built-in catalog key or any HTTPS STAC API URL."""
    name = (reference or DEFAULT_CATALOG).strip()
    known = CATALOGS.get(name.lower())
    if known is not None:
        return known
    if name.startswith("https://"):
        api = name.rstrip("/")
        host = urlparse(api).netloc.lower()
        return Catalog(
            key=api,
            label=host,
            api=api,
            description=f"A STAC API at {host}.",
            signed=host == PLANETARY_COMPUTER_HOST,
        )
    known_keys = ", ".join(CATALOGS)
    raise ToolInputError(
        f"unknown STAC catalog {reference!r}: pass one of {known_keys}, "
        "or an https:// STAC API URL"
    )


def list_collections(api: str, query: str = "", limit: int = 50) -> list[dict]:
    """Search a catalog's collections by keyword and return the ranked matches.

    ``query`` is matched against each collection's id, title, description, and
    keywords; matches are ranked by how many query terms they carry, so a
    multi-word query still finds the right collection. An empty query lists the
    catalog's collections by id.
    """
    limit = max(1, min(limit, MAX_LIMIT))
    terms = search_terms(query)
    found: list[dict] = []
    seen: set[str] = set()
    for document in _collection_pages(api):
        for raw in document.get("collections") or []:
            if not isinstance(raw, dict):
                continue
            collection_id = str(raw.get("id") or "").strip()
            if not collection_id or collection_id in seen:
                continue
            seen.add(collection_id)
            found.append(_collection_entry(raw, collection_id))
    if terms:
        ranked = sorted(
            (
                (len(terms & search_terms(_collection_haystack(entry))), entry)
                for entry in found
            ),
            key=lambda item: -item[0],
        )[:limit]
        return [entry for score, entry in ranked if score]
    found.sort(key=lambda entry: entry["id"])
    return found[:limit]


def search_scenes(
    catalog: Catalog,
    *,
    collections: list[str],
    cache: SceneCache,
    bounds: list[float] | None = None,
    start: str | None = None,
    end: str | None = None,
    cloud_max: float | None = None,
    sort: str = "date",
    limit: int = 20,
    sas: "SasTokenCache | None" = None,
) -> dict:
    """Search one catalog's items and remember each usable scene in ``cache``.

    Returns ``{"catalog", "matched", "scenes"}``: ``matched`` is the provider's
    own count of items in the query area when it reports one (``None`` when it
    does not), so the caller can tell "nothing published" apart from "this page
    was capped". Scenes whose item carries no readable GeoTIFF asset are dropped.
    """
    requested = [str(name).strip() for name in collections if str(name).strip()]
    if not requested:
        raise ToolInputError(
            "collections is required: pass one or more collection ids from "
            "search_stac_collections"
        )
    if bounds is not None and (
        len(bounds) != 4 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]
    ):
        raise ToolInputError("bounds must be [west, south, east, north]")
    if cloud_max is not None and not 0 <= cloud_max <= 100:
        raise ToolInputError("cloud_max must be a percentage between 0 and 100")
    if sort not in SORTS:
        raise ToolInputError(f"sort must be one of {', '.join(SORTS)}")
    limit = max(1, min(int(limit), MAX_LIMIT))
    interval = _datetime_interval(start, end)

    payload: dict[str, Any] = {
        "collections": list(dict.fromkeys(requested)),
        "limit": max(1, min(limit, MAX_PAGE_LIMIT)),
    }
    if bounds is not None:
        payload["bbox"] = list(bounds)
    if interval is not None:
        payload["datetime"] = interval

    items, matched = _search_pages(catalog.api, payload)
    signer = _signer(catalog, sas)
    scenes: list[dict] = []
    seen: set[str] = set()
    unusable = 0
    for item in items:
        if not bbox_intersects(item.get("bbox"), bounds):
            continue
        scene = _scene(item, catalog=catalog, requested=requested, cache=cache)
        if scene is None:
            unusable += 1
            continue
        if scene["scene_key"] in seen:
            continue
        if not _cloud_matches(scene.get("cloud_cover"), cloud_max):
            continue
        seen.add(scene["scene_key"])
        scenes.append(scene)
    _sort_scenes(scenes, sort)
    return {
        "catalog": catalog.key,
        "matched": matched,
        "scenes": [_present(scene, signer) for scene in scenes[:limit]],
        # Items the provider returned that no scene could be built from: without
        # this, a collection whose assets are point clouds or PDFs reads as "no
        # data in this area" rather than "nothing here is a raster".
        "unusable": unusable,
    }


class SasTokenCache:
    """Per-collection SAS tokens for a signed catalog, refreshed on expiry.

    Tokens are held in memory for the session: they are short-lived credentials,
    and a workspace file is exactly the wrong place for them. Scenes carry the
    catalog's own unsigned hrefs, so nothing this module caches holds a token.
    """

    def __init__(self, *, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._lock = threading.Lock()
        self._tokens: dict[str, tuple[str, float]] = {}

    def token(self, collection: str) -> str:
        """Return a usable token for ``collection``, fetching one when needed."""
        now = time.time()
        with self._lock:
            cached = self._tokens.get(collection)
            if cached is not None and cached[1] - SAS_EXPIRY_SKEW_SECONDS > now:
                return cached[0]
        token, expiry = self._fetch(collection)
        with self._lock:
            self._tokens[collection] = (token, expiry)
        return token

    # -- internals -------------------------------------------------------

    def _fetch(self, collection: str) -> tuple[str, float]:
        """Fetch one token; raise a model-visible error when it cannot be had."""
        url = f"{PLANETARY_COMPUTER_SAS_API}/{quote(collection, safe='')}"
        try:
            body = fetch_json(url, timeout=self._timeout)
        except Exception as exc:  # transport errors become actionable ones
            raise ToolInputError(
                f"could not get a signing token for collection {collection!r}: {exc}"
            ) from exc
        token = body.get("token")
        if not isinstance(token, str) or not token:
            raise ToolInputError(
                f"the signing endpoint returned no token for collection {collection!r}"
            )
        expiry = _expiry_epoch(body.get("msft:expiry")) or (
            time.time() + SAS_FALLBACK_TTL_SECONDS
        )
        return token, expiry


# -- internals -----------------------------------------------------------


def _collection_pages(api: str) -> list[dict]:
    """Fetch the catalog's collection pages (bounded), newest page last."""
    pages: list[dict] = []
    document = fetch_json(f"{api}/collections?limit={MAX_COLLECTIONS_PER_PAGE}")
    pages.append(document)
    for _ in range(MAX_COLLECTION_PAGES - 1):
        next_link = next(
            (link for link in links(document, "next") if isinstance(link.get("href"), str)),
            None,
        )
        if next_link is None:
            break
        document = fetch_json(str(next_link["href"]))
        pages.append(document)
    return pages


def _collection_entry(raw: dict, collection_id: str) -> dict:
    """Return one model-facing collection description."""
    description = " ".join(str(raw.get("description") or "").split())
    return {
        "id": collection_id,
        "title": str(raw.get("title") or collection_id),
        "description": description[:300],
        "keywords": [str(word) for word in raw.get("keywords") or []][:12],
    }


def _collection_haystack(entry: dict) -> str:
    """Return the text a query is matched against for one collection."""
    return " ".join(
        (entry["id"], entry["title"], entry["description"], *entry["keywords"])
    )


def _datetime_interval(start: str | None, end: str | None) -> str | None:
    """Build a STAC ``datetime`` interval from two ``YYYY-MM-DD`` bounds.

    Full RFC 3339 timestamps are required by the catalogs we search: a bare
    ``2024-01-01/2024-03-01`` interval is rejected as invalid.
    """
    for value in (start, end):
        if value is not None and not _DATE.match(value):
            raise ToolInputError("start_date and end_date must be YYYY-MM-DD")
    if start and end and start > end:
        raise ToolInputError("start_date must not be after end_date")
    if not start and not end:
        return None
    first = f"{start}T00:00:00Z" if start else ".."
    last = f"{end}T23:59:59Z" if end else ".."
    return f"{first}/{last}"


def _search_pages(api: str, payload: dict) -> tuple[list[dict], int | None]:
    """Post one STAC item search and follow its ``next`` links (bounded)."""
    items: list[dict] = []
    matched: int | None = None
    body = post_json(f"{api}/search", payload)
    for page in range(MAX_SEARCH_PAGES):
        context = body.get("context")
        if isinstance(context, dict) and isinstance(context.get("matched"), int):
            matched = context["matched"]
        items.extend(
            item for item in body.get("features") or [] if isinstance(item, dict)
        )
        if page + 1 >= MAX_SEARCH_PAGES:
            break
        next_link = next(
            (link for link in links(body, "next") if isinstance(link.get("href"), str)),
            None,
        )
        if next_link is None:
            break
        body = _next_page(next_link, payload)
    return items, matched


def _next_page(link: dict, payload: dict) -> dict:
    """Fetch the next search page, honoring the link's declared method."""
    href = str(link["href"])
    if str(link.get("method") or "GET").upper() == "POST":
        body = link.get("body")
        return post_json(href, body if isinstance(body, dict) else payload)
    return fetch_json(href)


def _shared_signer() -> "SasTokenCache":
    """Return the process-wide token cache, created on first use.

    Tokens are per collection and short-lived, so one cache serves every caller
    that has none of its own; the tools pass their session's cache instead.
    """
    global _SHARED_SIGNER
    if _SHARED_SIGNER is None:
        _SHARED_SIGNER = SasTokenCache()
    return _SHARED_SIGNER


def _signer(catalog: Catalog, sas: "SasTokenCache | None") -> "SasTokenCache | None":
    """Return the token cache a catalog needs, or ``None`` when it needs none."""
    if not catalog.signed:
        return None
    return sas if sas is not None else SasTokenCache()


def _scene(
    item: dict,
    *,
    catalog: Catalog,
    requested: list[str],
    cache: SceneCache,
) -> dict | None:
    """Normalize one STAC item into a cached scene, or ``None`` when unusable.

    The stored hrefs are the item's own, unsigned ones: a signed URL carries a
    credential that rotates, so it must not decide a scene's identity (the key is
    derived from the href) and must not reach the scene cache. ``_present`` signs
    a copy for the caller, and :func:`asset_href` signs at every later use.
    """
    item_id = str(item.get("id") or "").strip()
    assets = item.get("assets")
    if not item_id or not isinstance(assets, dict):
        return None
    bands = _geotiff_assets(assets)
    display = _display_asset(bands)
    if display is None:
        return None
    properties = item.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    collection = str(item.get("collection") or requested[0])
    asset_url = display[1]
    scene = {
        "scene_key": scene_key("stac", item_id, asset_url),
        "id": item_id,
        "title": str(properties.get("title") or item_id),
        "provider": catalog.label,
        "catalog": catalog.key,
        "collection": collection,
        "datetime": properties.get("datetime") or properties.get("start_datetime"),
        "sensor": properties.get("platform") or properties.get("constellation"),
        "cloud_cover": properties.get("eo:cloud_cover"),
        "gsd": properties.get("gsd") or properties.get("eo:gsd") or assets[display[0]].get("gsd"),
        "bbox": item.get("bbox"),
        "thumbnail_url": _thumbnail(assets),
        "asset_url": asset_url,
        "asset": display[0],
        "bands": dict(bands),
        "signed": catalog.signed,
        "render": "cog",
    }
    return cache.remember(scene)


def _present(scene: dict, signer: "SasTokenCache | None") -> dict:
    """Return the scene as the caller sees it, with asset URLs signed for use.

    The model is handed a URL it can fetch right away — ``clip`` on a remote asset
    is the normal next step — while the cached scene keeps the unsigned href.
    """
    if signer is None:
        return dict(scene)
    collection = str(scene.get("collection") or "")
    return dict(
        scene,
        asset_url=_signed(str(scene.get("asset_url") or ""), signer, collection),
        bands={
            name: _signed(href, signer, collection)
            for name, href in (scene.get("bands") or {}).items()
        },
    )


def _signed(href: str, signer: "SasTokenCache", collection: str) -> str:
    """Return ``href`` with ``collection``'s token appended."""
    token = signer.token(collection)
    if not token or not href:
        return href
    separator = "&" if urlparse(href).query else "?"
    return f"{href}{separator}{token}"


def asset_href(scene: dict, *, sas: "SasTokenCache | None" = None) -> str:
    """Return the scene's asset URL, signing it when its catalog needs that.

    Signing at use is what keeps a scene's identity, its cached copy, and the
    layer metadata derived from it free of a credential that expires within a day:
    the scene stores the unsigned href, and this is where a fresh token is added.
    """
    href = str(scene.get("asset_url") or "")
    if not href:
        raise ToolInputError(f"scene {scene.get('scene_key')!r} has no asset URL")
    if not scene.get("signed"):
        return href
    return _signed(href, sas or _shared_signer(), str(scene.get("collection") or ""))


def _geotiff_assets(assets: dict) -> dict[str, str]:
    """Return the item's GeoTIFF assets as ``name -> https href`` (bounded).

    Earth Search publishes many assets as ``s3://`` URIs rather than URLs; the
    public ones are reachable over HTTPS at the bucket's virtual-host address,
    so they are translated here and the archive stays usable. Addresses that
    cannot become an anonymous HTTPS URL are dropped rather than carried into a
    download that is guaranteed to fail.
    """
    direct: dict[str, str] = {}
    translated: dict[str, str] = {}
    for name, asset in assets.items():
        if not isinstance(asset, dict):
            continue
        media_type = str(asset.get("type") or "").lower()
        if "tiff" not in media_type and "geotiff" not in media_type:
            continue
        href = _https_href(asset.get("href"))
        if href is None:
            continue
        bucket = direct if href == asset.get("href") else translated
        bucket.setdefault(str(name), href)
    merged = {**direct, **translated}
    return dict(list(merged.items())[:MAX_BAND_ASSETS])


def _https_href(href: Any) -> str | None:
    """Return an anonymous HTTPS URL for an asset href, or ``None``.

    ``s3://bucket/key`` becomes the bucket's virtual-host HTTPS address, which
    is how the public AWS Open Data buckets that Earth Search indexes are read.
    A requester-pays bucket answers 403 there, so a collection served that way
    (Landsat Collection 2, NAIP) is better searched on Planetary Computer.
    """
    if not isinstance(href, str) or not href:
        return None
    if href.startswith("https://"):
        return href
    if href.startswith("s3://"):
        bucket, _, key = href[5:].partition("/")
        if bucket and key:
            return f"https://{bucket}.s3.amazonaws.com/{key}"
    return None


def _display_asset(bands: dict[str, str]) -> tuple[str, str] | None:
    """Return the ``(name, href)`` of the asset a scene should point at."""
    for name in DISPLAY_ASSETS:
        href = bands.get(name)
        if href:
            return name, href
    for name, href in bands.items():
        return name, href
    return None


def _thumbnail(assets: dict) -> str | None:
    """Return the item's preview image, if it has one."""
    for name in ("thumbnail", "rendered_preview"):
        asset = assets.get(name)
        href = asset.get("href") if isinstance(asset, dict) else None
        if isinstance(href, str) and href.startswith("https://"):
            return href
    return None


def _cloud_matches(cloud_cover: Any, cloud_max: float | None) -> bool:
    """Whether a scene passes the cloud filter; unknown cloud cover passes."""
    if cloud_max is None or cloud_cover is None:
        return True
    try:
        return float(cloud_cover) <= cloud_max
    except (TypeError, ValueError):
        return True


def _sort_scenes(scenes: list[dict], sort: str) -> None:
    """Sort in place: newest first, or clearest first.

    Under ``cloud``, scenes whose cloud cover is unknown sort last and keep
    newest-first order among themselves — a collection with no cloud property
    (a DEM) then reads exactly as it does under ``date``, while a mixed search
    still prefers a scene it can vouch for.
    """
    by_date = lambda scene: str(scene.get("datetime") or "")  # noqa: E731 - key
    if sort == "date":
        scenes.sort(key=by_date, reverse=True)
        return
    scenes.sort(key=by_date, reverse=True)
    scenes.sort(
        key=lambda scene: (
            scene.get("cloud_cover") is None,
            float(scene.get("cloud_cover") or 0.0),
        )
    )


def _expiry_epoch(raw: Any) -> float | None:
    """Parse a token's ISO expiry into an epoch, or ``None`` when unusable."""
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


__all__ = [
    "CATALOGS",
    "DEFAULT_CATALOG",
    "EARTH_SEARCH_API",
    "PLANETARY_COMPUTER_API",
    "PLANETARY_COMPUTER_SAS_API",
    "SORTS",
    "Catalog",
    "SasTokenCache",
    "asset_href",
    "list_catalogs",
    "list_collections",
    "resolve_catalog",
    "search_scenes",
]
