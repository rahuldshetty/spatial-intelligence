"""ArcGIS Online / Hub public catalog adapter.

One client covers ArcGIS Hub, federal/state/local portals, and any public
ArcGIS organization: search items, resolve their Feature/Map/Image services, and
return one dataset per service layer.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal
from urllib.parse import urlencode

from ...contracts.errors import ToolInputError
from ...geo.catalogs.base import fetch_json
from ..base import (
    Asset,
    DataQuery,
    Dataset,
    DataSource,
    FilterSpec,
    LayerSpec,
    SearchPage,
    SourceCapability,
    SourceDescriptor,
    filter_list,
)

#: Public ArcGIS Online search endpoint.
ARCGIS_SEARCH = "https://www.arcgis.com/sharing/rest/search"
#: Item types this adapter understands.
SUPPORTED_TYPES: tuple[tuple[str, str], ...] = (
    ("Feature Service", "Feature service"),
    ("Map Service", "Map service"),
    ("Image Service", "Image service"),
    ("GeoJson", "GeoJSON"),
)
DEFAULT_TYPES = tuple(name for name, _ in SUPPORTED_TYPES)
#: Layers resolved per item; a service with hundreds of layers is not one map.
MAX_LAYERS_PER_ITEM = 8
#: Service metadata fetches in flight.
MAX_WORKERS = 6
#: Largest raster export requested, in pixels per side.
MAX_EXPORT_SIZE = 2048

_LUCENE_META = re.compile(r'["\\/(){}[\]^~:!?+]|&&|\|\|')


class ArcGISSource(DataSource):
    """Search public ArcGIS items and map their service layers."""

    descriptor = SourceDescriptor(
        id="arcgis",
        name="ArcGIS Hub / Public ArcGIS",
        summary=(
            "Search public ArcGIS Online and Hub items: feature, map, and image "
            "services from agencies, states, cities, and organizations."
        ),
        category="authoritative",
        homepage="https://hub.arcgis.com/",
        attribution="ArcGIS item owner and service copyright text",
        capabilities=frozenset(
            {
                SourceCapability.SEARCH,
                SourceCapability.SPATIAL,
                SourceCapability.DOWNLOAD,
                SourceCapability.MAP,
            }
        ),
        filters=(
            FilterSpec(
                "item_type",
                "Item type",
                "multiselect",
                default=list(DEFAULT_TYPES),
                choices=SUPPORTED_TYPES,
                help="Leave empty to search feature, map, image, and GeoJSON items.",
            ),
            FilterSpec(
                "sort",
                "Sort",
                "select",
                default="relevance",
                choices=(("relevance", "Relevance"), ("title", "Title")),
            ),
        ),
        docs_url="https://developers.arcgis.com/rest/services-reference/enterprise/search.htm",
    )

    def search(self, query: DataQuery) -> SearchPage:
        text = _sanitize(query.text)
        types = filter_list(query.filters.get("item_type")) or list(DEFAULT_TYPES)
        type_query = " OR ".join(f'type:"{name}"' for name in types if name in DEFAULT_TYPES)
        if not type_query:
            raise ToolInputError("item_type must name a supported ArcGIS item type")
        clauses = [f"({type_query})", "access:public"]
        if text:
            clauses.insert(0, f"({text})")
        try:
            start = int(query.cursor or "1")
        except (TypeError, ValueError) as exc:
            raise ToolInputError("ArcGIS cursor must be a numeric start") from exc
        start = max(1, start)
        params = {
            "q": " AND ".join(clauses),
            "f": "json",
            "num": str(min(max(query.limit * 2, 10), 100)),
            "start": str(start),
            "sortField": "relevance" if text and query.filters.get("sort") != "title" else "title",
            "sortOrder": "desc" if text and query.filters.get("sort") != "title" else "asc",
        }
        if query.bbox is not None:
            params["bbox"] = ",".join(str(value) for value in query.bbox)
        body = fetch_json(f"{ARCGIS_SEARCH}?{urlencode(params)}")
        items = body.get("results") if isinstance(body.get("results"), list) else []
        datasets = self._resolve_items([item for item in items if isinstance(item, dict)])
        total = body.get("total") if isinstance(body.get("total"), int) else None
        next_cursor = (
            str(start + len(items))
            if total is not None and start + len(items) < total
            else None
        )
        return SearchPage(
            datasets=tuple(datasets[: query.limit]),
            total=total,
            next_cursor=next_cursor,
        )

    def materialize(
        self, dataset: Dataset, asset: Asset | None = None
    ) -> tuple[LayerSpec, ...]:
        service_type = str(dataset.properties.get("service_type") or "")
        name = dataset.title
        if dataset.properties.get("cached"):
            chosen = asset or (dataset.assets[0] if dataset.assets else None)
            if chosen is None:
                raise ToolInputError(f"ArcGIS dataset {dataset.id!r} has no tile URL")
            return (
                LayerSpec(
                    kind="tiles",
                    uri=chosen.href,
                    name=name,
                    options={"asset_id": chosen.id},
                    metadata={
                        "license": dataset.license,
                        "attribution": dataset.attribution or "ArcGIS",
                        "source_url": chosen.href,
                    },
                ),
            )
        if service_type == "Feature Service":
            query_url = str(dataset.properties.get("query_url") or "")
            if not query_url:
                raise ToolInputError(f"ArcGIS dataset {dataset.id!r} has no query URL")
            return (
                LayerSpec(
                    kind="vector",
                    uri=query_url,
                    name=name,
                    options={
                        "asset_id": "data",
                        "media_type": "application/geo+json",
                        "arcgis_feature": {
                            "service_url": dataset.properties.get("service_url"),
                            "layer_id": dataset.properties.get("layer_id"),
                        },
                    },
                    metadata={
                        "license": dataset.license,
                        "attribution": dataset.attribution or "ArcGIS",
                        "source_url": query_url,
                    },
                ),
            )
        if service_type in ("Map Service", "Image Service"):
            export_url = str(dataset.properties.get("export_url") or "")
            if not export_url:
                raise ToolInputError(f"ArcGIS dataset {dataset.id!r} has no export URL")
            return (
                LayerSpec(
                    kind="raster",
                    uri=export_url,
                    name=name,
                    options={"asset_id": "data", "force_download": True},
                    metadata={
                        "license": dataset.license,
                        "attribution": dataset.attribution or "ArcGIS",
                        "source_url": export_url,
                    },
                ),
            )
        if service_type == "GeoJson":
            href = str(dataset.properties.get("query_url") or "")
            if not href:
                raise ToolInputError(f"ArcGIS dataset {dataset.id!r} has no GeoJSON URL")
            return (
                LayerSpec(
                    kind="vector",
                    uri=href,
                    name=name,
                    options={"asset_id": "data", "media_type": "application/geo+json"},
                    metadata={
                        "license": dataset.license,
                        "attribution": dataset.attribution or "ArcGIS",
                        "source_url": href,
                    },
                ),
            )
        raise ToolInputError(f"unsupported ArcGIS service type {service_type!r}")

    # -- internals -------------------------------------------------------

    def _resolve_items(self, items: list[dict]) -> list[Dataset]:
        if not items:
            return []
        results: dict[int, list[Dataset]] = {}
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(items))) as pool:
            futures = {
                pool.submit(self._datasets_for_item, item): index
                for index, item in enumerate(items)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception:
                    # One unreadable service must not hide its siblings.
                    results[index] = []
        datasets: list[Dataset] = []
        for index in range(len(items)):
            datasets.extend(results.get(index, []))
        return datasets

    def _datasets_for_item(self, item: dict) -> list[Dataset]:
        item_type = str(item.get("type") or "")
        url = item.get("url")
        if item_type not in DEFAULT_TYPES or not isinstance(url, str):
            return []
        if not url.startswith("https://"):
            return []
        if item_type == "GeoJson":
            return [self._geojson_dataset(item, url)]
        base = re.sub(r"/\d+/?$", "", url.rstrip("/"))
        info = _service_info(base)
        if not info:
            return []
        if item_type == "Feature Service":
            return self._feature_datasets(item, base, info)
        if item_type == "Map Service":
            return self._raster_datasets(item, base, info, image_service=False)
        return self._raster_datasets(item, base, info, image_service=True)

    def _feature_datasets(self, item: dict, base: str, info: dict) -> list[Dataset]:
        layers = info.get("layers") if isinstance(info.get("layers"), list) else []
        rows: list[Dataset] = []
        for layer in layers[:MAX_LAYERS_PER_ITEM]:
            if not isinstance(layer, dict) or layer.get("id") is None:
                continue
            layer_id = int(layer["id"])
            layer_name = str(layer.get("name") or f"Layer {layer_id}")
            query_url = (
                f"{base}/{layer_id}/query?where=1%3D1&outFields=*"
                "&returnGeometry=true&outSR=4326&f=geojson"
            )
            rows.append(
                self._dataset(
                    item,
                    dataset_id=f"{item.get('id')}:{layer_id}",
                    title=f"{item.get('title') or 'ArcGIS layer'} — {layer_name}",
                    kind="vector",
                    bbox=_item_bbox(item) or _extent_bbox(layer.get("extent")),
                    service_type="Feature Service",
                    service_url=base,
                    layer_id=layer_id,
                    query_url=query_url,
                    export_url=None,
                    media_type="application/geo+json",
                    layer_name=layer_name,
                    info=info,
                )
            )
        return rows

    def _raster_datasets(
        self, item: dict, base: str, info: dict, *, image_service: bool
    ) -> list[Dataset]:
        if image_service:
            bbox = _item_bbox(item) or _extent_bbox(info.get("fullExtent"))
            if bbox is None:
                return []
            export_url = _export_url(base, bbox, image_service=True)
            return [
                self._dataset(
                    item,
                    dataset_id=str(item.get("id") or base),
                    title=str(item.get("title") or "ArcGIS image service"),
                    kind="raster",
                    bbox=bbox,
                    service_type="Image Service",
                    service_url=base,
                    layer_id=0,
                    query_url=None,
                    export_url=export_url,
                    media_type="image/tiff",
                    layer_name=str(info.get("name") or "Image"),
                    info=info,
                )
            ]
        layers = info.get("layers") if isinstance(info.get("layers"), list) else []
        cached = _cached_map_service(info)
        rows: list[Dataset] = []
        for layer in layers[:MAX_LAYERS_PER_ITEM]:
            if not isinstance(layer, dict) or layer.get("id") is None:
                continue
            layer_id = int(layer["id"])
            bbox = _item_bbox(item) or _extent_bbox(layer.get("extent"))
            if bbox is None:
                continue
            layer_name = str(layer.get("name") or f"Layer {layer_id}")
            tile_url = f"{base}/tile/{{z}}/{{y}}/{{x}}" if cached else None
            export_url = (
                None
                if cached
                else _export_url(base, bbox, image_service=False, layer_id=layer_id)
            )
            rows.append(
                self._dataset(
                    item,
                    dataset_id=f"{item.get('id')}:{layer_id}",
                    title=f"{item.get('title') or 'ArcGIS layer'} — {layer_name}",
                    kind="tiles" if cached else "raster",
                    bbox=bbox,
                    service_type="Map Service",
                    service_url=base,
                    layer_id=layer_id,
                    query_url=None,
                    export_url=export_url,
                    tile_url=tile_url,
                    cached=cached,
                    media_type="image/png" if cached else "image/tiff",
                    layer_name=layer_name,
                    info=info,
                )
            )
        return rows

    def _geojson_dataset(self, item: dict, url: str) -> Dataset:
        title = str(item.get("title") or "ArcGIS GeoJSON")
        return self._dataset(
            item,
            dataset_id=str(item.get("id") or url),
            title=title,
            kind="vector",
            bbox=_item_bbox(item),
            service_type="GeoJson",
            service_url=url,
            layer_id=0,
            query_url=url,
            export_url=None,
            media_type="application/geo+json",
            layer_name=title,
            info={},
        )

    def _dataset(
        self,
        item: dict,
        *,
        dataset_id: str,
        title: str,
        kind: Literal["raster", "vector", "tiles"],
        bbox: tuple[float, float, float, float] | None,
        service_type: str,
        service_url: str,
        layer_id: int,
        query_url: str | None,
        export_url: str | None,
        media_type: str,
        layer_name: str,
        info: dict,
        tile_url: str | None = None,
        cached: bool = False,
    ) -> Dataset:
        license_info = item.get("licenseInfo")
        attribution = (
            info.get("copyrightText")
            or item.get("accessInformation")
            or item.get("owner")
            or "ArcGIS"
        )
        href = query_url or export_url or tile_url
        assets = (
            (
                Asset(
                    id="data",
                    href=str(href),
                    media_type=media_type,
                    role="data",
                    mode="tile" if tile_url else "download",
                ),
            )
            if href
            else ()
        )
        return Dataset(
            source_id=self.descriptor.id,
            id=dataset_id,
            title=title,
            kind=kind,
            description=str(item.get("snippet") or item.get("description") or "")[:500],
            bbox=bbox,
            datetime=item.get("modified"),
            thumbnail_url=item.get("thumbnail"),
            license=str(license_info) if license_info else None,
            attribution=str(attribution),
            properties={
                "service_type": service_type,
                "service_url": service_url,
                "item_id": item.get("id"),
                "item_type": item.get("type"),
                "item_url": item.get("url"),
                "layer_id": layer_id,
                "layer_name": layer_name,
                "query_url": query_url,
                "export_url": export_url,
                "cached": cached,
            },
            assets=assets,
        )


def _cached_map_service(info: dict) -> bool:
    """Whether a MapServer serves a cache and therefore cannot export a TIFF."""
    capabilities = str(info.get("capabilities") or "")
    return bool(
        info.get("singleFusedMapCache")
        or "TilesOnly" in capabilities
        or "Tilemap" in capabilities
    )


def _service_info(url: str) -> dict:
    try:
        body = fetch_json(f"{url}?f=json")
    except Exception:
        return {}
    if body.get("error"):
        return {}
    return body


def _sanitize(value: str) -> str:
    text = _LUCENE_META.sub(" ", value or "")
    text = re.sub(r"\b(?:AND|OR|NOT)\b", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _item_bbox(item: dict) -> tuple[float, float, float, float] | None:
    extent = item.get("extent")
    if not isinstance(extent, list) or len(extent) != 2:
        return None
    try:
        (west, south), (east, north) = extent
        west, south, east, north = float(west), float(south), float(east), float(north)
    except (TypeError, ValueError):
        return None
    return (west, south, east, north) if west < east and south < north else None


def _extent_bbox(extent: object) -> tuple[float, float, float, float] | None:
    if not isinstance(extent, dict):
        return None
    xmin, ymin, xmax, ymax = (
        extent.get("xmin"),
        extent.get("ymin"),
        extent.get("xmax"),
        extent.get("ymax"),
    )
    if None in (xmin, ymin, xmax, ymax):
        return None
    try:
        west, south, east, north = float(xmin), float(ymin), float(xmax), float(ymax)
    except (TypeError, ValueError):
        return None
    return (west, south, east, north) if west < east and south < north else None


def _export_url(
    base: str,
    bbox: tuple[float, float, float, float],
    *,
    image_service: bool,
    layer_id: int | None = None,
) -> str:
    width, height = _export_size(bbox)
    operation = "exportImage" if image_service else "export"
    params = {
        "bbox": ",".join(f"{value:.6f}" for value in bbox),
        "bboxSR": "4326",
        "imageSR": "3857",
        "size": f"{width},{height}",
        "format": "tiff",
        "f": "image",
    }
    if layer_id is not None and not image_service:
        params["layers"] = f"show:{layer_id}"
    return f"{base}/{operation}?{urlencode(params)}"


def _export_size(bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    west, south, east, north = bbox
    span_x = max(abs(east - west), 1e-6)
    span_y = max(abs(north - south), 1e-6)
    if span_x >= span_y:
        width = MAX_EXPORT_SIZE
        height = max(256, int(MAX_EXPORT_SIZE * span_y / span_x))
    else:
        height = MAX_EXPORT_SIZE
        width = max(256, int(MAX_EXPORT_SIZE * span_x / span_y))
    return width, height


__all__ = ["ARCGIS_SEARCH", "ArcGISSource"]
