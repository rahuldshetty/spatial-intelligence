"""NASA GIBS adapter: date-addressable global imagery tiles."""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from datetime import date, timedelta

from ...contracts.errors import ToolInputError
from ...geo.catalogs.base import fetch_text
from ..base import (
    DataQuery,
    Dataset,
    DataSource,
    FilterSpec,
    LayerSpec,
    SearchPage,
    SourceCapability,
    SourceDescriptor,
)

#: WMTS capabilities for the Web Mercator tile matrix set.
GIBS_CAPABILITIES = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/WMTSCapabilities.xml"
#: The layer's default date when the client does not name one.
DEFAULT_DATE_LAG_DAYS = 1

_OWS = "http://www.opengis.net/ows/1.1"
_WMTS = "http://www.opengis.net/wmts/1.0"
_NS = {"ows": _OWS, "wmts": _WMTS}

_LAYERS: tuple[Dataset, ...] | None = None
_LOCK = threading.Lock()


class GIBSSource(DataSource):
    """Search 1,100+ pre-rendered global imagery layers and add them by date."""

    descriptor = SourceDescriptor(
        id="gibs",
        name="NASA GIBS",
        summary=(
            "NASA Global Imagery Browse Services: daily MODIS/VIIRS true color, "
            "vegetation, temperature, fire, snow, and nighttime-light tiles."
        ),
        category="imagery",
        homepage="https://earthdata.nasa.gov/gibs",
        license="Open; NASA EOSDIS GIBS attribution required",
        attribution="NASA EOSDIS GIBS",
        capabilities=frozenset(
            {
                SourceCapability.SEARCH,
                SourceCapability.TEMPORAL,
                SourceCapability.MAP,
            }
        ),
        filters=(
            FilterSpec(
                "date",
                "Date",
                "date",
                help="Imagery date; defaults to yesterday because GIBS publishes with a lag.",
            ),
        ),
        docs_url="https://earthdata.nasa.gov/gibs",
    )

    def search(self, query: DataQuery) -> SearchPage:
        layers = _load_layers()
        terms = query.text.lower().split()
        matched = [
            layer
            for layer in layers
            if not terms
            or all(
                term
                in f"{layer.title} {layer.id} {layer.description}".lower()
                for term in terms
            )
        ]
        chosen_date = _selected_date(query)
        rows = tuple(
            _with_date(layer, chosen_date) for layer in matched[: query.limit]
        )
        return SearchPage(datasets=rows, total=len(matched), next_cursor=None)

    def materialize(
        self, dataset: Dataset, asset=None
    ) -> tuple[LayerSpec, ...]:
        template = str(dataset.properties.get("tile_template") or "")
        if not template:
            raise ToolInputError(f"GIBS layer {dataset.id!r} has no tile template")
        tile_date = str(dataset.properties.get("date") or _default_date())
        if "{Time}" in template:
            template = template.replace("{Time}", tile_date)
        template = template.replace(
            "{TileMatrixSet}", str(dataset.properties.get("tile_matrix_set") or "")
        )
        template = template.replace("{TileMatrix}", "{z}")
        template = template.replace("{TileRow}", "{y}")
        template = template.replace("{TileCol}", "{x}")
        name = f"NASA GIBS {dataset.title}"
        if "{z}" in template and dataset.properties.get("time_enabled"):
            name = f"{name} {tile_date}"
        return (
            LayerSpec(
                kind="wmts",
                uri=template,
                name=name,
                options={"tile_size": 256},
                metadata={
                    "license": dataset.license,
                    "attribution": dataset.attribution,
                    "source_url": template,
                    "date": tile_date,
                },
            ),
        )


def _selected_date(query: DataQuery) -> str:
    """Return an ISO date from the query, defaulting to yesterday."""
    value = query.filters.get("date") or query.start or query.end
    if not value:
        return _default_date()
    text = str(value).strip()
    if len(text) < 10:
        raise ToolInputError("date must be YYYY-MM-DD")
    return text[:10]


def _default_date() -> str:
    return (date.today() - timedelta(days=DEFAULT_DATE_LAG_DAYS)).isoformat()


def _with_date(layer: Dataset, value: str) -> Dataset:
    properties = dict(layer.properties)
    if properties.get("time_enabled"):
        properties["date"] = value
    else:
        properties["date"] = ""
    return Dataset(
        source_id=layer.source_id,
        id=layer.id,
        title=layer.title,
        kind=layer.kind,
        description=layer.description,
        bbox=layer.bbox,
        datetime=value if properties.get("time_enabled") else None,
        thumbnail_url=layer.thumbnail_url,
        license=layer.license,
        attribution=layer.attribution,
        properties=properties,
        assets=layer.assets,
    )


def _load_layers() -> tuple[Dataset, ...]:
    """Fetch and parse the GIBS capabilities once, then reuse the layer list."""
    global _LAYERS
    with _LOCK:
        if _LAYERS is not None:
            return _LAYERS
        try:
            root = ET.fromstring(fetch_text(GIBS_CAPABILITIES))
        except ET.ParseError as exc:
            raise ToolInputError(f"GIBS capabilities were not valid XML: {exc}") from exc
        rows: list[Dataset] = []
        for layer in root.findall(".//wmts:Layer", _NS):
            dataset = _parse_layer(layer)
            if dataset is not None:
                rows.append(dataset)
        _LAYERS = tuple(rows)
        return _LAYERS


def _parse_layer(layer: ET.Element) -> Dataset | None:
    layer_id = layer.findtext("ows:Identifier", namespaces=_NS)
    if not layer_id:
        return None
    title = layer.findtext("ows:Title", namespaces=_NS) or layer_id
    abstract = layer.findtext("ows:Abstract", namespaces=_NS) or ""
    tile_template = ""
    tile_format = ""
    time_enabled = False
    for resource in layer.findall("wmts:ResourceURL", _NS):
        if resource.get("resourceType") != "tile":
            continue
        template = resource.get("template") or ""
        if not template:
            continue
        if not tile_template or "{Time}" in template:
            tile_template = template
            tile_format = resource.get("format") or ""
        time_enabled = time_enabled or "{Time}" in template
    if not tile_template:
        return None
    matrix_set = layer.findtext(
        "wmts:TileMatrixSetLink/wmts:TileMatrixSet", namespaces=_NS
    ) or ""
    bbox = _bbox(layer)
    default_date = ""
    for dimension in layer.findall("wmts:Dimension", _NS):
        name = dimension.findtext("ows:Identifier", namespaces=_NS)
        if name == "Time":
            default_date = dimension.findtext("ows:Default", namespaces=_NS) or ""
            break
    return Dataset(
        source_id="gibs",
        id=layer_id,
        title=title,
        kind="tiles",
        description=abstract,
        bbox=bbox,
        datetime=default_date or None,
        attribution="NASA EOSDIS GIBS",
        license="Open; NASA EOSDIS GIBS attribution required",
        properties={
            "tile_template": tile_template,
            "tile_matrix_set": matrix_set,
            "tile_format": tile_format,
            "time_enabled": time_enabled,
        },
    )


def _bbox(layer: ET.Element) -> tuple[float, float, float, float] | None:
    box = layer.find("ows:WGS84BoundingBox", _NS)
    if box is None:
        return None
    lower = box.findtext("ows:LowerCorner", namespaces=_NS)
    upper = box.findtext("ows:UpperCorner", namespaces=_NS)
    if not lower or not upper:
        return None
    try:
        west, south = (float(value) for value in lower.split())
        east, north = (float(value) for value in upper.split())
    except (ValueError, TypeError):
        return None
    if west >= east or south >= north:
        return None
    return (west, south, east, north)


def reset_cache() -> None:
    """Drop the parsed capabilities (tests use this between canned responses)."""
    global _LAYERS
    with _LOCK:
        _LAYERS = None


__all__ = ["GIBS_CAPABILITIES", "GIBSSource", "reset_cache"]
