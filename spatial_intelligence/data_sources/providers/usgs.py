"""USGS 3D Elevation Program (3DEP) adapter over The National Map API."""

from __future__ import annotations

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

#: The National Map products endpoint.
TNM_PRODUCTS = "https://tnmaccess.nationalmap.gov/api/v1/products"
#: Datasets the filter offers; the API accepts the same strings.
DEM_DATASETS: tuple[tuple[str, str], ...] = (
    ("Digital Elevation Model (DEM) 1 meter", "1 m DEM"),
    ("National Elevation Dataset (NED) 1/3 arc-second", "1/3 arc-second NED"),
    ("National Elevation Dataset (NED) 1 arc-second", "1 arc-second NED"),
    ("Digital Elevation Model (DEM) 1/3 arc-second", "1/3 arc-second DEM"),
    ("Digital Elevation Model (DEM) 1 arc-second", "1 arc-second DEM"),
    ("Alaska 5 meter DEM", "Alaska 5 m DEM"),
)
DEFAULT_DATASETS = tuple(name for name, _ in DEM_DATASETS[:3])


class USGSSource(DataSource):
    """Search 3DEP elevation products and download their GeoTIFF tiles."""

    descriptor = SourceDescriptor(
        id="usgs-3dep",
        name="USGS 3DEP Elevation",
        summary=(
            "USGS 3D Elevation Program DEMs: 1 m, 1/3 arc-second, and 1 arc-second "
            "tiles for the United States, as direct GeoTIFF downloads."
        ),
        category="elevation",
        homepage="https://www.usgs.gov/3d-elevation-program",
        license="Public domain (US Government)",
        attribution="U.S. Geological Survey, 3DEP",
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
                "datasets",
                "Elevation datasets",
                "multiselect",
                default=list(DEFAULT_DATASETS),
                choices=DEM_DATASETS,
                help="Leave empty to use the three standard DEM products.",
            ),
            FilterSpec(
                "format",
                "Format",
                "select",
                default="GeoTIFF",
                choices=(("GeoTIFF", "GeoTIFF"), ("IMG", "IMG"), ("All", "All")),
            ),
        ),
        docs_url="https://apps.nationalmap.gov/services/",
        bbox="required",
    )

    def search(self, query: DataQuery) -> SearchPage:
        if query.bbox is None:
            raise ToolInputError(
                "USGS 3DEP needs an area of interest: pass bbox "
                "[west, south, east, north]"
            )
        datasets = filter_list(query.filters.get("datasets")) or list(DEFAULT_DATASETS)
        file_format = str(query.filters.get("format") or "GeoTIFF")
        try:
            offset = int(query.cursor or "0")
        except (TypeError, ValueError) as exc:
            raise ToolInputError("USGS 3DEP cursor must be a numeric offset") from exc
        params = {
            "datasets": ",".join(datasets),
            "bbox": ",".join(str(value) for value in query.bbox),
            "max": str(query.limit),
            "offset": str(max(0, offset)),
        }
        if file_format and file_format != "All":
            params["prodFormats"] = file_format
        body = fetch_json(f"{TNM_PRODUCTS}?{urlencode(params)}")
        items = body.get("items") if isinstance(body.get("items"), list) else []
        rows = [self._item_to_dataset(item) for item in items if isinstance(item, dict)]
        rows = [row for row in rows if row is not None]
        if query.text:
            terms = query.text.lower().split()
            rows = [
                row
                for row in rows
                if all(term in f"{row.title} {row.description}".lower() for term in terms)
            ]
        total = body.get("total")
        total = int(total) if isinstance(total, int) else None
        next_cursor = (
            str(max(0, offset) + len(items))
            if total is not None and max(0, offset) + len(items) < total
            else None
        )
        return SearchPage(
            datasets=tuple(rows),
            total=total,
            next_cursor=next_cursor,
        )

    def materialize(
        self, dataset: Dataset, asset: Asset | None = None
    ) -> tuple[LayerSpec, ...]:
        chosen = asset or next(
            (item for item in dataset.assets if item.role == "data"), None
        )
        if chosen is None:
            raise ToolInputError(f"dataset {dataset.id!r} has no GeoTIFF asset")
        return (
            LayerSpec(
                kind="raster",
                uri=chosen.href,
                name=dataset.title,
                options={"asset_id": chosen.id},
                metadata={
                    "license": dataset.license,
                    "attribution": dataset.attribution,
                    "source_url": chosen.href,
                },
            ),
        )

    def _item_to_dataset(self, item: dict) -> Dataset | None:
        download = item.get("downloadURL")
        if not isinstance(download, str) or not download.startswith("https://"):
            return None
        title = str(item.get("title") or "USGS DEM")
        source_id = str(item.get("sourceId") or item.get("id") or title)
        bbox_data = item.get("boundingBox")
        bbox = None
        if isinstance(bbox_data, dict):
            try:
                bbox = (
                    float(bbox_data["minX"]),
                    float(bbox_data["minY"]),
                    float(bbox_data["maxX"]),
                    float(bbox_data["maxY"]),
                )
            except (KeyError, TypeError, ValueError):
                bbox = None
        size = item.get("sizeInBytes")
        try:
            size_int = int(size) if size is not None else None
        except (TypeError, ValueError):
            size_int = None
        return Dataset(
            source_id=self.descriptor.id,
            id=source_id,
            title=title,
            kind="raster",
            description=str(item.get("format") or "USGS 3DEP product"),
            bbox=bbox,
            datetime=item.get("publicationDate") or item.get("lastUpdated"),
            attribution="U.S. Geological Survey, 3DEP",
            license="Public domain (US Government)",
            properties={
                "dataset": item.get("dataset"),
                "format": item.get("format"),
                "size": size_int,
                "source_id": item.get("sourceId"),
                "meta_url": item.get("metaUrl"),
                "last_updated": item.get("lastUpdated"),
            },
            assets=(
                Asset(
                    id="download",
                    href=download,
                    media_type="image/tiff",
                    role="data",
                    mode="download",
                    size=size_int,
                ),
            ),
        )


__all__ = ["DEM_DATASETS", "TNM_PRODUCTS", "USGSSource"]
