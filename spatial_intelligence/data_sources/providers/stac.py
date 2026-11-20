"""STAC adapter over the existing Earth Search / Planetary Computer client."""

from __future__ import annotations

from dataclasses import replace

from ...contracts.errors import ToolInputError
from ...geo.catalogs import stac
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


class _MemorySceneCache:
    """The low-level client remembers scenes; this adapter owns the real cache."""

    def remember(self, scene: dict) -> dict:
        return scene

    def get(self, scene_key: str) -> dict:
        return {}


_MEMORY = _MemorySceneCache()


class STACSource(DataSource):
    """Search any STAC API, with the two hosted catalogs named for convenience."""

    descriptor = SourceDescriptor(
        id="stac",
        name="STAC Catalogs",
        summary=(
            "Search Sentinel, Landsat, NAIP, Sentinel-1, HLS, and DEM collections "
            "on Earth Search or Planetary Computer, or any HTTPS STAC API."
        ),
        category="imagery",
        homepage="https://stacspec.org/",
        attribution="Provider catalogs and their collections",
        capabilities=frozenset(
            {
                SourceCapability.SEARCH,
                SourceCapability.LOOKUP,
                SourceCapability.SPATIAL,
                SourceCapability.TEMPORAL,
                SourceCapability.DOWNLOAD,
                SourceCapability.MAP,
            }
        ),
        filters=(
            FilterSpec(
                "catalog",
                "Catalog",
                "select",
                default=stac.DEFAULT_CATALOG,
                choices=tuple((key, catalog.label) for key, catalog in stac.CATALOGS.items()),
                help="Hosted catalog; a custom STAC API URL is not exposed here yet.",
            ),
            FilterSpec(
                "collections",
                "Collections",
                "multiselect",
                help=(
                    "STAC collection ids. Leave empty to list collections; pass ids "
                    "from that list to search scenes."
                ),
                hidden=True,
            ),
            FilterSpec("cloud_max", "Max cloud %", "number", help="0–100; unknown cloud passes."),
            FilterSpec(
                "sort",
                "Sort",
                "select",
                default="date",
                choices=(("date", "Newest first"), ("cloud", "Clearest first")),
            ),
        ),
        docs_url="https://stacspec.org/",
    )

    def __init__(self) -> None:
        self._sas = stac.SasTokenCache()

    def search(self, query: DataQuery) -> SearchPage:
        catalog = self._catalog(query)
        collections = filter_list(query.filters.get("collections"))
        if not collections:
            return self._collections(catalog, query)

        try:
            cloud_max = (
                float(query.filters["cloud_max"])
                if query.filters.get("cloud_max") not in (None, "")
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ToolInputError("cloud_max must be a number between 0 and 100") from exc
        sort = str(query.filters.get("sort") or "date")
        body = stac.search_scenes(
            catalog,
            collections=collections,
            cache=_MEMORY,
            bounds=list(query.bbox) if query.bbox else None,
            start=query.start,
            end=query.end,
            cloud_max=cloud_max,
            sort=sort,
            limit=query.limit,
            sas=self._sas,
            sign=False,
        )
        datasets = tuple(
            self._scene_to_dataset(scene, catalog)
            for scene in body.get("scenes") or []
        )
        return SearchPage(
            datasets=datasets,
            total=body.get("matched"),
            next_cursor=None,
        )

    def present(self, dataset: Dataset) -> Dataset:
        if not dataset.properties.get("signed"):
            return dataset
        collection = str(dataset.properties.get("collection") or "")
        def sign(href: str) -> str:
            return stac.asset_href(
                {
                    "asset_url": href,
                    "signed": True,
                    "collection": collection,
                },
                sas=self._sas,
            )

        assets = tuple(replace(asset, href=sign(asset.href)) for asset in dataset.assets)
        properties = dict(dataset.properties)
        bands = properties.get("bands")
        if isinstance(bands, dict):
            properties["bands"] = {
                name: sign(str(href)) for name, href in bands.items()
            }
        return replace(dataset, assets=assets, properties=properties)

    def materialize(
        self, dataset: Dataset, asset: Asset | None = None
    ) -> tuple[LayerSpec, ...]:
        chosen = asset or next(
            (item for item in dataset.assets if item.role == "data"), None
        )
        if chosen is None:
            raise ToolInputError(f"dataset {dataset.id!r} has no raster asset")
        href = chosen.href
        if dataset.properties.get("signed"):
            href = stac.asset_href(
                {
                    "asset_url": href,
                    "signed": True,
                    "collection": str(dataset.properties.get("collection") or ""),
                },
                sas=self._sas,
            )
        return (
            LayerSpec(
                kind="raster",
                uri=href,
                name=dataset.title,
                options={"asset_id": chosen.id},
                metadata={
                    "license": dataset.license,
                    "attribution": dataset.attribution or "STAC",
                    "source_url": href,
                },
            ),
        )

    # -- internals -------------------------------------------------------

    def _catalog(self, query: DataQuery, parent: str | None = None) -> stac.Catalog:
        reference = (
            parent
            or query.filters.get("catalog")
            or stac.DEFAULT_CATALOG
        )
        try:
            return stac.resolve_catalog(str(reference))
        except ToolInputError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider input
            raise ToolInputError(f"invalid STAC catalog {reference!r}: {exc}") from exc

    def _collections(self, catalog: stac.Catalog, query: DataQuery) -> SearchPage:
        rows = stac.list_collections(catalog.api, query.text, query.limit)
        datasets = tuple(
            Dataset(
                source_id=self.descriptor.id,
                id=f"{catalog.key}:collection:{row['id']}",
                title=str(row.get("title") or row["id"]),
                kind="collection",
                description=str(row.get("description") or ""),
                attribution=catalog.label,
                properties={
                    "catalog": catalog.key,
                    "collection": row["id"],
                    "role": "collection",
                },
            )
            for row in rows
        )
        return SearchPage(datasets=datasets, total=len(datasets), next_cursor=None)

    def _scene_to_dataset(self, scene: dict, catalog: stac.Catalog) -> Dataset:
        bands = scene.get("bands") if isinstance(scene.get("bands"), dict) else {}
        display_name = str(scene.get("asset") or "data")
        display_url = str(scene.get("asset_url") or "")
        assets: list[Asset] = []
        seen: set[str] = set()
        for name, href in ((display_name, display_url), *bands.items()):
            if not href or href in seen:
                continue
            seen.add(href)
            assets.append(
                Asset(
                    id=str(name),
                    href=str(href),
                    media_type="image/tiff",
                    role="data",
                    mode="download",
                )
            )
        item_id = str(scene.get("id") or "scene")
        bbox = scene.get("bbox")
        return Dataset(
            source_id=self.descriptor.id,
            id=f"{catalog.key}:{item_id}",
            title=str(scene.get("title") or item_id),
            kind="raster",
            description=str(scene.get("provider") or ""),
            bbox=tuple(float(value) for value in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
            datetime=scene.get("datetime"),
            thumbnail_url=scene.get("thumbnail_url"),
            attribution=str(scene.get("provider") or catalog.label),
            properties={
                "catalog": catalog.key,
                "collection": scene.get("collection"),
                "item_id": item_id,
                "provider": scene.get("provider"),
                "scene_key": scene.get("scene_key"),
                "asset": display_name,
                "bands": dict(bands),
                "signed": bool(scene.get("signed")),
                "sensor": scene.get("sensor"),
                "cloud_cover": scene.get("cloud_cover"),
                "gsd": scene.get("gsd"),
            },
            assets=tuple(assets),
        )


__all__ = ["STACSource"]
