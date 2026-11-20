"""OpenAerialMap adapter over the existing metadata client."""

from __future__ import annotations

from ...contracts.errors import ToolInputError
from ...geo.catalogs import openaerialmap
from ..base import (
    Asset,
    DataQuery,
    Dataset,
    DataSource,
    LayerSpec,
    SearchPage,
    SourceCapability,
    SourceDescriptor,
)


class _MemorySceneCache:
    def remember(self, scene: dict) -> dict:
        return scene

    def get(self, scene_key: str) -> dict:
        return {}


_MEMORY = _MemorySceneCache()


class OpenAerialMapSource(DataSource):
    """Openly licensed drone and aerial imagery; search needs a bbox."""

    descriptor = SourceDescriptor(
        id="openaerialmap",
        name="OpenAerialMap",
        summary="Openly licensed drone and aerial imagery as Cloud-Optimized GeoTIFFs.",
        category="imagery",
        homepage="https://openaerialmap.org/",
        attribution="OpenAerialMap contributors",
        capabilities=frozenset(
            {
                SourceCapability.SEARCH,
                SourceCapability.SPATIAL,
                SourceCapability.DOWNLOAD,
                SourceCapability.MAP,
            }
        ),
        filters=(),
        docs_url="https://openaerialmap.org/",
        bbox="required",
    )

    def search(self, query: DataQuery) -> SearchPage:
        if query.bbox is None:
            raise ToolInputError(
                "OpenAerialMap needs an area of interest: pass bbox "
                "[west, south, east, north]"
            )
        try:
            page = int(query.cursor or "1")
        except (TypeError, ValueError) as exc:
            raise ToolInputError("OpenAerialMap cursor must be a page number") from exc
        body = openaerialmap.search_openaerialmap(
            list(query.bbox),
            limit=query.limit,
            page=max(1, page),
            cache=_MEMORY,
        )
        scenes = body.get("scenes") or []
        if query.text:
            terms = query.text.lower().split()
            scenes = [
                scene
                for scene in scenes
                if all(
                    term
                    in f"{scene.get('title', '')} {scene.get('provider', '')}".lower()
                    for term in terms
                )
            ]
        datasets = tuple(self._scene_to_dataset(scene) for scene in scenes)
        next_cursor = str(page + 1) if len(scenes) >= query.limit else None
        return SearchPage(
            datasets=datasets,
            total=body.get("found"),
            next_cursor=next_cursor,
        )

    def materialize(
        self, dataset: Dataset, asset: Asset | None = None
    ) -> tuple[LayerSpec, ...]:
        chosen = asset or next(
            (item for item in dataset.assets if item.role == "data"), None
        )
        if chosen is None:
            raise ToolInputError(f"dataset {dataset.id!r} has no COG asset")
        return (
            LayerSpec(
                kind="raster",
                uri=chosen.href,
                name=dataset.title,
                options={"asset_id": chosen.id},
                metadata={
                    "license": dataset.license,
                    "attribution": dataset.attribution or "OpenAerialMap",
                    "source_url": chosen.href,
                },
            ),
        )

    def _scene_to_dataset(self, scene: dict) -> Dataset:
        item_id = str(scene.get("id") or "scene")
        bbox = scene.get("bbox")
        return Dataset(
            source_id=self.descriptor.id,
            id=item_id,
            title=str(scene.get("title") or item_id),
            kind="raster",
            description="OpenAerialMap image",
            bbox=tuple(float(value) for value in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
            datetime=scene.get("datetime"),
            thumbnail_url=scene.get("thumbnail_url"),
            attribution=str(scene.get("provider") or "OpenAerialMap"),
            properties={
                "scene_key": scene.get("scene_key"),
                "provider": scene.get("provider"),
                "platform": scene.get("platform"),
                "gsd": scene.get("gsd"),
                "tile_url": scene.get("tile_url"),
            },
            assets=(
                Asset(
                    id="cog",
                    href=str(scene.get("asset_url") or ""),
                    media_type="image/tiff",
                    role="data",
                    mode="download",
                ),
            ),
        )


__all__ = ["OpenAerialMapSource"]
