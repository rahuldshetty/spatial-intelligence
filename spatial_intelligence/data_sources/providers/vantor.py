"""Vantor Open Data adapter over the existing disaster-event client."""

from __future__ import annotations

from ...contracts.errors import ToolInputError
from ...geo.catalogs import vantor
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
)


class _MemorySceneCache:
    def remember(self, scene: dict) -> dict:
        return scene

    def get(self, scene_key: str) -> dict:
        return {}


_MEMORY = _MemorySceneCache()


class VantorSource(DataSource):
    """Disaster-event imagery; an empty search lists events, an event id lists scenes."""

    descriptor = SourceDescriptor(
        id="vantor",
        name="Vantor Open Data",
        summary=(
            "Search disaster events and their pre/post-event satellite imagery "
            "as Cloud-Optimized GeoTIFFs."
        ),
        category="imagery",
        homepage="https://www.vantor.com/",
        attribution="Vantor Open Data",
        capabilities=frozenset(
            {
                SourceCapability.SEARCH,
                SourceCapability.LOOKUP,
                SourceCapability.SPATIAL,
                SourceCapability.DOWNLOAD,
                SourceCapability.MAP,
            }
        ),
        filters=(
            FilterSpec(
                "event_id",
                "Event",
                "text",
                help="Leave empty to list events; pass an event id to list its scenes.",
                hidden=True,
            ),
            FilterSpec(
                "phase",
                "Phase",
                "select",
                default="all",
                choices=(("all", "All"), ("pre", "Pre-event"), ("post", "Post-event")),
            ),
        ),
        docs_url="https://www.vantor.com/",
    )

    def search(self, query: DataQuery) -> SearchPage:
        event_id = str(query.filters.get("event_id") or "").strip()
        if not event_id:
            return self._events(query)
        phase = str(query.filters.get("phase") or "all")
        scenes = vantor.search_vantor_imagery(
            event_id,
            bounds=list(query.bbox) if query.bbox else None,
            phase=phase,
            limit=query.limit,
            cache=_MEMORY,
        )
        datasets = tuple(self._scene_to_dataset(scene) for scene in scenes)
        return SearchPage(datasets=datasets, total=len(datasets), next_cursor=None)

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
                    "attribution": dataset.attribution or "Vantor Open Data",
                    "source_url": chosen.href,
                },
            ),
        )

    # -- internals -------------------------------------------------------

    def _events(self, query: DataQuery) -> SearchPage:
        events = vantor.search_vantor_events(query.text, query.limit)
        datasets = tuple(
            Dataset(
                source_id=self.descriptor.id,
                id=str(event["id"]),
                title=str(event.get("title") or event["id"]),
                kind="collection",
                description="Vantor Open Data disaster event",
                attribution="Vantor Open Data",
                properties={"event_id": event["id"], "role": "event"},
            )
            for event in events
        )
        return SearchPage(datasets=datasets, total=len(datasets), next_cursor=None)

    def _scene_to_dataset(self, scene: dict) -> Dataset:
        item_id = str(scene.get("id") or "scene")
        bbox = scene.get("bbox")
        return Dataset(
            source_id=self.descriptor.id,
            id=item_id,
            title=str(scene.get("title") or item_id),
            kind="raster",
            description="Vantor Open Data scene",
            bbox=tuple(float(value) for value in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
            datetime=scene.get("datetime"),
            thumbnail_url=scene.get("thumbnail_url"),
            attribution="Vantor Open Data",
            properties={
                "scene_key": scene.get("scene_key"),
                "event_id": scene.get("event"),
                "phase": scene.get("phase"),
                "sensor": scene.get("sensor"),
                "cloud_cover": scene.get("cloud_cover"),
                "gsd": scene.get("gsd"),
            },
            assets=(
                Asset(
                    id="visual",
                    href=str(scene.get("asset_url") or ""),
                    media_type="image/tiff",
                    role="data",
                    mode="download",
                ),
            ),
        )


__all__ = ["VantorSource"]
