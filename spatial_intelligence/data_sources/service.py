"""Search, download, and map a dataset through the common source contract.

This is the one place that knows about the workspace, the live map, downloads,
and provenance.  Sources stay pure adapters; the agent tools and (later) the
registry modal call this service.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ..contracts.errors import ToolInputError
from ..contracts.progress import NULL_REPORTER, Reporter
from ..geo.catalogs.base import fetch_json
from ..map import document, layers
from ..workspace import Workspace
from ..workspace import files as fileops
from .base import Asset, DataQuery, Dataset, LayerSpec, SearchPage, validate_bbox
from .cache import DatasetCache
from .text import clean_dataset
from .registry import DataSourceRegistry

#: Most datasets one query may return.
MAX_LIMIT = 100
#: Most features one ArcGIS layer download may assemble.
MAX_ARCGIS_FEATURES = 200_000
#: Features requested per ArcGIS page.
ARCGIS_PAGE_SIZE = 2000


class DataService:
    """The common operations shared by the agent tools and the registry UI.

    It receives callables for the live map, reporter, and notifications rather
    than a tool runtime, so the data-source layer stays independent of the tool
    layer and the future registry modal can use it with the app state directly.
    """

    def __init__(
        self,
        workspace: Workspace | None,
        registry: DataSourceRegistry,
        *,
        map_obj: Any | Callable[[], Any] | None = None,
        reporter: Reporter | Callable[[], Reporter] | None = None,
        file_url: Callable[[str], str] | None = None,
        notify_files: Callable[[], None] | None = None,
        notify_map: Callable[[], None] | None = None,
        record_artifact: Callable[[Path], str] | None = None,
    ) -> None:
        self._workspace = workspace
        self._registry = registry
        self._cache = DatasetCache(workspace)
        self._map_source = map_obj
        self._reporter_source = reporter
        self._file_url = file_url
        self._notify_files = notify_files
        self._notify_map = notify_map
        self._record_artifact = record_artifact

    @property
    def registry(self) -> DataSourceRegistry:
        return self._registry

    # -- runtime pieces --------------------------------------------------

    def _map(self) -> Any:
        source = self._map_source
        value = source() if callable(source) else source
        if value is None:
            raise ToolInputError("the live map is not available in this session")
        return value

    def _reporter(self) -> Reporter:
        source = self._reporter_source
        return source() if callable(source) else (source or NULL_REPORTER)

    def _require_workspace(self) -> Workspace:
        if self._workspace is None:
            raise ToolInputError("open a workspace before adding or downloading data")
        return self._workspace

    def _notify(self, hook: Callable[[], None] | None) -> None:
        if hook is not None:
            try:
                hook()
            except Exception:
                # A dead SSE subscriber must never fail a completed download.
                pass

    def _record(self, path: Path) -> None:
        if self._record_artifact is not None:
            self._record_artifact(path)
            return
        workspace = self._workspace
        if workspace is not None:
            workspace.record_output(workspace.relative(path))
            self._notify(self._notify_files)

    # -- search ----------------------------------------------------------

    def search(self, source_id: str, query: DataQuery) -> SearchPage:
        """Search one enabled source and remember the unsigned results."""
        source = self._registry.get(source_id, enabled_only=True)
        normalized = self._normalize_query(query)
        page = source.search(normalized)
        cleaned = tuple(clean_dataset(dataset) for dataset in page.datasets)
        self._cache.remember_all(cleaned)
        return SearchPage(
            datasets=tuple(source.present(dataset) for dataset in cleaned),
            total=page.total,
            next_cursor=page.next_cursor,
        )

    def describe(self, source_id: str) -> dict:
        """Return one source's descriptor plus its current status."""
        source = self._registry.get(source_id)
        descriptor = source.descriptor
        return {
            **descriptor.as_dict(),
            "enabled": self._registry.is_enabled(descriptor.id),
            "status": source.status().as_dict(),
        }

    # -- add / download --------------------------------------------------

    def add(
        self,
        source_id: str,
        dataset_id: str,
        *,
        asset_id: str | None = None,
        name: str | None = None,
        download: bool = True,
    ) -> dict:
        """Resolve, download when needed, and add one dataset to the live map."""
        source, dataset, asset = self._resolve(source_id, dataset_id, asset_id)
        specs = source.materialize(dataset, asset)
        if not specs:
            raise ToolInputError(f"dataset {dataset.id!r} produced no map layers")
        m = self._map()
        created: list[dict] = []
        for index, spec in enumerate(specs):
            existing = self._existing_layer(m, dataset, spec, index)
            if existing is not None:
                created.append(existing)
                continue
            created.append(
                self._add_spec(dataset, asset, spec, index=index, name=name, download=download)
            )
        self._notify(self._notify_map)
        return {
            "status": "added",
            "source": dataset.source_id,
            "dataset": dataset.id,
            "title": dataset.title,
            "license": dataset.license,
            "attribution": dataset.attribution,
            "layers": created,
        }

    def download(
        self,
        source_id: str,
        dataset_id: str,
        *,
        asset_id: str | None = None,
        filename: str | None = None,
    ) -> dict:
        """Download one dataset's assets into ``data/`` without mapping them."""
        workspace = self._require_workspace()
        source, dataset, asset = self._resolve(source_id, dataset_id, asset_id)
        specs = source.materialize(dataset, asset)
        downloaded: list[dict] = []
        for spec in specs:
            if spec.kind in ("tiles", "wmts", "wms"):
                continue
            path = self._download_spec(
                dataset, asset, spec, filename=filename if len(specs) == 1 else None
            )
            downloaded.append(
                {
                    "path": workspace.relative(path),
                    "name": spec.name,
                    "license": dataset.license,
                    "attribution": dataset.attribution,
                }
            )
        if not downloaded:
            raise ToolInputError(
                f"dataset {dataset.id!r} has no downloadable asset; it is a tile service"
            )
        return {
            "status": "downloaded",
            "source": dataset.source_id,
            "dataset": dataset.id,
            "title": dataset.title,
            "files": downloaded,
        }

    # -- internals -------------------------------------------------------

    def _normalize_query(self, query: DataQuery) -> DataQuery:
        return DataQuery(
            text=str(query.text or "").strip(),
            bbox=validate_bbox(query.bbox),
            start=query.start,
            end=query.end,
            limit=max(1, min(int(query.limit or 20), MAX_LIMIT)),
            cursor=query.cursor,
            filters=dict(query.filters or {}),
        )

    def _resolve(
        self, source_id: str, dataset_id: str, asset_id: str | None
    ) -> tuple[Any, Dataset, Asset | None]:
        source = self._registry.get(source_id, enabled_only=True)
        prefix = f"{source.descriptor.id}:"
        if dataset_id.startswith(prefix):
            dataset_id = dataset_id[len(prefix) :]
        key = f"{source.descriptor.id}:{dataset_id}"
        dataset = self._cache.get(key)
        if dataset is not None:
            dataset = clean_dataset(dataset)
        if dataset is None:
            dataset = source.get(dataset_id)
            if dataset is not None:
                dataset = clean_dataset(dataset)
                self._cache.remember(dataset)
        if dataset is None:
            raise ToolInputError(
                f"dataset {dataset_id!r} is not in the current data source cache; "
                "search again"
            )
        asset = None
        if asset_id:
            asset = next((item for item in dataset.assets if item.id == asset_id), None)
            if asset is None:
                known = ", ".join(item.id for item in dataset.assets) or "none"
                raise ToolInputError(
                    f"dataset {dataset_id!r} has no asset {asset_id!r}; available: {known}"
                )
        return source, dataset, asset

    def _existing_layer(
        self, map_obj: Any, dataset: Dataset, spec: LayerSpec, index: int
    ) -> dict | None:
        for layer in layers.layers(map_obj):
            metadata = (layer.get("metadata") or {}).get("geoaiDataSource") or {}
            if (
                metadata.get("source_id") == dataset.source_id
                and metadata.get("dataset_id") == dataset.id
                and metadata.get("layer_index") == index
            ):
                layer_id = layer.get("id")
                if isinstance(layer_id, str) and layer_id:
                    return {
                        "status": "existing",
                        "layer_id": layer_id,
                        "name": layer.get("name"),
                        "layer_index": index,
                    }
        return None

    def _add_spec(
        self,
        dataset: Dataset,
        asset: Asset | None,
        spec: LayerSpec,
        *,
        index: int,
        name: str | None,
        download: bool,
    ) -> dict:
        workspace = self._require_workspace()
        m = self._map()
        layer_name = name or spec.name or dataset.title
        uri = spec.uri
        local_path: str | None = None
        force_download = bool(spec.options.get("force_download"))
        if (download or force_download) and spec.kind in ("raster", "vector") and _is_http(uri):
            path = self._download_spec(dataset, asset, spec)
            local_path = workspace.relative(path)
            uri = local_path

        if spec.kind == "raster":
            layer_id = layers.add_raster(
                workspace, m, uri, layer_name, file_url=self._file_url
            )
        elif spec.kind == "vector":
            if _is_geojson(uri, spec):
                layer_id = layers.add_geojson(workspace, m, uri, layer_name)
            else:
                layer_id = layers.add_vector(workspace, m, uri, layer_name)
        elif spec.kind == "tiles":
            layer_id = layers.add_tile_layer(
                workspace,
                m,
                uri,
                layer_name,
                attribution=dataset.attribution,
            )
        elif spec.kind == "wmts":
            layer_id = layers.add_wmts(
                workspace,
                m,
                uri,
                layer_name,
                tile_size=int(spec.options.get("tile_size") or 256),
            )
        elif spec.kind == "wms":
            layer_id = layers.add_wms(
                workspace,
                m,
                uri,
                str(spec.options.get("layers") or ""),
                layer_name,
                styles=str(spec.options.get("styles") or ""),
            )
        else:
            raise ToolInputError(f"unknown layer kind {spec.kind!r}")

        provenance = {
            "source_id": dataset.source_id,
            "dataset_id": dataset.id,
            "title": dataset.title,
            "asset_id": spec.options.get("asset_id") or (asset.id if asset else None),
            "layer_index": index,
            "kind": spec.kind,
            "source_url": spec.metadata.get("source_url") or spec.uri,
            "license": dataset.license,
            "attribution": dataset.attribution,
            "datetime": dataset.datetime,
            "properties": dict(dataset.properties),
        }
        if local_path is not None:
            provenance["local_path"] = local_path
        layers.set_layer_metadata(m, layer_id, "geoaiDataSource", provenance)
        document.persist_map(m, workspace)
        return {
            "status": "added",
            "layer_id": layer_id,
            "name": layer_name,
            "layer_index": index,
            "local_path": local_path,
        }

    def _download_spec(
        self,
        dataset: Dataset,
        asset: Asset | None,
        spec: LayerSpec,
        *,
        filename: str | None = None,
    ) -> Path:
        if spec.kind in ("tiles", "wmts", "wms"):
            raise ToolInputError("tile services cannot be downloaded as a file")
        arcgis = spec.options.get("arcgis_feature")
        if isinstance(arcgis, dict):
            return self._download_arcgis_features(dataset, spec, arcgis)
        workspace = self._require_workspace()
        uri = spec.uri
        if not _is_http(uri):
            raise ToolInputError(f"asset URL must be http(s): {uri!r}")
        name = filename or self._filename(dataset, asset, spec)
        job = self._reporter().job("download", Path(name).name, unit="bytes")
        with job:
            path = fileops.download_file(
                workspace,
                uri,
                name,
                job=job,
            )
            relative = workspace.relative(path)
            job.done(artifact=relative)
        self._record(path)
        return path

    def _download_arcgis_features(
        self, dataset: Dataset, spec: LayerSpec, arcgis: dict
    ) -> Path:
        """Page an ArcGIS FeatureServer layer and write one GeoJSON file.

        ArcGIS caps a single query response (commonly 2,000 features) and signals
        the cap with ``exceededTransferLimit``.  Reading one page and calling it
        the layer silently drops most of a parcel or incident dataset, so page
        until the service says it is complete.
        """
        service_url = str(arcgis.get("service_url") or "").rstrip("/")
        layer_id = int(arcgis.get("layer_id") or 0)
        if not service_url.startswith("https://"):
            raise ToolInputError("ArcGIS feature services must use HTTPS")
        features: list[dict] = []
        offset = 0
        while len(features) < MAX_ARCGIS_FEATURES:
            url = (
                f"{service_url}/{layer_id}/query?where=1%3D1&outFields=*"
                "&returnGeometry=true&outSR=4326&f=geojson"
                f"&resultOffset={offset}&resultRecordCount={ARCGIS_PAGE_SIZE}"
            )
            body = fetch_json(url)
            page = body.get("features")
            if not isinstance(page, list):
                break
            features.extend(item for item in page if isinstance(item, dict))
            if not body.get("exceededTransferLimit") or not page:
                break
            offset += len(page)
        if not features:
            raise ToolInputError(f"ArcGIS layer {dataset.id!r} returned no features")
        workspace = self._require_workspace()
        name = self._filename(dataset, None, spec)
        path = workspace.resolve_under(workspace.data, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"type": "FeatureCollection", "features": features}
        path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._record(path)
        return path

    def _filename(
        self, dataset: Dataset, asset: Asset | None, spec: LayerSpec
    ) -> str:
        base = f"{dataset.source_id}-{dataset.id}"
        if asset is not None:
            base = f"{base}-{asset.id}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", base).strip(".-") or "dataset"
        suffix = _suffix(dataset, asset, spec)
        return f"{safe[:120]}{suffix}"


def _is_http(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _is_geojson(uri: str, spec: LayerSpec) -> bool:
    path = urlparse(uri).path.lower()
    if path.endswith((".geojson", ".json")):
        return True
    media_type = str(spec.options.get("media_type") or "").lower()
    return "geo+json" in media_type or "json" == media_type


def _suffix(dataset: Dataset, asset: Asset | None, spec: LayerSpec) -> str:
    if spec.kind == "vector":
        if spec.options.get("arcgis_feature"):
            return ".geojson"
        media_type = str(
            spec.options.get("media_type") or (asset.media_type if asset else "") or ""
        ).lower()
        if "json" in media_type:
            return ".geojson"
        if "gpkg" in media_type or "geopackage" in media_type:
            return ".gpkg"
        if "zip" in media_type:
            return ".zip"
        if "csv" in media_type:
            return ".csv"
    if spec.kind == "raster":
        suffix = Path(unquote(urlparse(spec.uri).path)).suffix.lower()
        if suffix in {".tif", ".tiff", ".img", ".vrt", ".asc", ".nc", ".jp2"}:
            return suffix
        return ".tif"
    suffix = Path(unquote(urlparse(spec.uri).path)).suffix
    return suffix if suffix and len(suffix) <= 8 else ".dat"


__all__ = ["DataService", "MAX_LIMIT"]
