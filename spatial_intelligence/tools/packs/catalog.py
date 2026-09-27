"""Open imagery catalog tools: Vantor, OpenAerialMap, and STAC (Sentinel/Landsat/DEM).

Every provider normalizes its results into the same scene shape and hands them
to one :class:`SceneCache`, so the download and map tools take a ``scene_key``
and never need to know which catalog produced it.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from ...contracts.effects import Effect
from ...contracts.errors import ToolInputError
from ...geo.catalogs import openaerialmap, stac, vantor
from ...geo.catalogs.scenes import SceneCache
from ...map import document, layers
from ...map.layers import LOCAL_SOURCE_KEY
from ...workspace import files as fileops
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool


@pack(category="catalog")
class CatalogPack:
    """Read-only open imagery catalogs plus scene downloads.

    Search results are remembered by one :class:`SceneCache` per session, which
    is what lets ``download_catalog_scene`` and ``add_catalog_scene`` take a
    ``scene_key`` instead of re-running the search or trusting a URL the model
    retyped.
    """

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._scenes = SceneCache(runtime.workspace)
        #: Signed-catalog tokens for this session. Scenes hold the catalog's own
        #: hrefs and are signed when used, so no token reaches the scene cache.
        self._sas = stac.SasTokenCache()

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}))
    def search_vantor_events(self, query: str = "", limit: int = 20) -> list[dict]:
        """Search Vantor Open Data disaster events by title.

        This uses the Vantor STAC catalog consumed by GeoLibre's Vantor Open Data
        plugin. Call ``search_vantor_imagery`` with one returned event id.
        """
        return vantor.search_vantor_events(query, limit)

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}))
    def search_vantor_imagery(
        self,
        event_id: str,
        bounds: list[float] | None = None,
        phase: str = "all",
        limit: int = 30,
    ) -> list[dict]:
        """List renderable Vantor scenes for an event, optionally filtered by AOI.

        ``bounds`` is ``[west, south, east, north]`` and ``phase`` is ``all``,
        ``pre``, or ``post``. Results contain stable scene keys, thumbnails, dates,
        resolution, cloud cover, and source COG URLs.
        """
        return vantor.search_vantor_imagery(
            event_id, bounds, phase, limit, cache=self._scenes
        )

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}))
    def search_openaerialmap(
        self, bounds: list[float], limit: int = 20, page: int = 1
    ) -> dict:
        """Search OpenAerialMap using the same metadata and TiTiler contract as GeoLibre."""
        return openaerialmap.search_openaerialmap(bounds, limit, page, cache=self._scenes)

    # -- STAC: Sentinel, Landsat, NAIP, DEM, and any other STAC API ---------

    @tool(effects=frozenset({Effect.READ}))
    def list_stac_catalogs(self) -> list[dict]:
        """List the STAC catalogs ``search_stac_scenes`` can reach, and their common collections.

        Use this to pick a ``catalog``. Earth Search serves public AWS
        Cloud-Optimized GeoTIFFs with no credentials; Planetary Computer serves
        Azure assets that this build signs before download. Any other STAC API
        can be passed to the search tools as an https:// URL.
        """
        return stac.list_catalogs()

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}))
    def search_stac_collections(
        self, query: str = "", catalog: str = stac.DEFAULT_CATALOG, limit: int = 50
    ) -> list[dict]:
        """Search a STAC catalog's collections to find the ids scenes are searched in.

        Call this first when the collection id is not known: "sentinel-2-l2a",
        "landsat-c2-l2", "naip", "sentinel-1-grd", and "cop-dem-glo-30" on the
        default Earth Search catalog, for example. Results are ranked by how many
        query terms they match, not filtered to an exact phrase.
        """
        return stac.list_collections(stac.resolve_catalog(catalog).api, query, limit)

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}))
    def search_stac_scenes(
        self,
        collections: list[str],
        catalog: str = stac.DEFAULT_CATALOG,
        bounds: list[float] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        cloud_max: float | None = None,
        sort: str = "date",
        limit: int = 20,
    ) -> dict:
        """Search imagery and elevation scenes in one or more STAC collections.

        Returns ``scenes`` (and the provider's own ``matched`` count, when it
        reports one) with a stable ``scene_key`` per scene; pass that key to
        ``add_catalog_scene`` to download the source COG into ``data/`` and show
        the local copy on the map, or to ``download_catalog_scene`` to fetch it
        without mapping it. Both dates are ``YYYY-MM-DD``; omit them for static
        collections, whose datetime is the dataset's own, so a recent window
        matches nothing. ``cloud_max`` drops scenes with more cloud than that
        (scenes with no cloud property are kept). ``sort`` is ``date`` (newest
        first) or ``cloud`` (clearest first, scenes without a cloud property
        last). Several pages are read, so sorting considers more scenes than
        ``limit``.

        On ``earth-search``, Landsat Collection 2 and NAIP assets live in
        requester-pays AWS buckets that cannot be downloaded anonymously: search
        those collections on ``planetary-computer``, whose assets are signed.
        """
        return stac.search_scenes(
            stac.resolve_catalog(catalog),
            collections=list(collections),
            cache=self._scenes,
            bounds=bounds,
            start=start_date,
            end=end_date,
            cloud_max=cloud_max,
            sort=sort,
            limit=limit,
            sas=self._sas,
        )

    @tool(
        effects=frozenset({Effect.WORKSPACE_WRITE, Effect.NETWORK}),
        kind=ToolKind.REPORTING,
    )
    def download_catalog_scene(self, scene_key: str, filename: str | None = None) -> str:
        """Download a previously searched scene into ``data/`` for local analysis.

        Returns the workspace-relative path of the downloaded asset.
        """
        scene = self._scenes.get(scene_key)
        if not scene:
            raise ToolInputError(
                "scene is not in the current catalog search cache; search again"
            )
        if not filename:
            suffix = Path(urlparse(scene["asset_url"]).path).suffix or ".tif"
            filename = f"{scene.get('id') or 'catalog-scene'}{suffix}"
        job = self._rt.reporter.job("download", Path(filename).name, unit="bytes")
        with job:
            # Signed here rather than when the scene was found: a signed catalog's
            # token lasts about a day, and a cached scene must still download after
            # that (or after a restart).
            path = fileops.download_file(
                self._rt.workspace,
                stac.asset_href(scene, sas=self._sas),
                filename,
                job=job,
            )
            relative = self._rt.workspace.relative(path)
            job.done(artifact=relative)
        self._rt.record_artifact(path)
        return relative

    @tool(effects=frozenset({Effect.WORKSPACE_WRITE, Effect.MAP_WRITE, Effect.NETWORK}))
    def add_catalog_scene(self, scene_key: str, name: str | None = None) -> dict:
        """Download a searched scene into ``data/`` and add the local copy to the map.

        The download is intentionally completed before the map layer is created.
        This keeps the saved project independent of the catalog's remote COG or
        TiTiler service and gives the user a progress cell while the asset arrives.
        Repeated calls return the existing layer instead of adding a duplicate.
        """
        scene = self._scenes.get(scene_key)
        if not scene:
            raise ToolInputError(
                "scene is not in the current catalog search cache; search again"
            )
        workspace = self._rt.workspace
        m = self._rt.require_map()
        layer_name = name or scene.get("title") or scene.get("id") or "Catalog scene"
        suffix = Path(urlparse(scene["asset_url"]).path).suffix or ".tif"
        provider = re.sub(r"[^A-Za-z0-9_-]+", "-", str(scene.get("provider") or "catalog"))
        item_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(scene.get("id") or "scene"))
        key_suffix = re.sub(r"[^A-Za-z0-9]+", "", scene_key.rsplit(":", 1)[-1])[:12]
        filename = f"{provider.lower()}-{item_id}-{key_suffix}{suffix}"
        existing = workspace.resolve_under(workspace.data, filename)
        rel = existing.relative_to(workspace.root).as_posix()
        for layer in layers.layers(m):
            metadata = layer.get("metadata") or {}
            catalog = metadata.get("geoaiCatalog") or {}
            if (
                catalog.get("scene_key") == scene_key
                or catalog.get("local_path") == rel
                or metadata.get(LOCAL_SOURCE_KEY) == rel
            ):
                layer_id = layer.get("id")
                if isinstance(layer_id, str) and layer_id:
                    return {
                        "status": "existing",
                        "layer_id": layer_id,
                        "name": layer.get("name"),
                        "local_path": rel,
                    }
        downloaded = (
            str(existing)
            if existing.is_file() and existing.stat().st_size > 0
            else self.download_catalog_scene(scene_key, filename)
        )
        rel = (
            Path(downloaded).relative_to(workspace.root).as_posix()
            if Path(downloaded).is_absolute()
            else downloaded.replace("\\", "/")
        )
        workspace.resolve(rel, must_exist=True)
        layer_id = layers.add_raster(
            workspace, m, rel, layer_name, file_url=self._rt.file_url
        )
        catalog_metadata = {
            key: scene.get(key)
            for key in (
                "scene_key",
                "id",
                "provider",
                "catalog",
                "collection",
                "event",
                "datetime",
                "phase",
                "sensor",
                "gsd",
                "cloud_cover",
                "asset",
                "asset_url",
                "local_path",
            )
            if scene.get(key) is not None
        }
        catalog_metadata["local_path"] = rel
        layers.set_layer_metadata(m, layer_id, "geoaiCatalog", catalog_metadata)
        document.persist_map(m, workspace)
        self._rt.events.notify_map()
        return {
            "status": "added",
            "layer_id": layer_id,
            "name": layer_name,
            "local_path": rel,
        }


__all__ = ["CatalogPack"]
