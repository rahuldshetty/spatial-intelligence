"""Generic data-source tools: one workflow for every registered source.

The provider-specific tools this replaces made the model learn a new name per
catalog.  Here ``list_data_sources`` names the sources, ``describe_data_source``
gives their filters, and ``search_data``/``add_data`` work the same way
everywhere.  A new source needs no new tool.
"""

from __future__ import annotations

from ...contracts.effects import Effect
from ...contracts.errors import ToolInputError
from ...contracts.services import DATA_SOURCES_SERVICE
from ...data_sources import DataQuery, DataService
from ...data_sources.base import validate_bbox
from ...data_sources.registry import DataSourceRegistry
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool


@pack(category="data")
class DataPack:
    """Discover sources, search them, and add their datasets to the map."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._service: DataService | None = None

    @property
    def _registry(self) -> DataSourceRegistry:
        registry = self._rt.services.get(DATA_SOURCES_SERVICE)
        if not isinstance(registry, DataSourceRegistry):
            raise ToolInputError("the data source registry is not available in this session")
        return registry

    @property
    def _data(self) -> DataService:
        if self._service is None:
            self._service = DataService(
                self._rt.workspace,
                self._registry,
                map_obj=lambda: self._rt.require_map(),
                reporter=lambda: self._rt.reporter,
                file_url=self._rt.file_url,
                notify_files=self._rt.events.notify_files,
                notify_map=self._rt.events.notify_map,
                record_artifact=self._rt.record_artifact,
            )
        return self._service

    @tool(effects=frozenset({Effect.READ}))
    def list_data_sources(self, query: str = "", category: str = "") -> list[dict]:
        """List enabled data sources, optionally filtered by text and category.

        ``query`` and ``category`` are optional. Call this before choosing a
        source; ``describe_data_source`` then returns that source's search
        filters and license.
        """
        terms = [term for term in (query or "").lower().split() if term]
        category_name = (category or "").strip().lower()
        rows: list[dict] = []
        for row in self._registry.payload():
            if not row.get("enabled"):
                continue
            if category_name and str(row.get("category") or "").lower() != category_name:
                continue
            haystack = " ".join(
                str(row.get(key) or "")
                for key in ("id", "name", "summary", "category", "attribution")
            ).lower()
            if terms and not all(term in haystack for term in terms):
                continue
            rows.append(row)
        return rows

    @tool(effects=frozenset({Effect.READ}))
    def describe_data_source(self, source_id: str) -> dict:
        """Return one source's description, capabilities, filters, and license.

        Call this before ``search_data`` when the source-specific filter names
        are not obvious. ``source_id`` comes from ``list_data_sources``.
        """
        return self._data.describe(source_id)

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}))
    def search_data(
        self,
        source_id: str,
        text: str = "",
        bbox: list[float] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        """Search one source; all search terms are optional.

        ``text`` may be empty, and ``bbox`` is needed only by sources that
        declare the spatial capability. ``filters`` takes the source-specific
        names from ``describe_data_source``. Results carry a stable ``key``;
        pass its source and dataset id to ``add_data`` or ``download_data``.
        """
        query = DataQuery(
            text=text or "",
            bbox=validate_bbox(bbox),
            start=start_date,
            end=end_date,
            limit=limit,
            cursor=cursor,
            filters=filters or {},
        )
        page = self._data.search(source_id, query)
        return _page_payload(source_id, page)

    @tool(
        effects=frozenset({Effect.NETWORK, Effect.WORKSPACE_WRITE, Effect.MAP_WRITE}),
        kind=ToolKind.REPORTING,
        sequential=True,
    )
    def add_data(
        self,
        source_id: str,
        dataset_id: str,
        asset_id: str | None = None,
        name: str | None = None,
        download: bool = True,
    ) -> dict:
        """Download when needed and add a searched dataset to the live map.

        ``download`` defaults to true for analysis-ready rasters and vectors;
        tile services are added remotely. A dataset carries a stable key from
        ``search_data``, so no URL is retyped.
        """
        return self._data.add(
            source_id,
            dataset_id,
            asset_id=asset_id,
            name=name,
            download=download,
        )

    @tool(
        effects=frozenset({Effect.NETWORK, Effect.WORKSPACE_WRITE}),
        kind=ToolKind.REPORTING,
    )
    def download_data(
        self,
        source_id: str,
        dataset_id: str,
        asset_id: str | None = None,
        filename: str | None = None,
    ) -> dict:
        """Download a searched dataset into ``data/`` without adding a layer.

        Use this when the file is input to a later processing step rather than
        something to show immediately.
        """
        return self._data.download(
            source_id,
            dataset_id,
            asset_id=asset_id,
            filename=filename,
        )


def _page_payload(source_id: str, page) -> dict:
    return {
        "source": source_id,
        "total": page.total,
        "next_cursor": page.next_cursor,
        "datasets": [
            {**dataset.as_dict(), "key": dataset.key} for dataset in page.datasets
        ],
    }


__all__ = ["DataPack"]
