"""The data-source contract every provider implements.

A source is a small adapter over one public catalog or service.  It turns the
provider's own response into :class:`Dataset` records and, only when the
provider needs custom map semantics, into :class:`LayerSpec` records.  It never
writes to the workspace or the live map; :mod:`spatial_intelligence.data_sources.service`
owns downloading, provenance, and layer creation.

That split is what keeps a new source to one file: the provider knows its API,
the harness knows the workspace and map.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping

from ..contracts.errors import ToolInputError


class SourceCapability(StrEnum):
    """What a source can do; the UI and the generic tools read these flags."""

    SEARCH = "search"
    LOOKUP = "lookup"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    DOWNLOAD = "download"
    MAP = "map"


@dataclass(frozen=True)
class FilterSpec:
    """One source-specific search control, rendered by the UI and passed by name."""

    name: str
    label: str
    kind: Literal[
        "text", "number", "date", "daterange", "bbox", "select", "multiselect", "boolean"
    ] = "text"
    required: bool = False
    default: Any = None
    choices: tuple[tuple[str, str], ...] = ()
    help: str = ""
    #: Hidden filters are still passed by the API and used by drill-down, but
    #: the registry form does not show a control for them.
    hidden: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "required": self.required,
            "default": self.default,
            "choices": [{"value": value, "label": label} for value, label in self.choices],
            "help": self.help,
            "hidden": self.hidden,
        }


@dataclass(frozen=True)
class SourceDescriptor:
    """Stable source metadata: identity, capabilities, and search controls."""

    id: str
    name: str
    summary: str
    category: str
    homepage: str = ""
    license: str | None = None
    attribution: str | None = None
    capabilities: frozenset[SourceCapability] = frozenset()
    filters: tuple[FilterSpec, ...] = ()
    requires_key: bool = False
    enabled_by_default: bool = True
    docs_url: str | None = None
    #: How the registry UI treats an area-of-interest field: hidden, optional,
    #: or required. Providers that cannot search without one declare required.
    bbox: Literal["hidden", "optional", "required"] = "hidden"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "summary": self.summary,
            "category": self.category,
            "homepage": self.homepage,
            "license": self.license,
            "attribution": self.attribution,
            "capabilities": sorted(capability.value for capability in self.capabilities),
            "filters": [spec.as_dict() for spec in self.filters],
            "requires_key": self.requires_key,
            "enabled_by_default": self.enabled_by_default,
            "docs_url": self.docs_url,
            "bbox": self.bbox,
        }


@dataclass(frozen=True)
class DataQuery:
    """One normalized query. ``text`` is always optional."""

    text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    start: str | None = None
    end: str | None = None
    limit: int = 20
    cursor: str | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "start": self.start,
            "end": self.end,
            "limit": self.limit,
            "cursor": self.cursor,
            "filters": dict(self.filters),
        }


@dataclass(frozen=True)
class Asset:
    """One downloadable or renderable asset of a dataset."""

    id: str
    href: str
    media_type: str | None = None
    role: Literal["data", "visual", "thumbnail"] = "data"
    mode: Literal["download", "remote", "tile"] = "download"
    size: int | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "href": self.href,
            "media_type": self.media_type,
            "role": self.role,
            "mode": self.mode,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Asset":
        return cls(
            id=str(data.get("id") or "asset"),
            href=str(data.get("href") or ""),
            media_type=data.get("media_type"),
            role=data.get("role") or "data",
            mode=data.get("mode") or "download",
            size=data.get("size"),
        )


@dataclass(frozen=True)
class Dataset:
    """A provider-neutral dataset record; the cache stores exactly this shape."""

    source_id: str
    id: str
    title: str
    kind: Literal["raster", "vector", "tiles", "table", "collection"]
    description: str = ""
    bbox: tuple[float, float, float, float] | None = None
    datetime: str | None = None
    thumbnail_url: str | None = None
    license: str | None = None
    attribution: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)
    assets: tuple[Asset, ...] = ()

    @property
    def key(self) -> str:
        """Stable cache key within one workspace."""
        return f"{self.source_id}:{self.id}"

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "description": self.description,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "datetime": self.datetime,
            "thumbnail_url": self.thumbnail_url,
            "license": self.license,
            "attribution": self.attribution,
            "properties": dict(self.properties),
            "assets": [asset.as_dict() for asset in self.assets],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Dataset":
        bbox = data.get("bbox")
        return cls(
            source_id=str(data.get("source_id") or ""),
            id=str(data.get("id") or ""),
            title=str(data.get("title") or data.get("id") or "Dataset"),
            kind=data.get("kind") or "table",
            description=str(data.get("description") or ""),
            bbox=tuple(float(value) for value in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
            datetime=data.get("datetime"),
            thumbnail_url=data.get("thumbnail_url"),
            license=data.get("license"),
            attribution=data.get("attribution"),
            properties=dict(data.get("properties") or {}),
            assets=tuple(
                Asset.from_dict(asset)
                for asset in data.get("assets") or []
                if isinstance(asset, Mapping)
            ),
        )


@dataclass(frozen=True)
class SearchPage:
    """One page of normalized results."""

    datasets: tuple[Dataset, ...]
    total: int | None = None
    next_cursor: str | None = None

    def as_dict(self) -> dict:
        return {
            "datasets": [dataset.as_dict() for dataset in self.datasets],
            "total": self.total,
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True)
class LayerSpec:
    """A map layer a source wants created from a dataset."""

    kind: Literal["raster", "vector", "tiles", "wmts", "wms"]
    uri: str
    name: str
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceStatus:
    """Whether a source can currently be used, and why not."""

    available: bool = True
    reason: str = ""

    def as_dict(self) -> dict:
        return {"available": self.available, "reason": self.reason}


class UnsupportedCapability(ToolInputError):
    """Raised when a caller asks for an operation the source did not declare."""

    def __init__(self, source_id: str, capability: SourceCapability | str) -> None:
        value = capability.value if isinstance(capability, SourceCapability) else capability
        super().__init__(f"data source {source_id!r} does not support {value}")


def validate_bbox(bounds: Any) -> tuple[float, float, float, float] | None:
    """Return a clean WGS84 bbox, or ``None``; reject malformed values."""
    if bounds is None:
        return None
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ToolInputError("bbox must be [west, south, east, north]")
    try:
        west, south, east, north = (float(value) for value in bounds)
    except (TypeError, ValueError) as exc:
        raise ToolInputError("bbox must contain four numbers") from exc
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ToolInputError("bbox must be [west, south, east, north] in WGS84")
    return (west, south, east, north)


def filter_list(value: Any) -> list[str]:
    """Normalize a filter that may be one string, a list, or absent."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


class DataSource(ABC):
    """One public catalog or service.

    Only :meth:`search` is mandatory.  ``get``, ``present``, and
    ``materialize`` have defaults that cover the common cases; a provider
    overrides only the parts its API or map semantics require.
    """

    descriptor: SourceDescriptor

    @abstractmethod
    def search(self, query: DataQuery) -> SearchPage:
        """Return one page of datasets matching ``query`` (text may be empty)."""

    def get(self, dataset_id: str) -> Dataset | None:
        """Return one dataset by id when the provider can resolve it cheaply."""
        return None

    def present(self, dataset: Dataset) -> Dataset:
        """Return the model/UI-facing copy, signing short-lived URLs if needed.

        The cache stores the dataset this method receives, never the signed
        copy, so credentials do not reach disk.
        """
        return dataset

    def materialize(
        self, dataset: Dataset, asset: Asset | None = None
    ) -> tuple[LayerSpec, ...]:
        """Turn a dataset into one or more map layers.

        Sources that do not override this get the obvious mapping: raster and
        vector assets become local or remote layers, tile assets become XYZ
        layers.  Providers with multi-layer services or date-templated tiles
        override it.
        """
        chosen = asset or _first_data_asset(dataset)
        if chosen is None:
            raise ToolInputError(
                f"dataset {dataset.id!r} has no asset to map"
            )
        if dataset.kind == "collection" or dataset.kind == "table":
            raise UnsupportedCapability(self.descriptor.id, SourceCapability.MAP)
        kind = "tiles" if chosen.mode == "tile" else dataset.kind
        if kind not in ("raster", "vector", "tiles"):
            raise UnsupportedCapability(self.descriptor.id, SourceCapability.MAP)
        metadata = {
            "license": dataset.license,
            "attribution": dataset.attribution,
            "source_url": chosen.href,
        }
        return (
            LayerSpec(
                kind=kind,
                uri=chosen.href,
                name=dataset.title,
                options={"asset_id": chosen.id},
                metadata=metadata,
            ),
        )

    def status(self) -> SourceStatus:
        """Return availability; keyed sources override this after checking env."""
        return SourceStatus()


def _first_data_asset(dataset: Dataset) -> Asset | None:
    """Return the first data asset, falling back to any non-thumbnail asset."""
    for asset in dataset.assets:
        if asset.role == "data":
            return asset
    for asset in dataset.assets:
        if asset.role != "thumbnail":
            return asset
    return None


__all__ = [
    "Asset",
    "DataQuery",
    "Dataset",
    "DataSource",
    "FilterSpec",
    "LayerSpec",
    "SearchPage",
    "SourceCapability",
    "SourceDescriptor",
    "SourceStatus",
    "UnsupportedCapability",
    "filter_list",
    "validate_bbox",
]
