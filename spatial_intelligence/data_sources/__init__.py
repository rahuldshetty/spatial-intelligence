"""The data-source registry and the built-in source adapters."""

from __future__ import annotations

from .base import (
    Asset,
    DataQuery,
    Dataset,
    DataSource,
    FilterSpec,
    LayerSpec,
    SearchPage,
    SourceCapability,
    SourceDescriptor,
    SourceStatus,
    UnsupportedCapability,
)
from .providers import (
    ArcGISSource,
    GIBSSource,
    OpenAerialMapSource,
    STACSource,
    USGSSource,
    VantorSource,
)
from .registry import DataSourceRegistry
from .service import DataService


def default_sources() -> list[DataSource]:
    """Return the built-in sources in discovery order.

    A new source is one adapter module plus one line here.  The generic agent
    tools, the registry payload, and the modal all pick it up automatically.
    """
    return [
        STACSource(),
        ArcGISSource(),
        GIBSSource(),
        USGSSource(),
        VantorSource(),
        OpenAerialMapSource(),
    ]


__all__ = [
    "Asset",
    "ArcGISSource",
    "DataQuery",
    "DataService",
    "Dataset",
    "DataSource",
    "DataSourceRegistry",
    "FilterSpec",
    "GIBSSource",
    "LayerSpec",
    "OpenAerialMapSource",
    "STACSource",
    "SearchPage",
    "SourceCapability",
    "SourceDescriptor",
    "SourceStatus",
    "USGSSource",
    "UnsupportedCapability",
    "VantorSource",
    "default_sources",
]
