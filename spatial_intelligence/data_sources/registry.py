"""The data-source registry: adapters plus the enabled/disabled configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..contracts.errors import ToolInputError
from .base import DataSource

#: Display labels for the categories sources declare.
CATEGORY_LABELS: dict[str, str] = {
    "imagery": "Imagery",
    "elevation": "Elevation",
    "authoritative": "Authoritative GIS",
    "boundaries": "Boundaries",
    "urban": "Urban & local data",
    "humanitarian": "Humanitarian",
    "weather": "Weather & climate",
}


class DataSourceRegistry:
    """One ordered set of sources, filtered by the effective configuration."""

    def __init__(
        self,
        sources: Iterable[DataSource],
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self._sources: dict[str, DataSource] = {}
        for source in sources:
            source_id = source.descriptor.id
            if source_id in self._sources:
                raise ValueError(f"duplicate data source id: {source_id!r}")
            self._sources[source_id] = source
        self._disabled: set[str] = set()
        self.configure(config or {})

    def configure(self, config: Mapping[str, Any]) -> None:
        """Apply ``disabled_sources``; unknown names are kept for forward compatibility."""
        disabled = config.get("disabled_sources") if isinstance(config, Mapping) else None
        if isinstance(disabled, str):
            disabled = [disabled]
        self._disabled = {
            str(name).strip()
            for name in (disabled or [])
            if str(name).strip()
        }

    def all(self) -> list[DataSource]:
        """Every registered source, enabled or not."""
        return list(self._sources.values())

    def enabled(self) -> list[DataSource]:
        """Every source the agent and UI may use right now."""
        return [
            source
            for source in self._sources.values()
            if source.descriptor.id not in self._disabled
        ]

    def get(self, source_id: str, *, enabled_only: bool = False) -> DataSource:
        """Return one source, raising a model-visible error when unavailable."""
        key = str(source_id or "").strip()
        source = self._sources.get(key)
        if source is None:
            known = ", ".join(self._sources) or "none"
            raise ToolInputError(
                f"unknown data source {source_id!r}; available: {known}"
            )
        if enabled_only and key in self._disabled:
            raise ToolInputError(
                f"data source {key!r} is disabled; enable it in the data source registry"
            )
        return source

    def is_enabled(self, source_id: str) -> bool:
        """Whether ``source_id`` is registered and enabled."""
        return source_id in self._sources and source_id not in self._disabled

    def categories(self) -> list[dict]:
        """Return the categories the registered sources actually use."""
        seen: list[str] = []
        for source in self._sources.values():
            category = source.descriptor.category
            if category and category not in seen:
                seen.append(category)
        return [
            {"id": category, "label": CATEGORY_LABELS.get(category, category.title())}
            for category in seen
        ]

    def payload(self) -> list[dict]:
        """Return source descriptors annotated with enabled/available state."""
        rows: list[dict] = []
        for source in self._sources.values():
            descriptor = source.descriptor
            status = source.status()
            row = descriptor.as_dict()
            row.update(
                {
                    "enabled": descriptor.id not in self._disabled,
                    "available": status.available,
                    "status": status.as_dict(),
                }
            )
            rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(self._sources)


__all__ = ["CATEGORY_LABELS", "DataSourceRegistry"]
