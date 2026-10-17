"""Capability discovery: the compact index of what this build can actually do."""

from __future__ import annotations

import inspect
from typing import Any

from pydantic_ai import ToolReturn

from ... import discovery
from ...contracts.effects import Effect
from ..runtime import ToolRuntime
from ..spec import pack, tool

#: Service key under which the session publishes its tool registry.
REGISTRY_SERVICE = "tools.registry"


def _type_name(annotation: Any) -> str | None:
    """Render a parameter annotation as a short, model-friendly name."""
    if annotation is inspect.Parameter.empty:
        return None
    name = getattr(annotation, "__name__", None)
    return name if name else str(annotation)


@pack(category="capability", effects=frozenset({Effect.READ}))
class CapabilityPack:
    """Capability discovery and tool introspection, backed by the registry."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime

    @tool(core=True)
    def discover_capabilities(self, goal: str, limit: int = 5) -> list[dict]:
        """Find capabilities relevant to a task, and make their tools callable.

        Use this before improvising with ``run_python`` or claiming an integration
        is unavailable. Results identify executable backend tools and GeoLibre
        plugins that currently require an interactive handoff. Every backend tool
        named in the result is revealed, so it can be called without a separate
        ``search_tools`` round-trip.
        """
        found = discovery.discover(self._registry(), goal, limit)
        # A conditional capability (vision) is only useful when its tool is
        # actually registered; drop empty available entries from the answer.
        found = [
            item
            for item in found
            if item.get("tools")
            or item.get("geolibre_plugins")
            or item.get("status") != "available"
        ]
        names = [
            tool["name"]
            for item in found
            for tool in item.get("tools", [])
            if tool.get("name")
        ]
        return ToolReturn(found, tools=names)

    @tool(core=True)
    def describe_tool(self, name: str = "") -> dict:
        """Inspect any registered tool: what it does and how to call it.

        Pass a tool name to get its full description, parameters, and effects;
        with no name (or an unknown name) return the directory of every tool
        with a one-line summary. Use this before calling a tool whose arguments
        or side effects you are unsure about. The named tool is made available
        to call as part of the answer.
        """
        registry = self._registry()
        if registry is None:
            return {"tools": []}
        if not name:
            return {"tools": self._directory(registry)}
        try:
            spec = registry.get(name)
        except KeyError:
            return {"name": name, "known": False, "tools": self._directory(registry)}
        return ToolReturn(self._describe(spec), tools=[spec.name])

    @staticmethod
    def _directory(registry: Any) -> list[dict]:
        return [
            {"name": spec.name, "summary": spec.summary, "category": spec.category}
            for spec in registry.specs()
        ]

    @staticmethod
    def _describe(spec: Any) -> dict:
        params: list[dict] = []
        if spec.fn is not None:
            for param in inspect.signature(spec.fn).parameters.values():
                params.append(
                    {
                        "name": param.name,
                        "type": _type_name(param.annotation),
                        "default": None if param.default is inspect.Parameter.empty else repr(param.default),
                        "required": param.default is inspect.Parameter.empty,
                    }
                )
        return {
            "name": spec.name,
            "summary": spec.summary,
            "description": inspect.getdoc(spec.fn) if spec.fn is not None else None,
            "parameters": params,
            "category": spec.category,
            "effects": sorted(effect.value for effect in spec.effects),
            "kind": spec.kind.value,
        }

    def _registry(self) -> Any | None:
        """Return the registry the session published, if it has done so yet.

        The pack is constructed with the runtime alone, so the registry -- which
        is assembled by the session, after the packs it describes -- arrives
        through the service bag under ``'tools.registry'``. Reading (never
        creating) it keeps a session that has not wired one up working: the
        catalog is still ranked, just without tool summaries.
        """
        return self._rt.services.get(REGISTRY_SERVICE)


__all__ = ["CapabilityPack"]
