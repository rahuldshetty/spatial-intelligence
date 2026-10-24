"""Tool Configuration: what the agent may use, and what run_python may import.

The configuration is small and explicit: three lists of names. It is stored in
``tools.json`` — at the workspaces root for the global scope, and inside a
workspace directory for that workspace's override. A workspace configuration
takes precedence while it exists; the global configuration still applies to
workspaces without one.

Tool names are the unit of configuration rather than category flags, so a
saved file always says exactly which tools were removed. The UI derives its
category switches from these lists.
"""

from __future__ import annotations

from typing import Any

from .registry import ToolRegistry

#: Tools that stay enabled regardless of the saved configuration.
#:
#: These are the harness's spine: the system prompt names them as the way to
#: discover, pause, inspect, read, and run code. Disabling one would either
#: strand the model (a deferred tool it cannot discover) or leave a prompt rule
#: pointing at a missing tool. Model-specific vision tools are deliberately
#: *not* here: object detection, segmentation, and image analysis are what the
#: model choices control.
REQUIRED_TOOL_REASONS: dict[str, str] = {
    "request_user_input": (
        "System-prompt rule 2: the only way a run pauses for structured user choices."
    ),
    "discover_capabilities": (
        "System-prompt rule 1: finds the right deferred tools for a broad request."
    ),
    "describe_tool": (
        "System-prompt rule 1: documents a discovered tool and makes it callable."
    ),
    "run_python": (
        "System-prompt rule 4: the fallback for math or processing no tool covers."
    ),
    "python_help": (
        "System-prompt rule 4: documents the run_python API instead of guessing."
    ),
    "inspect_output": (
        "Runtime rule: pages the bounded preview run_python returns."
    ),
    "query_output": (
        "Runtime rule: extracts a JSON/XML value from a large run_python result."
    ),
    "read_file": "Every text input path starts here; no other tool reads file contents.",
    "list_files": "Orientation: lists the workspace inputs before an analysis starts.",
    "find_files": "Orientation: locates inputs by name across data/ and results/.",
    "write_file": "Writes notes and small text artifacts into results/.",
    "describe_map": "System-prompt rule 10: required after a map change.",
    "get_drawn_features": "System-prompt rule 10: the only read of user-drawn AOIs.",
    "describe_geolibre_bridge": (
        "Reports which map methods the embedded bridge exposes; supports rule 10."
    ),
    "skill": "The skills index documents installed Python packages and their APIs.",
    "ai_models": (
        "System-prompt rule 5: names the local models and their download state "
        "before an AI task starts."
    ),
}

#: Categories whose tools are model-driven rather than part of the core loop.
MODEL_SPECIFIC_CATEGORIES = frozenset({"ai", "segmentation", "detection", "vision"})

#: Individual model-specific tools that do not sit in one of those categories.
MODEL_SPECIFIC_TOOLS = frozenset({"analyze_image"})

#: Display order and labels for the Tools tab.
CATEGORY_LABELS: dict[str, str] = {
    "interaction": "Interaction",
    "capability": "Capability discovery",
    "files": "Workspace files",
    "catalog": "Imagery catalogs",
    "layers": "Map & layers",
    "raster": "Raster analysis",
    "vector": "Vector analysis",
    "ai": "Local model management",
    "segmentation": "Image segmentation",
    "detection": "Object detection",
    "python": "Python runtime",
    "skills": "Skills reference",
    "vision": "Image understanding",
    "plan": "Planning (harness)",
    "search": "Tool search (harness)",
    "web": "Web search (harness)",
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "interaction": "Structured questions that pause a run for user input.",
    "capability": "Discovery and documentation for deferred tools.",
    "files": "Workspace file listing, reading, writing, and downloads.",
    "catalog": "Vantor, OpenAerialMap, and STAC imagery search.",
    "layers": "Adding, styling, comparing, and inspecting map layers.",
    "raster": "Clipping, indices, terrain, zonal stats, and rasters on the map.",
    "vector": "Vector inspection, geometry analysis, and map publication.",
    "ai": "The local model cache, downloads, and memory.",
    "segmentation": "Promptable segmentation with local SAM-family models.",
    "detection": "Known-object detection with local ONNX detectors.",
    "python": "The run_python sandbox and its output readers.",
    "skills": "Reading the installed package API references.",
    "vision": "Asking the configured image-capable model about the map.",
    "plan": "Task planning tools registered by the harness.",
    "search": "The harness's deferred-tool search; always available.",
    "web": "Provider-native or local web search; always available.",
}


def default_tool_config() -> dict:
    """Return the empty configuration (everything enabled)."""
    return {
        "disabled_tools": [],
        "python": {"disabled_packages": [], "disabled_methods": []},
    }


def _names(value: Any) -> list[str]:
    """Return unique, non-empty strings from an arbitrary JSON value."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    seen: set[str] = set()
    names: list[str] = []
    for item in value:
        name = str(item).strip() if item is not None else ""
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return sorted(names)


def normalize_tool_config(config: Any) -> dict:
    """Return a well-formed configuration, preserving unknown names.

    Unknown names are kept rather than dropped: a configuration written by a
    newer build, or for a tool that is not available on this model, should still
    be there when the tool returns.
    """
    data = config if isinstance(config, dict) else {}
    python = data.get("python")
    python = python if isinstance(python, dict) else {}
    return {
        "disabled_tools": _names(data.get("disabled_tools")),
        "python": {
            "disabled_packages": _names(python.get("disabled_packages")),
            "disabled_methods": _names(python.get("disabled_methods")),
        },
    }


def required_tool_names() -> frozenset[str]:
    """Names that Tool Configuration cannot disable."""
    return frozenset(REQUIRED_TOOL_REASONS)


def required_tool_reason(name: str) -> str:
    """Return why ``name`` is locked, or an empty string."""
    return REQUIRED_TOOL_REASONS.get(name, "")


def is_model_specific(name: str, category: str = "") -> bool:
    """Whether a tool depends on a model choice rather than the core harness."""
    return name in MODEL_SPECIFIC_TOOLS or category in MODEL_SPECIFIC_CATEGORIES


def rejected_disables(config: Any) -> list[str]:
    """Return required tools the configuration tries to disable."""
    disabled = set(normalize_tool_config(config)["disabled_tools"])
    return sorted(disabled & required_tool_names())


def apply_tool_config(registry: ToolRegistry, config: dict | None) -> list[str]:
    """Remove configurable disabled tools from ``registry``; return their names.

    Required tools are ignored even if an old or hand-edited file names them,
    which is the same rule the save endpoint enforces. Tools implemented by a
    capability (plan, tool search, web search) are not ours to remove; their
    capability owns them and the registry spec exists only for discovery.
    """
    disabled = set(normalize_tool_config(config)["disabled_tools"])
    removed: list[str] = []
    for name in list(registry.names()):
        if name not in disabled or name in required_tool_names():
            continue
        spec = registry.get(name)
        if not spec.implemented or spec.origin != "builtin":
            continue
        registry.remove(name)
        removed.append(name)
    return removed


__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "CATEGORY_LABELS",
    "MODEL_SPECIFIC_CATEGORIES",
    "MODEL_SPECIFIC_TOOLS",
    "REQUIRED_TOOL_REASONS",
    "apply_tool_config",
    "default_tool_config",
    "is_model_specific",
    "normalize_tool_config",
    "rejected_disables",
    "required_tool_names",
    "required_tool_reason",
]
