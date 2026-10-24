"""The JSON catalog Tool Configuration renders.

Two consumers share this payload: the modal (tools grouped by category, then
the run_python import/helper lists) and tests (the required-tool and
model-specific flags). It is built from declarations, not from a live registry,
so the page works before a workspace is open and still lists the vision tool
that the current model cannot serve.
"""

from __future__ import annotations

from ..pythonruntime import packages
from ..pythonruntime.policy import (
    CORE_IMPORTS,
    METHOD_OPTIONS,
    PythonPolicy,
    import_groups,
    method_groups,
)
from .build import declared_specs
from .config import (
    CATEGORY_DESCRIPTIONS,
    CATEGORY_LABELS,
    REQUIRED_TOOL_REASONS,
    is_model_specific,
    normalize_tool_config,
    required_tool_reason,
)
from .spec import ToolSpec

#: Import names that are always present (they are in the base install) rather
#: than optional extras.
_CORE_NAMES = frozenset(option.name for option in CORE_IMPORTS)


def _tool_row(spec: ToolSpec, disabled: set[str], required: frozenset[str], vision_supported: bool) -> dict:
    configurable = bool(spec.fn is not None and spec.origin == "builtin")
    required_here = spec.name in required
    configurable = configurable and not required_here
    available = spec.name != "analyze_image" or vision_supported
    return {
        "name": spec.name,
        "summary": spec.summary,
        "category": spec.category,
        "origin": spec.origin,
        "core": spec.core,
        "required": required_here,
        "required_reason": required_tool_reason(spec.name),
        "configurable": configurable,
        "enabled": required_here or not configurable or spec.name not in disabled,
        "available": available,
        "model_specific": is_model_specific(spec.name, spec.category),
    }


def _category_payload(rows: list[dict]) -> dict:
    configurable = [row for row in rows if row["configurable"]]
    enabled = [row for row in configurable if row["enabled"]]
    return {
        "id": rows[0]["category"],
        "label": CATEGORY_LABELS.get(rows[0]["category"], rows[0]["category"].title()),
        "description": CATEGORY_DESCRIPTIONS.get(rows[0]["category"], ""),
        "tools": rows,
        "configurable_count": len(configurable),
        "enabled_count": len(enabled),
    }


def tool_categories(config: dict | None, vision_supported: bool = True) -> list[dict]:
    """Return tool rows grouped into ordered categories."""
    normalized = normalize_tool_config(config)
    disabled = set(normalized["disabled_tools"])
    required = frozenset(REQUIRED_TOOL_REASONS)
    grouped: dict[str, list[dict]] = {}
    for spec in declared_specs():
        grouped.setdefault(spec.category, []).append(
            _tool_row(spec, disabled, required, vision_supported)
        )
    order = {category: index for index, category in enumerate(CATEGORY_LABELS)}
    ordered = sorted(
        grouped.items(),
        key=lambda item: (order.get(item[0], len(order)), item[0]),
    )
    return [_category_payload(rows) for _, rows in ordered]


def _package_rows(config: dict | None) -> list[dict]:
    normalized = normalize_tool_config(config)
    disabled = set(normalized["python"]["disabled_packages"])
    rows: list[dict] = []
    for option in CORE_IMPORTS:
        rows.append(
            {
                "name": option.name,
                "group": option.group,
                "purpose": option.purpose,
                "optional": False,
                "installed": True,
                "version": "",
                "enabled": option.name not in disabled,
            }
        )
    for package in packages.PACKAGES:
        rows.append(
            {
                "name": package.import_name,
                "group": "Optional extras",
                "purpose": package.purpose,
                "optional": True,
                "installed": packages.installed(package),
                "version": packages.version(package),
                "enabled": package.import_name not in disabled,
            }
        )
    return rows


def _method_rows(config: dict | None) -> list[dict]:
    normalized = normalize_tool_config(config)
    disabled = set(normalized["python"]["disabled_methods"])
    return [
        {
            "name": option.name,
            "group": option.group,
            "purpose": option.purpose,
            "enabled": option.name not in disabled,
        }
        for option in METHOD_OPTIONS
    ]


def catalog_payload(config: dict | None, vision_supported: bool = True) -> dict:
    """Return the complete Tool Configuration catalog."""
    normalized = normalize_tool_config(config)
    categories = tool_categories(normalized, vision_supported)
    return {
        "categories": categories,
        "required_tools": dict(REQUIRED_TOOL_REASONS),
        "python": {
            "packages": _package_rows(normalized),
            "package_groups": import_groups(),
            "methods": _method_rows(normalized),
            "method_groups": method_groups(),
        },
        "policy": PythonPolicy.from_config(normalized).summary(),
    }


def package_names() -> frozenset[str]:
    """Every import name the runtime page lists (for tests and validation)."""
    return _CORE_NAMES | frozenset(package.import_name for package in packages.PACKAGES)


__all__ = ["catalog_payload", "package_names", "tool_categories"]
