"""Structured introspection over the ``run_python`` sandbox namespace.

The agent used to answer "what does this symbol look like?" by generating
``dir()`` / ``__doc__`` / ``inspect.signature`` snippets inside ``run_python``,
one round-trip per slice of a long docstring. :func:`describe` returns the same
information in one call: kind, signature, trimmed docstring, and public
members. API discovery becomes a single structured tool instead of a probe loop.
"""

from __future__ import annotations

import enum
import inspect
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rioxarray
import rio_cogeo
import scipy
import shapely
import skimage
import xarray


from . import packages
from .policy import PythonPolicy

#: Cap on a returned docstring.
MAX_DOC = 4000
#: Cap on how many members a ``dir``-style listing returns.
MAX_MEMBERS = 48
#: Cap on a rendered member/value.
MAX_REPR = 200

#: Module roots the agent may reference inside ``run_python`` (mirrors the
#: namespace bound there, plus the long-name aliases the model tends to use).
ROOTS = {
    "rasterio": rasterio,
    "rioxarray": rioxarray,
    "numpy": np,
    "np": np,
    "pandas": pd,
    "pd": pd,
    "geopandas": gpd,
    "gpd": gpd,
    "xarray": xarray,
    "xr": xarray,
    "skimage": skimage,
    "scipy": scipy,
    "rio_cogeo": rio_cogeo,
    "shapely": shapely,
    "pyproj": pyproj,
    "json": json,
}


#: Alias -> import root, for filtering with the runtime policy.
_ROOT_PACKAGES = {
    "rasterio": "rasterio",
    "rioxarray": "rioxarray",
    "numpy": "numpy",
    "np": "numpy",
    "pandas": "pandas",
    "pd": "pandas",
    "geopandas": "geopandas",
    "gpd": "geopandas",
    "xarray": "xarray",
    "xr": "xarray",
    "skimage": "skimage",
    "scipy": "scipy",
    "rio_cogeo": "rio_cogeo",
    "shapely": "shapely",
    "pyproj": "pyproj",
    "json": "json",
}


def roots(policy: PythonPolicy | None = None) -> dict:
    """Return the core namespace plus installed, enabled optional packages.

    Called per lookup rather than built at import: importing torch to answer a
    question about ``rasterio`` would cost seconds on every app start.
    """
    core = {
        name: module
        for name, module in ROOTS.items()
        if policy is None or policy.allows_import(_ROOT_PACKAGES[name])
    }
    return {**core, **packages.bindings(policy)}

#: The workspace facade ``run_python`` binds, described as its own "symbol".
VISION_DOC = (
    "Image-input helper available in run_python.\n"
    "  vision.supported -> bool   (a property: use it without parentheses)\n"
    "  vision.ask(question, image='map', max_side=1400, bands=None,\n"
    "             thinking=False, max_tokens=2048) -> str\n"
    "      image is 'map' (the current GeoLibre viewport) or a workspace path\n"
    "      to a PNG/JPEG/GeoTIFF. Returns the model's answer; use JSON in the\n"
    "      question if the snippet needs boxes. Rasters are downsampled and\n"
    "      converted to RGB JPEG; bands picks the 1-based R,G,B channels\n"
    "      (default first three), e.g. bands=[4,3,2]. thinking is DeepSeek's\n"
    "      reasoning switch and defaults off for direct JSON/box answers.\n"
    "Only available when the configured endpoint and model accept images."
)

WS_DOC = (
    "Confined workspace facade available in run_python (no raw path/network).\n"
    "  ws.resolve(rel, must_exist=False, write=False) -> Path\n"
    "      Resolve a workspace-relative path; raises WorkspaceError outside root.\n"
    "  ws.read_text(rel, encoding='utf-8') -> str\n"
    "  ws.write_text(rel, content, encoding='utf-8') -> str  (absolute path)\n"
    "  ws.list_files(subdir='', pattern='*') -> list[str]\n"
    "  ws.data / ws.results / ws.maps -> str  (absolute directory paths)\n"
    "Prefer the structured tools (read_file/write_file) over ws for file I/O."
)

MODELS_DOC = (
    "Read-only view of the local model cache, bound in run_python as models.\n"
    "  models.dir(repo='') -> str   cache root, or one repository directory\n"
    "  models.path(repo, filename='') -> str   absolute path; raises when missing\n"
    "  models.list(repo='') -> list[str]   cached repositories or one repo's files\n"
    "Weights arrive through ai_fetch_model; the sandbox itself has no network."
)


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _trim(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) > limit:
        return text[:limit].rstrip() + f"\n... [{limit} char cap]"
    return text


def _doc(obj) -> str:
    raw = getattr(obj, "__doc__", None) or ""
    return _trim(inspect.cleandoc(raw), MAX_DOC)


def _sig(obj) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        pass
    text_sig = getattr(obj, "__text_signature__", None)
    return str(text_sig) if text_sig else None


def _brief(obj) -> str:
    """One-line description of a member (for ``dir``-style listings)."""
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        sig = _sig(obj)
        return "class" + (sig or "()")
    if inspect.isroutine(obj):
        return _sig(obj) or "built-in (see python_help for the doc)"
    if isinstance(obj, enum.Enum):
        return f"enum member ({obj.name})"
    try:
        rep = repr(obj)
    except Exception:  # noqa: BLE001 - members can raise on repr
        rep = f"<{type(obj).__name__}>"
    return f"{type(obj).__name__} = {_trim(rep, MAX_REPR)}"


def _members(obj) -> dict:
    out: dict[str, str] = {}
    for name in dir(obj):
        if not _is_public(name):
            continue
        try:
            out[name] = _brief(getattr(obj, name))
        except Exception:  # noqa: BLE001 - skip members that raise on access
            out[name] = "?"
        if len(out) >= MAX_MEMBERS:
            out["..."] = f"[truncated; {MAX_MEMBERS} of many members shown]"
            break
    return out


def _describe(name: str, obj) -> dict:
    result: dict = {"name": name, "kind": "", "doc": _doc(obj)}
    if isinstance(obj, type) and issubclass(obj, enum.Enum):
        result["kind"] = "enum"
        result["members"] = {m.name: m.value for m in obj}
    elif inspect.ismodule(obj):
        result["kind"] = "module"
        result["members"] = _members(obj)
    elif inspect.isclass(obj):
        result["kind"] = "class"
        result["signature"] = _sig(obj) or "()"
        result["members"] = _members(obj)
    elif inspect.isroutine(obj):
        result["kind"] = "function"
        sig = _sig(obj)
        if sig:
            result["signature"] = sig
    else:
        result["kind"] = "attribute"
        result["type"] = type(obj).__name__
        result["value"] = _trim(repr(obj), MAX_REPR)
    return result


def describe(name: str = "", policy: PythonPolicy | None = None) -> dict:
    """Return kind, signature, docstring, and members for a sandbox symbol.

    Pass a dotted name reachable from the ``run_python`` namespace, e.g.
    ``"rasterio.warp.reproject"``, ``"rasterio.control.GroundControlPoint"``,
    ``"rasterio.warp.Resampling"``, ``"numpy.ndarray"``, or ``"ws"``. With no
    argument it lists the available top-level roots. ``policy`` hides imports
    and helper methods disabled in Tool Configuration.
    """
    name = (name or "").strip()
    available = roots(policy)
    helpers: list[str] = []
    if policy is None or policy.any_method("ws."):
        helpers.append("ws")
    if policy is None or policy.any_method("vision."):
        helpers.append("vision")
    if policy is None or policy.any_method("models."):
        helpers.append("models")
    if not name:
        return {
            "kind": "namespace",
            "roots": sorted(available) + helpers,
            "doc": "Top-level run_python names. Query a dotted path for details.",
        }

    helper_docs = {"ws": WS_DOC, "vision": VISION_DOC, "models": MODELS_DOC}
    helper_kind = {
        "ws": "workspace helper",
        "vision": "image helper",
        "models": "model cache helper",
    }
    for helper, doc in helper_docs.items():
        if name == helper or name.startswith(helper + "."):
            if helper not in helpers:
                return {
                    "name": name,
                    "kind": "error",
                    "doc": f"{helper} is disabled in Tool Configuration",
                }
            return {"name": name, "kind": helper_kind[helper], "doc": doc}

    parts = name.split(".")
    root = parts[0]
    if root not in available:
        return {
            "name": name,
            "kind": "error",
            "doc": f"unknown root {root!r}; available: {sorted(available) + helpers}",
        }

    obj = available[root]
    try:
        for part in parts[1:]:
            obj = getattr(obj, part)
    except AttributeError as exc:
        return {"name": name, "kind": "error", "doc": f"no attribute: {exc}"}

    return _describe(name, obj)


__all__ = [
    "MAX_DOC",
    "MAX_MEMBERS",
    "MAX_REPR",
    "MODELS_DOC",
    "ROOTS",
    "VISION_DOC",
    "WS_DOC",
    "describe",
    "roots",
]
