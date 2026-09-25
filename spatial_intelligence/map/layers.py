"""Layer operations on the shared live GeoLibre map.

Ported from the pre-package ``skills/map_tools.py``: each operation is a plain
service function taking the workspace and the live map explicitly (never
ambient state), and every mutation persists the project through
:func:`spatial_intelligence.map.document.persist_map` so the map survives a
kernel restart and stays in sync with the workspace.

``add_raster`` receives the serving URL through a ``file_url`` callable instead
of importing the runtime, which keeps this module free of tool-layer
dependencies and lets headless callers pass any URL builder.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse

from geolibre import Map
from geolibre import authoring as geolibre_authoring

from ..contracts.errors import ToolInputError
from ..workspace import Workspace, WorkspaceError
from ..geo import raster
from . import styles
from .document import persist_map

#: Layer metadata key holding the workspace-relative path of a local source.
LOCAL_SOURCE_KEY = "geoaiSourcePath"

__all__ = [
    "LOCAL_SOURCE_KEY",
    "add_colorbar",
    "add_geojson",
    "add_legend",
    "add_raster",
    "add_tile_layer",
    "add_vector",
    "add_wms",
    "classify_layer",
    "clean_layer_name",
    "clear_layers",
    "describe",
    "export_html",
    "find_layer",
    "fit_bounds",
    "layers",
    "list_colormaps",
    "list_style_keys",
    "remove_layer",
    "repoint_local_rasters",
    "save_map",
    "set_basemap",
    "set_layer_metadata",
    "set_layer_opacity",
    "set_layer_visibility",
    "set_view",
    "swipe_compare",
    "style_layer",
]


# -- inspection ------------------------------------------------------------


def layers(map_obj: Map) -> list[dict[str, Any]]:
    """Return the live project's layer records (the mutable dicts themselves)."""
    return map_obj.project.get("layers", [])


def find_layer(map_obj: Map, layer_id: str) -> dict[str, Any] | None:
    """Return the layer record with ``layer_id``, or ``None`` when absent."""
    for layer in layers(map_obj):
        if layer.get("id") == layer_id:
            return layer
    return None


def set_layer_metadata(map_obj: Map, layer_id: str, key: str, value: Any) -> None:
    """Set ``metadata[key]`` on a layer; a no-op when the layer is gone."""
    layer = find_layer(map_obj, layer_id)
    if layer is not None:
        layer.setdefault("metadata", {})[key] = value


def describe(map_obj: Map) -> dict:
    """Return a compact summary of the current map (layers, view, basemap).

    Each layer summary carries ``style``: the style GeoLibre renders for it,
    so a caller can confirm a styling call instead of assuming it landed.
    """
    summary = map_obj.describe()
    records = {str(layer.get("id")): layer for layer in layers(map_obj)}
    for entry in summary.get("layers", []):
        record = records.get(str(entry.get("id")))
        if record is not None:
            entry["style"] = styles.summary(map_obj.project, record)
    return summary


def list_style_keys() -> dict:
    """Return the style vocabulary ``style_layer`` accepts.

    ``style`` lists the layer style keys, ``labels`` the keys of the nested
    text-label object, and ``aliases`` the foreign names translated for you.
    Call this instead of guessing a key: GeoLibre ignores keys it does not
    know, and ``style_layer`` rejects them for that reason.
    """
    return styles.style_keys()


def list_colormaps() -> dict[str, list[str]]:
    """Return the named color ramps valid for ``colormap``/``palette`` arguments.

    Each key is a valid ``colormap`` value for ``add_raster``/``add_colorbar``
    and a valid ``palette`` value for ``classify_layer``/``add_vector_to_map``;
    the value is the ramp's anchor CSS colors. Prefer this over guessing a name.
    """
    from geolibre.color_ramp import VECTOR_COLOR_RAMPS

    return {name: list(colors) for name, colors in VECTOR_COLOR_RAMPS.items()}


# -- local-source bookkeeping ---------------------------------------------


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _is_geojson_literal(s: str) -> bool:
    return s.lstrip().startswith(("{", "["))


def _tag_local_source(map_obj: Map, layer_id: str, rel: str) -> None:
    """Record the workspace-relative path of a locally-served raster layer."""
    set_layer_metadata(map_obj, layer_id, LOCAL_SOURCE_KEY, rel)


def _local_rel_for_layer(layer: dict, workspace: Workspace) -> str | None:
    """Return the workspace-relative path of a locally-served raster layer.

    Prefers the ``geoaiSourcePath`` metadata tag. Falls back to recovering the
    path from a session URL — either the old ``_geolibre_local/...`` scheme (by
    filename under ``results/``/``data/``/``maps/``) or a ``/api/files/<rel>``
    URL saved under a previous server port — so projects persist across restarts
    and port changes.
    """
    meta = layer.get("metadata") or {}
    rel = meta.get(LOCAL_SOURCE_KEY)
    if rel:
        return rel
    raw = layer.get("sourcePath")
    if not isinstance(raw, str):
        source = layer.get("source")
        if isinstance(source, dict):
            raw = source.get("url")
    if not isinstance(raw, str):
        return None
    if "_geolibre_local/" in raw:
        name = Path(urlparse(raw).path).name
        if not name:
            return None
        for sub in ("results", "data", "maps"):
            rel = f"{sub}/{name}"
            try:
                candidate = workspace.resolve(rel, must_exist=True)
            except WorkspaceError:
                continue
            if candidate.is_file():
                return rel
        return None
    if "/api/files/" in raw:
        rel = unquote(urlparse(raw).path.split("/api/files/", 1)[1])
        try:
            candidate = workspace.resolve(rel, must_exist=True)
        except WorkspaceError:
            return None
        return rel if candidate.is_file() else None
    return None


def repoint_local_rasters(
    map_obj: Map, workspace: Workspace, file_url: Callable[[str], str]
) -> int:
    """Re-point local raster layers to the app's stable file route.

    ``add_raster`` embeds a stable ``/api/files/<rel>`` URL served by this
    harness's own server (with CORS), so local rasters survive a server restart
    without the geolibre static server's per-session token. This migrates layers
    saved with the old ``_geolibre_local/...`` session URL back to the stable URL
    (recovering the workspace path from ``metadata.geoaiSourcePath`` or, when
    that tag is missing, by filename). Returns the number of layers re-pointed.
    """
    count = 0
    for layer in layers(map_obj):
        rel = _local_rel_for_layer(layer, workspace)
        if not rel:
            continue
        url = file_url(rel)
        source = layer.get("source")
        if isinstance(source, dict):
            source["url"] = url
        layer["sourcePath"] = url
        layer.setdefault("metadata", {})[LOCAL_SOURCE_KEY] = rel
        layer["metadata"].pop("error", None)
        count += 1
    return count


# -- writers ---------------------------------------------------------------


def clean_layer_name(name: str) -> str:
    """Return a new layer's display name, cleaned the way GeoLibre's own setter is.

    GeoLibre strips and refuses a blank name when a layer is *renamed* but not
    when one is *added*, so a padded or empty name would otherwise persist
    verbatim and render as an unreferenceable row. The basemap pseudo-id is
    refused too: ``resolve_layer_ids`` passes it through as the basemap
    sentinel, so a layer wearing it would be silently unaddressable there.

    Raises:
        ToolInputError: If the name is blank or the reserved basemap id.
    """
    clean = str(name).strip()
    if not clean:
        raise ToolInputError("a layer name must be a non-empty string")
    if clean == geolibre_authoring.BASEMAP_LAYER_ID:
        raise ToolInputError(
            f"{geolibre_authoring.BASEMAP_LAYER_ID!r} is reserved for the basemap"
        )
    return clean


def add_geojson(
    workspace: Workspace,
    map_obj: Map,
    data: str,
    name: str,
    *,
    style: dict[str, Any] | None = None,
) -> str:
    """Add a GeoJSON layer and return its id.

    ``data`` may be a workspace-relative path, an http(s) URL, or a literal
    GeoJSON string; ``name`` is the layer's display name (unique names are what
    later by-name references resolve against). ``style`` is applied through the
    same path as :func:`style_layer` (see there for the accepted keys).
    """
    name = clean_layer_name(name)
    if not _is_url(data) and not _is_geojson_literal(data):
        data = str(workspace.resolve(data, must_exist=True))
    layer_id = map_obj.add_geojson(data, name)
    if style:
        styles.apply_style(map_obj.project, layer_id, style)
    persist_map(map_obj, workspace)
    return layer_id


def add_vector(
    workspace: Workspace,
    map_obj: Map,
    data: str,
    name: str,
    *,
    data_format: str | None = None,
    source_layer: str | None = None,
    style: dict[str, Any] | None = None,
) -> str:
    """Add a vector layer from a path/URL and return its id.

    ``name`` is the layer's display name. ``style`` is applied through the same
    path as :func:`style_layer`.
    """
    name = clean_layer_name(name)
    if not _is_url(data):
        data = str(workspace.resolve(data, must_exist=True))
    layer_id = map_obj.add_vector(
        data, name, data_format=data_format, source_layer=source_layer
    )
    if style:
        styles.apply_style(map_obj.project, layer_id, style)
    persist_map(map_obj, workspace)
    return layer_id


def add_raster(
    workspace: Workspace,
    map_obj: Map,
    path: str,
    name: str,
    *,
    colormap: str | None = None,
    rescale: list[float] | None = None,
    bands: list[int] | None = None,
    file_url: Callable[[str], str] | None = None,
) -> str:
    """Add a raster (COG/GeoTIFF) layer and return its id.

    ``bands`` selects the bands to draw (1-based) -- ``[1]`` for one band of a
    multi-band file, ``[1, 2, 3]`` for a colour composite. ``rescale`` is a
    ``[min, max]`` stretch for a single band. ``colormap`` is
    one of the names from :func:`list_colormaps` (e.g. ``"viridis"``, ``"gray"``,
    ``"blues"``, ``"terrain"``); omit it to render the raw values.

    A workspace-local ``path`` is validated for confinement/existence and, when
    ``file_url`` is given, embedded as that stable ``/api/files/<rel>`` URL (the
    workspace path is tagged in layer metadata so it can be re-pointed later).
    """
    name = clean_layer_name(name)
    rel = None
    if not _is_url(path):
        resolved = workspace.resolve(path, must_exist=True)
        # The app loads a COG: handed a striped GeoTIFF it asks its Python
        # sidecar to convert the file, and this harness embeds the static app
        # without that sidecar, so the conversion could never succeed. Convert
        # here, so every raster any tool wrote is loadable on the map.
        loadable, _ = raster.to_cog_if_needed(workspace, path)
        rel = workspace.relative(loadable)
        path = file_url(rel) if file_url is not None else str(loadable)
    rescale_arg = [list(rescale)] if rescale else None
    layer_id = map_obj.add_raster(
        path, name, bands=bands, colormap=colormap, rescale=rescale_arg
    )
    if rel is not None and file_url is not None:
        _tag_local_source(map_obj, layer_id, rel)
    persist_map(map_obj, workspace)
    return layer_id


def add_tile_layer(
    workspace: Workspace, map_obj: Map, url: str, name: str, attribution: str | None = None
) -> str:
    """Add an XYZ tile layer and return its id."""
    layer_id = map_obj.add_tile_layer(url, clean_layer_name(name), attribution=attribution)
    persist_map(map_obj, workspace)
    return layer_id


def add_wms(
    workspace: Workspace,
    map_obj: Map,
    endpoint: str,
    layers: str,
    name: str,
    styles: str | None = None,
) -> str:
    """Add a WMS tiled layer and return its id.

    ``styles`` is the service's ``STYLES`` parameter; an empty string asks for the
    default style, which is also what a caller who passes nothing means. It must
    reach the layer builder as a string: GeoLibre encodes the query with
    ``urllib.parse.quote``, which rejects ``None`` ("quote_from_bytes() expected
    bytes") and would fail the whole call.

    A service that selects its frame by date carries that date on the endpoint
    (``.../wms.cgi?TIME=2026-09-24``); the builder appends its own parameters to
    whatever query is already there, so the date survives.
    """
    layer_id = map_obj.add_wms(
        endpoint, layers, clean_layer_name(name), styles=styles or ""
    )
    persist_map(map_obj, workspace)
    return layer_id


def set_view(
    workspace: Workspace,
    map_obj: Map,
    center: list[float] | None = None,
    zoom: float | None = None,
) -> dict:
    """Center/zoom the map. ``center`` is ``[lng, lat]``."""
    if center is not None:
        map_obj.set_center(center[0], center[1], zoom=zoom)
    elif zoom is not None:
        map_obj.set_zoom(zoom)
    persist_map(map_obj, workspace)
    return {"status": "applied", "mapView": deepcopy(map_obj.project.get("mapView"))}


def set_basemap(workspace: Workspace, map_obj: Map, basemap: str) -> dict:
    """Set the background basemap (name or MapLibre style URL)."""
    map_obj.set_basemap(basemap)
    persist_map(map_obj, workspace)
    return {"status": "applied", "basemap": basemap}



#: Swipe placeholder selecting the basemap as one side of the comparison.
#: Geolibre's own sentinel, not a second copy of the literal: it crosses this
#: boundary in both directions, so one definition has to win.
BASEMAP_SIDE = geolibre_authoring.BASEMAP_LAYER_ID


def _refs(value: str | Iterable[str]) -> list[str]:
    """Return layer references from a single name or a list of them.

    A bare name is the common call, and iterating it directly would split it into
    characters ("unknown layer 'R'").
    """
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _layer_ids(map_obj: Map, refs: list[str]) -> list[str]:
    """Resolve layer references (id or display name) to layer ids.

    ``__basemap__`` passes through: GeoLibre's split map can compare a layer
    against the basemap itself, which is how "before/after imagery over the
    basemap" is expressed.
    """
    known = layers(map_obj)
    ids = {str(layer.get("id")) for layer in known}
    by_name = {str(layer.get("name")): str(layer.get("id")) for layer in known}
    resolved: list[str] = []
    for ref in refs:
        if ref == BASEMAP_SIDE:
            resolved.append(ref)
        elif ref in ids:
            resolved.append(ref)
        elif ref in by_name:
            resolved.append(by_name[ref])
        else:
            raise ToolInputError(
                f"unknown layer {ref!r}; call describe_map for the layer ids and names"
            )
    return resolved


def swipe_compare(
    workspace: Workspace,
    map_obj: Map,
    left: str | list[str],
    right: str | list[str],
    orientation: str = "vertical",
    position: float = 50,
    control_position: str = "top-right",
) -> dict:
    """Configure the split-map (swipe) control between two sets of layers.

    ``left``/``right`` are layer ids or display names — one name or a list of them
    — and ``__basemap__`` stands for the background map. ``orientation`` is
    ``vertical`` or ``horizontal``; ``position`` is the initial slider percentage;
    ``control_position`` is which corner holds the handle. GeoLibre draws the two
    sides as one comparison, so the layers stay in the project and remain
    individually styleable.
    """
    left_ids = _layer_ids(map_obj, _refs(left))
    right_ids = _layer_ids(map_obj, _refs(right))
    if not left_ids or not right_ids:
        raise ToolInputError("swipe_compare needs at least one layer on each side")
    try:
        state = geolibre_authoring.add_swipe(
            map_obj.project,
            left_layers=left_ids,
            right_layers=right_ids,
            orientation=orientation,
            position=position,
            control_position=control_position,
        )
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc
    persist_map(map_obj, workspace)
    return {"status": "applied", "swipe": state}

def style_layer(
    workspace: Workspace, map_obj: Map, layer: str, style: dict[str, Any]
) -> dict:
    """Merge style overrides onto a layer (e.g. ``{"fillColor": "#ff0000"}``).

    ``layer`` is a layer id or display name. ``style`` uses GeoLibre's own
    style keys -- ``fillColor``, ``strokeColor``, ``strokeWidth``,
    ``fillOpacity``, ``circleRadius``, ... -- plus the nested text-label object::

        {"labels": {"enabled": true, "field": "pop_fmt", "size": 13}}

    Label text needs both ``labels.enabled`` and a ``field`` naming a feature
    property. A point layer draws circle markers unless told otherwise, so a
    text-only label layer also sets ``{"circleRadius": 0}``.

    A few foreign names are translated (``textField`` -> ``labels.field``,
    ``color`` -> ``strokeColor``); every other unknown key is rejected rather
    than accepted as the silent no-op GeoLibre makes of it. Call
    :func:`list_style_keys` for the full vocabulary.

    The result carries the layer's complete, rendered style, so the caller can
    read back what the map will draw.
    """
    target, merged = styles.apply_style(map_obj.project, layer, style)
    persist_map(map_obj, workspace)
    return {
        "status": "applied",
        "layer": target.get("name"),
        "layerId": target.get("id"),
        "style": merged,
    }


def classify_layer(
    workspace: Workspace,
    map_obj: Map,
    layer: str,
    column: str,
    palette: str = "viridis",
    method: str = "quantile",
    k: int = 5,
) -> dict:
    """Symbolize a GeoJSON layer as a choropleth on a numeric ``column``.

    ``method`` is ``"quantile"`` or ``"equal-interval"``; ``k`` is the class
    count; ``palette`` is a color-ramp name.
    """
    project = map_obj.project
    target = styles.resolve(project, layer)
    try:
        fragment = geolibre_authoring.build_choropleth_style(
            geolibre_authoring.column_values(target, column),
            column,
            class_count=k,
            colormap=palette,
            scheme=method,
        )
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc
    _, merged = styles.apply_style(project, target["id"], fragment)
    persist_map(map_obj, workspace)
    return {
        "status": "applied",
        "layer": target.get("name"),
        "layerId": target.get("id"),
        "column": column,
        "palette": palette,
        "method": method,
        "classes": k,
        "style": merged,
    }


def set_layer_visibility(
    workspace: Workspace, map_obj: Map, layer: str, visible: bool
) -> dict:
    """Show or hide a layer."""
    map_obj.set_layer_visibility(layer, visible)
    persist_map(map_obj, workspace)
    return {"status": "applied", "layer": layer, "visible": visible}


def set_layer_opacity(
    workspace: Workspace, map_obj: Map, layer: str, opacity: float
) -> dict:
    """Set a layer's opacity in ``[0, 1]``."""
    map_obj.set_layer_opacity(layer, opacity)
    persist_map(map_obj, workspace)
    return {"status": "applied", "layer": layer, "opacity": opacity}


def remove_layer(workspace: Workspace, map_obj: Map, layer: str) -> dict:
    """Remove a layer by id or display name."""
    map_obj.remove_layer(layer)
    persist_map(map_obj, workspace)
    return {"status": "applied", "removed": layer}


def clear_layers(workspace: Workspace, map_obj: Map) -> dict:
    """Remove all layers from the map."""
    map_obj.clear_layers()
    persist_map(map_obj, workspace)
    return {"status": "applied", "layerCount": 0}


def add_legend(
    workspace: Workspace,
    map_obj: Map,
    title: str | None = None,
    items: dict[str, str] | None = None,
    shape: str = "square",
) -> dict:
    """Add a legend. ``items`` maps label -> CSS color."""
    map_obj.add_legend(title=title, legend_dict=items, shape=shape)
    persist_map(map_obj, workspace)
    return {"status": "applied", "title": title, "items": items, "shape": shape}


def add_colorbar(
    workspace: Workspace,
    map_obj: Map,
    colormap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> dict:
    """Add a colorbar for a continuous (single-band) raster.

    ``colormap`` is one of the names from :func:`list_colormaps`.
    """
    map_obj.add_colorbar(colormap=colormap, vmin=vmin, vmax=vmax)
    persist_map(map_obj, workspace)
    return {
        "status": "applied",
        "colormap": colormap,
        "min": vmin,
        "max": vmax,
    }


def fit_bounds(workspace: Workspace, map_obj: Map, bounds: list[float]) -> dict:
    """Fit the map camera and return confirmation of the resulting view."""
    map_obj.fit_project_bounds(bounds)
    persist_map(map_obj, workspace)
    return {
        "status": "applied",
        "mapView": deepcopy(map_obj.project.get("mapView")),
    }


def save_map(workspace: Workspace, map_obj: Map, path: str) -> Path:
    """Save the current project under ``maps/``; returns the written path."""
    out = workspace.resolve_under(workspace.maps, path)
    map_obj.save_project(str(out))
    return out


def export_html(
    workspace: Workspace, map_obj: Map, path: str, title: str = "GeoLibre Map"
) -> Path:
    """Export the map as a standalone HTML page under ``results/``."""
    out = workspace.resolve_under(workspace.results, path)
    map_obj.to_html(str(out), title=title)
    return out
