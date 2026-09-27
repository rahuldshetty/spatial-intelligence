"""Vector services: read, inspect, analyze, convert, and map layers.

Services are plain functions over a :class:`~spatial_intelligence.workspace.Workspace`
(and, for map layers, the live map). They never read the ambient runtime, so the
server and the agent path share one implementation. Writers return the written
:class:`~pathlib.Path`; recording the output in the manifest is the tool layer's
job.

The analysis tools mirror the semantics of the vector tools GeoLibre ships
(``packages/processing/src/vector-tools.ts``), so a result computed here matches
what the same named tool produces in the app — down to the units, which are the trap in this domain:
linear measurements (``buffer`` distance, ``simplify`` tolerance, ``points_along``
interval, ``grid`` cell size) are interpretable only against a stated unit, and
each function here says which one it uses.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.ops import triangulate, voronoi_diagram
from shapely.validation import explain_validity, make_valid

from ..contracts.errors import ToolInputError
from ..map import layers as layerops
from ..map import styles
from ..workspace import Workspace

#: Driver per output extension: what a writer may produce and nothing else.
_DRIVERS = {
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".gpkg": "GPKG",
    ".shp": "ESRI Shapefile",
}
#: Value operators ``select_by_value`` accepts (the upstream vocabulary).
VALUE_OPERATORS = (
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "starts-with",
    "is-null",
    "is-not-null",
)
#: Buffer units ``buffer`` accepts.
BUFFER_UNITS = ("meters", "degrees")
#: Spatial predicates ``select_by_location`` accepts.
LOCATION_PREDICATES = ("intersects", "within", "contains", "disjoint")
#: Join predicates ``spatial_join`` accepts, read as input → join layer.
JOIN_PREDICATES = ("intersects", "within", "contains")
#: Join kinds ``spatial_join`` accepts.
JOIN_KINDS = ("inner", "left")
#: Summary statistics ``aggregate`` accepts.
AGGREGATE_STATS = ("count", "sum", "mean", "min", "max", "median")
#: Overlay operations ``overlay`` accepts.
OVERLAY_OPERATIONS = ("intersection", "difference", "union")
#: Cell shapes ``grid`` accepts.
GRID_KINDS = ("rectangle", "hexagon")
#: Diagram kinds ``voronoi`` accepts.
DIAGRAM_KINDS = ("voronoi", "delaunay")
#: Most cells one ``grid`` call may produce.
MAX_GRID_CELLS = 200_000
#: Most points one ``points_along`` call may produce.
MAX_ALONG_POINTS = 200_000

__all__ = [
    "AGGREGATE_STATS",
    "add_vector_to_map",
    "add_heatmap_to_map",
    "aggregate",
    "attribute_join",
    "bounding_box",
    "buffer",
    "centroids",
    "check_geometry",
    "clip_vector",
    "convex_hull",
    "dissolve",
    "explode",
    "fix_geometry",
    "grid",
    "list_layers",
    "overlay",
    "points_along",
    "read_vector",
    "reproject_vector",
    "select_by_location",
    "select_by_value",
    "simplify",
    "spatial_join",
    "export_vector",
    "voronoi",
]


# -- reading, writing, and argument checks -------------------------------


def _read(workspace: Workspace, path: str, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read one vector layer from the workspace, confined and must-exist.

    A container with several layers and no ``layer`` is refused rather than
    resolved to the driver's first layer: silently analysing the wrong layer of
    a GeoPackage is a worse outcome than one error naming the layers to choose
    from.
    """
    source = str(workspace.resolve(path, must_exist=True))
    if layer is None:
        names = _layer_names(source)
        if len(names) > 1:
            raise ToolInputError(
                f"{path!r} holds {len(names)} layers ({', '.join(names)}); "
                "pass layer=... to choose one (list_layers names them)"
            )
    try:
        return gpd.read_file(source, layer=layer) if layer else gpd.read_file(source)
    except Exception as exc:  # a driver error is a model-visible input problem
        raise ToolInputError(f"could not read {path!r} as a vector layer: {exc}") from exc


def _layer_names(source: str) -> list[str]:
    """Return a dataset's layer names, or ``[]`` when they cannot be listed."""
    try:
        return [str(name) for name, _ in pyogrio.list_layers(source)]
    except Exception:
        return []


def _write(workspace: Workspace, out: str, gdf: gpd.GeoDataFrame) -> Path:
    """Write a layer under ``results/``, ``maps/``, or ``data/``."""
    target = workspace.resolve(out, write=True)
    driver = _DRIVERS.get(target.suffix.lower())
    if driver is None:
        supported = ", ".join(sorted(_DRIVERS))
        raise ToolInputError(f"unsupported output extension {target.suffix!r}; use {supported}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdf.to_file(target, driver=driver)
    except Exception as exc:
        raise ToolInputError(f"could not write {out!r} as {driver}: {exc}") from exc
    return target


def _require_columns(gdf: gpd.GeoDataFrame, names: Iterable[str]) -> None:
    """Fail with the available column names when one is missing."""
    missing = [name for name in names if name not in gdf.columns]
    if missing:
        available = ", ".join(str(column) for column in gdf.columns)
        raise ToolInputError(
            f"column(s) {missing} not in this layer; available columns: {available}"
        )


def _require_features(gdf: gpd.GeoDataFrame, path: str, what: str) -> None:
    """Refuse an empty layer where the result would be a summary of nothing.

    A tool that returns one geometry for the whole layer (a hull, a dissolve, a
    bounding box) has no honest answer for zero features: writing the empty
    geometry collection shapely produces would register a plausible-looking
    artifact that means "no data" only to someone who inspects it.
    """
    if gdf.empty:
        raise ToolInputError(f"{path!r} has no features to {what}")


def _number(value: float, name: str, *, positive: bool = False) -> float:
    """Return a finite ``value``, refusing what the arithmetic cannot take.

    ``positive`` demands strictly more than zero (a distance, a cell size); the
    default allows zero (a tolerance, an intensity).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolInputError(f"{name} must be a finite number, not {value!r}")
    if positive and value <= 0:
        raise ToolInputError(f"{name} must be greater than zero, not {value!r}")
    if not positive and value < 0:
        raise ToolInputError(f"{name} must be zero or more, not {value!r}")
    return float(value)


def _choice(value: str, name: str, allowed: tuple[str, ...]) -> str:
    """Return ``value`` when it is one of ``allowed``, naming them otherwise."""
    if value not in allowed:
        raise ToolInputError(
            f"{name} must be one of {', '.join(allowed)}, not {value!r}"
        )
    return value


def _require_crs(gdf: gpd.GeoDataFrame) -> Any:
    """Return the layer's CRS, refusing a layer that has none."""
    if gdf.crs is None:
        raise ToolInputError(
            "this layer has no CRS, so distance and area are meaningless for it; "
            "assign one or use reproject_vector on a source that has one"
        )
    return gdf.crs


def _metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return the layer in a metric CRS, for accurate distance/area work.

    A geographic layer is projected to its local UTM zone (the same choice the
    upstream tools make), so a buffer offset, a centroid, or a diagram cell is
    measured in real meters rather than degrees. A layer already in a projected
    CRS is returned unchanged.
    """
    crs = _require_crs(gdf)
    if not crs.is_geographic:
        return gdf
    try:
        return gdf.to_crs(gdf.estimate_utm_crs())
    except Exception as exc:
        raise ToolInputError(f"could not derive a metric CRS for this layer: {exc}") from exc


def _aligned(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, what: str) -> gpd.GeoDataFrame:
    """Return ``right`` reprojected onto ``left``'s CRS (a no-op when equal)."""
    _require_crs(left)
    _require_crs(right)
    if left.crs != right.crs:
        try:
            return right.to_crs(left.crs)
        except Exception as exc:
            raise ToolInputError(f"could not align the {what} layer to the input CRS: {exc}") from exc
    return right


def _finite_bounds(gdf: gpd.GeoDataFrame, what: str) -> list[float]:
    """Return the layer's finite bounds, refusing an empty/undefined extent."""
    bounds = [float(value) for value in gdf.total_bounds]
    if not all(math.isfinite(value) for value in bounds) or len(gdf) == 0:
        raise ToolInputError(f"{what} has no valid geometry to measure")
    return bounds


# -- inspection ----------------------------------------------------------


def list_layers(workspace: Workspace, path: str) -> dict:
    """List the layers inside a container dataset (GeoPackage, FileGDB, …).

    A GeoPackage routinely holds several layers, and ``read_vector`` reads only
    one, so this is how the layer to read or extract is chosen. Feature counts
    come from the container's own metadata and are omitted for a layer whose
    metadata cannot be read.
    """
    source = str(workspace.resolve(path, must_exist=True))
    try:
        raw = pyogrio.list_layers(source)
    except Exception as exc:
        raise ToolInputError(f"could not list layers in {path!r}: {exc}") from exc
    layers: list[dict] = []
    for name, geometry_type in raw:
        entry: dict[str, Any] = {
            "layer": str(name),
            "geometry_type": str(geometry_type) if geometry_type is not None else None,
        }
        try:
            entry["features"] = int(pyogrio.read_info(source, layer=str(name))["features"])
        except Exception:
            entry["features"] = None
        layers.append(entry)
    return {"path": path, "layers": layers}


def read_vector(workspace: Workspace, path: str, layer: str | None = None) -> dict:
    """Return CRS, columns, row count, bounds and geometry types."""
    gdf = _read(workspace, path, layer)
    if getattr(gdf, "geometry", None) is None:
        raise ToolInputError(
            f"{path!r} holds no geometry, so it is a table rather than a layer; "
            "join it onto a layer with attribute_join, or read it in run_python"
        )
    return {
        "crs": str(gdf.crs),
        "columns": list(gdf.columns),
        "len": int(len(gdf)),
        "bounds": [float(x) for x in gdf.total_bounds],
        "geom_types": [str(t) for t in gdf.geom_type.unique().tolist()],
        "layer": layer,
    }


def check_geometry(workspace: Workspace, path: str, layer: str | None = None) -> dict:
    """Report invalid and empty geometries with the reason GEOS gives.

    Every overlay, dissolve, and buffer downstream assumes valid geometry, so
    this is the check to run before one of them fails for a reason that reads
    like a driver bug.
    """
    gdf = _read(workspace, path, layer)
    details: list[dict] = []
    empty = 0
    for index, geometry in enumerate(gdf.geometry):
        if geometry is None or geometry.is_empty:
            empty += 1
            continue
        if geometry.is_valid:
            continue
        details.append({"index": int(index), "reason": explain_validity(geometry)})
    return {
        "len": int(len(gdf)),
        "invalid": len(details),
        "empty": empty,
        "details": details[:50],
        "valid": not details and not empty,
    }


# -- single-layer tools --------------------------------------------------


def reproject_vector(
    workspace: Workspace, path: str, out: str, dst_crs: str, layer: str | None = None
) -> Path:
    """Reproject a vector dataset; returns the path of the written output."""
    gdf = _read(workspace, path, layer)
    try:
        result = gdf.to_crs(dst_crs)
    except Exception as exc:
        raise ToolInputError(f"could not reproject to {dst_crs!r}: {exc}") from exc
    return _write(workspace, out, result)


def buffer(
    workspace: Workspace,
    path: str,
    out: str,
    distance: float,
    unit: str = "meters",
    dissolve: bool = False,
    layer: str | None = None,
) -> Path:
    """Buffer geometries by ``distance`` (metric units use the UTM CRS).

    ``unit`` is ``meters`` (the default; the distance is applied in a metric
    projection) or ``degrees`` (applied in the layer's own units, for a layer
    whose data is really in degrees). With ``dissolve`` set, the buffers are
    merged into a single feature with their overlaps dissolved away.
    """
    _number(distance, "distance")
    _choice(unit, "unit", BUFFER_UNITS)
    gdf = _read(workspace, path, layer)
    if unit == "meters":
        projected = _metric(gdf)
        result = projected.copy()
        result["geometry"] = projected.geometry.buffer(distance)
        result = result.to_crs(gdf.crs)
    else:
        result = gdf.copy()
        result["geometry"] = result.geometry.buffer(distance)
    if dissolve:
        result = gpd.GeoDataFrame(geometry=[result.geometry.union_all()], crs=result.crs)
    return _write(workspace, out, result)


def clip_vector(
    workspace: Workspace,
    path: str,
    out: str,
    mask: str,
    layer: str | None = None,
    mask_layer: str | None = None,
) -> Path:
    """Clip a vector dataset to a mask polygon layer."""
    gdf = _read(workspace, path, layer)
    mask_gdf = _aligned(gdf, _read(workspace, mask, mask_layer), "mask")
    try:
        result = gpd.clip(gdf, mask_gdf)
    except Exception as exc:
        raise ToolInputError(f"clip failed: {exc}") from exc
    return _write(workspace, out, result)


def dissolve(
    workspace: Workspace, path: str, out: str, column: str | None = None, layer: str | None = None
) -> Path:
    """Merge features into one geometry, or one per distinct ``column`` value.

    Grouping keeps the other attribute columns from the first feature of each
    group (GeoPandas' default), so treat them as representative rather than
    summed: ``aggregate`` is the tool for per-group statistics.
    """
    gdf = _read(workspace, path, layer)
    _require_features(gdf, path, "dissolve")
    try:
        if column:
            _require_columns(gdf, [column])
            result = gdf.dissolve(by=column, as_index=False)
        else:
            result = gpd.GeoDataFrame(geometry=[gdf.geometry.union_all()], crs=gdf.crs)
    except Exception as exc:
        raise ToolInputError(
            f"dissolve failed: {exc} (invalid geometry is the usual cause; "
            "run check_geometry, then fix_geometry)"
        ) from exc
    return _write(workspace, out, result)


def centroids(workspace: Workspace, path: str, out: str, layer: str | None = None) -> Path:
    """Replace each feature with its centroid (computed in a metric CRS)."""
    gdf = _read(workspace, path, layer)
    projected = _metric(gdf)
    result = projected.copy()
    result["geometry"] = projected.geometry.centroid
    return _write(workspace, out, result.to_crs(gdf.crs))


def convex_hull(workspace: Workspace, path: str, out: str, layer: str | None = None) -> Path:
    """Compute the single convex hull enclosing every feature."""
    gdf = _read(workspace, path, layer)
    _require_features(gdf, path, "put a hull around")
    hull = gdf.geometry.union_all().convex_hull
    result = gpd.GeoDataFrame(geometry=[hull], crs=gdf.crs)
    return _write(workspace, out, result)


def bounding_box(workspace: Workspace, path: str, out: str, layer: str | None = None) -> Path:
    """Compute the axis-aligned bounding box of every feature as one polygon."""
    gdf = _read(workspace, path, layer)
    minx, miny, maxx, maxy = _finite_bounds(gdf, "the layer")
    result = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=gdf.crs)
    return _write(workspace, out, result)


def simplify(
    workspace: Workspace,
    path: str,
    out: str,
    tolerance: float = 0.001,
    preserve_topology: bool = True,
    layer: str | None = None,
) -> Path:
    """Reduce vertices with Douglas-Peucker; ``tolerance`` is in layer units.

    For a layer in ``EPSG:4326`` that unit is degrees, so the useful magnitudes
    are small (0.0001 ≈ 11 m at the equator). A projected layer's tolerance is
    in that CRS's units, which is usually what you want for real simplification.
    """
    _number(tolerance, "tolerance")
    gdf = _read(workspace, path, layer)
    result = gdf.copy()
    result["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=preserve_topology)
    return _write(workspace, out, result)


def explode(workspace: Workspace, path: str, out: str, layer: str | None = None) -> Path:
    """Split multipart geometries into one feature per part, keeping attributes."""
    gdf = _read(workspace, path, layer)
    result = gdf.explode(index_parts=False).reset_index(drop=True)
    return _write(workspace, out, result)


def fix_geometry(workspace: Workspace, path: str, out: str, layer: str | None = None) -> Path:
    """Repair invalid geometries with ``make_valid``; valid ones pass through."""
    gdf = _read(workspace, path, layer)
    result = gdf.copy()
    repaired: list[Any] = []
    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty or geometry.is_valid:
            repaired.append(geometry)
            continue
        try:
            fixed = make_valid(geometry)
        except Exception:  # keep the original rather than losing the feature
            repaired.append(geometry)
            continue
        repaired.append(geometry if fixed is None or fixed.is_empty else fixed)
    result["geometry"] = repaired
    return _write(workspace, out, result)


def aggregate(
    workspace: Workspace,
    path: str,
    out: str,
    column: str,
    statistic: str = "count",
    field: str | None = None,
    layer: str | None = None,
) -> Path:
    """Merge polygons by ``column`` and attach one summary statistic per group.

    ``statistic`` is ``count`` (features per group) or a reduction of the numeric
    ``field``: ``sum``, ``mean``, ``min``, ``max``, ``median``. The output carries
    the group column and one statistic column, named ``count`` or
    ``<field>_<statistic>``. Polygon layers only, as upstream.
    """
    _choice(statistic, "statistic", AGGREGATE_STATS)
    if statistic != "count" and not field:
        raise ToolInputError(f"statistic {statistic!r} needs a numeric field to reduce")
    gdf = _read(workspace, path, layer)
    _require_columns(gdf, [column])
    polygons = gdf[gdf.geometry.geom_type.isin({"Polygon", "MultiPolygon"})]
    if polygons.empty:
        raise ToolInputError("aggregate merges polygons; this layer has none")
    if statistic != "count":
        _require_columns(gdf, [field])
    try:
        # Carry only the group column into the merge: GeoPandas' default would
        # keep every other attribute from an arbitrary "first" feature of each
        # group, which reads as data and is not.
        merged = polygons[[column, polygons.geometry.name]].dissolve(by=column)
        if statistic == "count":
            values = polygons.groupby(column).size().rename("count")
        else:
            values = polygons.groupby(column)[field].agg(statistic).rename(f"{field}_{statistic}")
        result = merged.join(values).reset_index()
    except Exception as exc:
        raise ToolInputError(f"aggregate failed: {exc}") from exc
    return _write(workspace, out, result)


def voronoi(
    workspace: Workspace,
    path: str,
    out: str,
    kind: str = "voronoi",
    layer: str | None = None,
) -> Path:
    """Build a Voronoi diagram (polygons) or Delaunay triangulation from points.

    Both are computed in a metric CRS and returned in the input CRS, so cells are
    not stretched by degrees. A Voronoi diagram is clipped to the points' extent,
    matching the upstream tool. Needs at least three points that are neither
    collinear nor coincident.
    """
    _choice(kind, "kind", DIAGRAM_KINDS)
    gdf = _read(workspace, path, layer)
    projected = _metric(gdf)
    points: list[Point] = []
    for geometry in projected.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "Point":
            points.append(geometry)
        elif geometry.geom_type == "MultiPoint":
            points.extend(list(geometry.geoms))
    if len(points) < 3:
        raise ToolInputError("a Voronoi diagram or Delaunay triangulation needs at least 3 points")
    multipoint = MultiPoint(points)
    minx, miny, maxx, maxy = multipoint.bounds
    if minx == maxx or miny == maxy:
        raise ToolInputError("the points are collinear or coincident; spread them out first")
    if kind == "delaunay":
        geometries = [triangle for triangle in triangulate(multipoint)]
    else:
        diagram = voronoi_diagram(multipoint, envelope=box(minx, miny, maxx, maxy))
        geometries = list(diagram.geoms)
    result = gpd.GeoDataFrame(geometry=geometries, crs=projected.crs).to_crs(gdf.crs)
    return _write(workspace, out, result)


def points_along(
    workspace: Workspace,
    path: str,
    out: str,
    interval: float,
    layer: str | None = None,
) -> Path:
    """Place points every ``interval`` along lines and polygon boundaries.

    ``interval`` is in layer units — degrees for ``EPSG:4326`` — and is applied
    from the start of each line, plus the start of a polygon's exterior ring. The
    parent feature's attributes are carried onto every point it produces. At most
    ``200000`` points may be placed in one call, and a smaller interval is refused
    with the smallest one that fits rather than returning a truncated layer.
    """
    _number(interval, "interval", positive=True)
    gdf = _read(workspace, path, layer)
    # Total length is cheap (no interpolation), so an interval that cannot fit the
    # budget is refused before a single point is built.
    total = sum(
        line.length for geometry in gdf.geometry for line in _lines_of(geometry)
    )
    if total / interval + 1 > MAX_ALONG_POINTS:
        raise ToolInputError(
            f"interval {interval} would place more than {MAX_ALONG_POINTS} points "
            f"along {total:g} of line; use an interval of at least "
            f"{total / (MAX_ALONG_POINTS - 1):g}"
        )
    records: list[dict[str, Any]] = []
    geometries: list[Point] = []
    for _, row in gdf.iterrows():
        attributes = row.drop(labels="geometry").to_dict()
        for line in _lines_of(row.geometry):
            length = line.length
            distance = 0.0
            while distance <= length:
                if len(geometries) >= MAX_ALONG_POINTS:
                    # Truncating here would return a plausible half a layer, so the
                    # cap is an error the caller can act on, like grid's.
                    raise ToolInputError(
                        f"interval {interval} would place more than "
                        f"{MAX_ALONG_POINTS} points; use a larger interval"
                    )
                geometries.append(line.interpolate(distance))
                records.append(attributes)
                distance += interval
    if not geometries:
        raise ToolInputError("no lines or polygon boundaries found to place points along")
    result = gpd.GeoDataFrame(records, geometry=geometries, crs=gdf.crs)
    return _write(workspace, out, result)


def grid(
    workspace: Workspace,
    out: str,
    path: str | None = None,
    bounds: list[float] | None = None,
    layer: str | None = None,
    cell_width: float = 0.01,
    cell_height: float | None = None,
    kind: str = "rectangle",
) -> Path:
    """Generate a regular grid over ``bounds`` or the extent of ``path``.

    ``cell_width`` and ``cell_height`` are in the units of ``bounds``/the layer —
    degrees for a ``EPSG:4326`` layer — and ``cell_height`` defaults to
    ``cell_width``. ``kind`` is ``rectangle`` (a fishnet) or ``hexagon``, where
    ``cell_width`` is the across-corners width and the cells tile exactly. Pass
    ``bounds`` as ``[west, south, east, north]``, or ``path`` — plus ``layer`` for
    one layer of a container — to grid that layer's extent; one of the two is
    required.
    """
    _choice(kind, "kind", GRID_KINDS)
    _number(cell_width, "cell_width", positive=True)
    height = cell_width if cell_height is None else cell_height
    _number(height, "cell_height", positive=True)
    if bounds is None:
        if path is None:
            raise ToolInputError(
                "pass bounds=[west, south, east, north], or path to a layer to grid"
            )
        source = _read(workspace, path, layer)
        bounds = _finite_bounds(source, "the layer to grid")
        crs: Any = source.crs
    else:
        if len(bounds) != 4 or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ToolInputError("bounds must be [west, south, east, north]")
        for value in bounds:
            _number(value, "bounds entry")
        crs = "EPSG:4326"
    geometries = _grid_cells(kind, [float(value) for value in bounds], cell_width, height)
    result = gpd.GeoDataFrame(geometry=geometries, crs=crs)
    return _write(workspace, out, result)


def _grid_cells(
    kind: str, bounds: list[float], cell_width: float, cell_height: float
) -> list[Polygon]:
    """Build the cells of a rectangle or hexagon grid over ``bounds``."""
    west, south, east, north = bounds
    if kind == "rectangle":
        columns = math.ceil((east - west) / cell_width)
        rows = math.ceil((north - south) / cell_height)
        count = columns * rows
        if count > MAX_GRID_CELLS:
            raise ToolInputError(
                f"that cell size would create {count} cells (limit {MAX_GRID_CELLS}); "
                "use a larger cell_width/cell_height"
            )
        return [
            box(west + column * cell_width, south + row * cell_height,
                min(west + (column + 1) * cell_width, east),
                min(south + (row + 1) * cell_height, north))
            for row in range(rows)
            for column in range(columns)
        ]
    # Hexagons: circumradius half the across-corners width, columns 1.5·r apart
    # and alternate columns offset by half a row, which tiles without gaps.
    radius = cell_width / 2
    column_step = 1.5 * radius
    row_step = math.sqrt(3) * radius
    columns = math.ceil((east - west) / column_step) + 1
    rows = math.ceil((north - south) / row_step) + 1
    count = columns * rows
    if count > MAX_GRID_CELLS:
        raise ToolInputError(
            f"that cell size would create {count} cells (limit {MAX_GRID_CELLS}); "
            "use a larger cell_width"
        )
    rings = [
        (radius * math.cos(math.pi / 3 * corner), radius * math.sin(math.pi / 3 * corner))
        for corner in range(6)
    ]
    cells: list[Polygon] = []
    for column in range(columns):
        for row in range(rows):
            offset = row_step / 2 if column % 2 else 0.0
            center_x = west + column * column_step
            center_y = south + row * row_step + offset
            cell = Polygon([(center_x + x, center_y + y) for x, y in rings])
            if cell.bounds[2] < west or cell.bounds[0] > east:
                continue
            if cell.bounds[3] < south or cell.bounds[1] > north:
                continue
            cells.append(cell)
    return cells


def _lines_of(geometry: Any) -> list[Any]:
    """Return the lines a geometry contributes: itself, or its boundaries."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "Polygon":
        return [geometry.exterior, *list(geometry.interiors)]
    if geometry.geom_type == "MultiPolygon":
        return [line for part in geometry.geoms for line in _lines_of(part)]
    return []


# -- two-layer tools -----------------------------------------------------


def overlay(
    workspace: Workspace,
    path: str,
    out: str,
    overlay: str,
    operation: str = "intersection",
    layer: str | None = None,
    overlay_layer: str | None = None,
) -> Path:
    """Combine two polygon layers with ``intersection``, ``difference``, or ``union``.

    ``intersection`` keeps the areas both layers cover and carries attributes
    from both; ``difference`` keeps the input layer's area outside the overlay
    layer (input attributes only, polygonal output only); ``union`` returns one
    combined geometry with no attributes, matching the upstream tool.
    """
    _choice(operation, "operation", OVERLAY_OPERATIONS)
    left = _read(workspace, path, layer)
    right = _aligned(left, _read(workspace, overlay, overlay_layer), "overlay")
    try:
        if operation == "union":
            merged = shapely.union_all([left.geometry.union_all(), right.geometry.union_all()])
            result = gpd.GeoDataFrame(geometry=[merged], crs=left.crs)
        else:
            # Both operations are documented to return area, so the line and point
            # rows an edge-touching overlay produces are dropped rather than mixed
            # into a polygon layer.
            result = gpd.overlay(left, right, how=operation, keep_geom_type=True)
    except Exception as exc:
        raise ToolInputError(
            f"{operation} failed: {exc} (both layers must be polygonal; "
            "run check_geometry, then fix_geometry, on invalid geometry)"
        ) from exc
    return _write(workspace, out, result)


def spatial_join(
    workspace: Workspace,
    path: str,
    out: str,
    join: str,
    predicate: str = "intersects",
    how: str = "inner",
    layer: str | None = None,
    join_layer: str | None = None,
) -> Path:
    """Attach ``join`` layer attributes to each input feature by a spatial test.

    ``predicate`` is ``intersects``, ``within``, or ``contains``, read as input →
    join layer. ``how`` is ``inner`` (keep only matches; a feature matching two
    join features is duplicated, one row each) or ``left`` (keep every input
    feature, unmatched ones with empty join attributes).
    """
    _choice(predicate, "predicate", JOIN_PREDICATES)
    _choice(how, "how", JOIN_KINDS)
    left = _read(workspace, path, layer)
    right = _aligned(left, _read(workspace, join, join_layer), "join")
    try:
        joined = gpd.sjoin(left, right, predicate=predicate, how=how)
    except Exception as exc:
        raise ToolInputError(f"spatial join failed: {exc}") from exc
    return _write(workspace, out, joined.drop(columns=["index_right"], errors="ignore"))


def attribute_join(
    workspace: Workspace,
    path: str,
    out: str,
    join: str,
    field: str,
    join_field: str | None = None,
    columns: list[str] | None = None,
    layer: str | None = None,
    join_layer: str | None = None,
) -> Path:
    """Attach a table's columns onto the layer by matching key values.

    ``field`` is the key in the input layer and ``join_field`` the key in the
    joined table (defaults to the same name). ``columns`` limits which of the
    table's columns are carried over — every one except the key by default. Rows
    with no matching key are kept with empty joined values, so a partial match
    stays visible instead of silently dropping features.
    """
    left = _read(workspace, path, layer)
    _require_columns(left, [field])
    right = _read(workspace, join, join_layer)
    right_key = join_field or field
    _require_columns(right, [right_key])
    # A joined table need not be a layer: ``_read`` returns a plain DataFrame for
    # a CSV, which has no geometry column to leave behind.
    geometry_name = getattr(getattr(right, "geometry", None), "name", None)
    keep = (
        list(columns)
        if columns
        else [name for name in right.columns if name != right_key and name != geometry_name]
    )
    _require_columns(right, keep)
    subset = right[[right_key, *keep]] if keep else right[[right_key]]
    try:
        merged = left.merge(subset, left_on=field, right_on=right_key, how="left")
    except Exception as exc:
        raise ToolInputError(f"attribute join failed: {exc}") from exc
    return _write(workspace, out, merged)


def select_by_value(
    workspace: Workspace,
    path: str,
    out: str,
    column: str,
    operator: str,
    value: str | None = None,
    layer: str | None = None,
) -> Path:
    """Keep the features whose ``column`` satisfies ``operator`` against ``value``.

    Operators: ``eq``, ``neq``, ``gt``, ``gte``, ``lt``, ``lte``, ``contains``,
    ``starts-with``, ``is-null``, ``is-not-null``. Ordering comparisons are
    numeric when the value and the cell are both numbers and string-based
    otherwise, and an empty cell matches only the two null operators.
    """
    _choice(operator, "operator", VALUE_OPERATORS)
    if operator not in ("is-null", "is-not-null") and value is None:
        raise ToolInputError(f"operator {operator!r} needs a value")
    gdf = _read(workspace, path, layer)
    _require_columns(gdf, [column])
    keep = [_matches(cell, operator, "" if value is None else str(value)) for cell in gdf[column]]
    return _write(workspace, out, gdf[np.array(keep, dtype=bool)])


def _matches(cell: Any, operator: str, raw: str) -> bool:
    """Evaluate one cell against an operator and the caller's text."""
    empty = cell is None or (isinstance(cell, float) and math.isnan(cell)) or str(cell) == ""
    if operator == "is-null":
        return empty
    if operator == "is-not-null":
        return not empty
    if empty:
        return False
    text = str(cell)
    if operator in ("eq", "neq"):
        matched = _as_number(text) == _as_number(raw) if _both_numbers(text, raw) else text == raw
        return matched if operator == "eq" else not matched
    if operator == "contains":
        return raw in text
    if operator == "starts-with":
        return text.startswith(raw)
    if _both_numbers(text, raw):
        left_value, right_value = float(text), float(raw)
    else:
        left_value, right_value = text, raw
    if operator == "gt":
        return left_value > right_value
    if operator == "gte":
        return left_value >= right_value
    if operator == "lt":
        return left_value < right_value
    return left_value <= right_value


def _both_numbers(*values: str) -> bool:
    """Whether every value parses as a finite number."""
    return all(_as_number(value) is not None for value in values)


def _as_number(value: str) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not one."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def select_by_location(
    workspace: Workspace,
    path: str,
    out: str,
    mask: str,
    predicate: str = "intersects",
    layer: str | None = None,
    mask_layer: str | None = None,
) -> Path:
    """Keep the features whose geometry satisfies ``predicate`` against ``mask``.

    ``predicate`` is ``intersects``, ``within`` (input inside the mask),
    ``contains`` (input contains the mask), or ``disjoint``. The mask's features
    are unioned first, so any overlap with any mask feature counts once.
    """
    _choice(predicate, "predicate", LOCATION_PREDICATES)
    gdf = _read(workspace, path, layer)
    mask_gdf = _aligned(gdf, _read(workspace, mask, mask_layer), "mask")
    if mask_gdf.empty:
        raise ToolInputError("the mask layer has no features")
    combined = mask_gdf.geometry.union_all()
    # A null or empty geometry matches nothing: GeoJSON allows it, and check_geometry
    # reports it, so failing the whole call on one is not an option.
    keep = np.array(
        [
            bool(getattr(geometry, predicate)(combined))
            if geometry is not None and not geometry.is_empty
            else False
            for geometry in gdf.geometry
        ],
        dtype=bool,
    )
    return _write(workspace, out, gdf[keep])


# -- conversion and map --------------------------------------------------


def export_vector(
    workspace: Workspace, path: str, out: str, layer: str | None = None
) -> Path:
    """Write one vector layer to the format the output extension names.

    ``.geojson``/``.json``, ``.gpkg``, and ``.shp`` are supported, so the same
    call both converts to GeoJSON and packages a layer as a GeoPackage. ``layer``
    selects a layer of a multi-layer container (see :func:`list_layers`).
    """
    gdf = _read(workspace, path, layer)
    return _write(workspace, out, gdf)


def add_heatmap_to_map(
    workspace: Workspace,
    map_obj: Any,
    path: str,
    name: str,
    radius: float = 30.0,
    intensity: float = 1.0,
    layer: str | None = None,
) -> str:
    """Add a point dataset to the live map as a heatmap; returns the layer id.

    ``radius`` is the kernel size in pixels and ``intensity`` its weight, both
    passed to GeoLibre's heatmap layer. The points are read here (so a GeoPackage
    layer or a CSV of coordinates works) and handed to the map as a GeoDataFrame.
    """
    _number(radius, "radius", positive=True)
    _number(intensity, "intensity")
    gdf = _read(workspace, path, layer)
    if not set(gdf.geom_type.unique().tolist()) <= {"Point", "MultiPoint"}:
        raise ToolInputError(
            "a heatmap needs point geometry; use centroids or points_along first, "
            "or add the layer as a polygon layer instead"
        )
    layer_id = map_obj.add_heatmap(
        gdf, layerops.clean_layer_name(name), radius=radius, intensity=intensity
    )
    # GeoLibre writes the heatmap renderer onto layer.style; the app reads the
    # project-level copy, so both have to agree (see map/styles.py).
    styles.mirror(map_obj.project, layer_id)
    return layer_id


def add_vector_to_map(
    workspace: Workspace,
    map_obj: Any,
    path: str,
    name: str,
    column: str | None = None,
    palette: str = "viridis",
    layer: str | None = None,
) -> str:
    """Load a vector dataset and add it to the live map; returns the layer id.

    A ``column`` choropleth is computed by geolibre, which writes only the
    layer's own style; the result is mirrored into the project-level style map
    GeoLibre's app actually reads (see :mod:`spatial_intelligence.map.styles`).
    """
    name = layerops.clean_layer_name(name)
    gdf = _read(workspace, path, layer)
    if column:
        layer_id = map_obj.add_gdf(gdf, name, column=column, colormap=palette)
        styles.mirror(map_obj.project, layer_id)
        return layer_id
    return map_obj.add_gdf(gdf, name)
