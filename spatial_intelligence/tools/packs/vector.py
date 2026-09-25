"""Vector tools backed by GeoPandas. Outputs land under ``results/``."""

from __future__ import annotations

from ...contracts.effects import Effect
from ...geo import vector
from ...map import document
from ..runtime import ToolRuntime
from ..spec import pack, tool


@pack(category="vector", effects=frozenset({Effect.WORKSPACE_WRITE}))
class VectorPack:
    """GeoPandas vector tools; every written output is recorded in the manifest.

    The analysis tools mirror the semantics of GeoLibre's own vector toolbox, so
    ``dissolve``, ``overlay``, ``spatial_join``, and the rest produce what the
    same named tool produces in the app. Units differ per tool and are stated in
    each docstring: metric tools measure in meters via a local projection, and
    the ones that speak in layer units say so.
    """

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime

    # -- inspection --------------------------------------------------------

    @tool(effects=frozenset({Effect.READ}))
    def list_layers(self, path: str) -> dict:
        """List the layers inside a container dataset (GeoPackage, FileGDB).

        Call this before ``read_vector`` or ``export_vector`` on a GeoPackage: those
        read one layer at a time and need its exact name.
        """
        return vector.list_layers(self._rt.workspace, path)

    @tool(effects=frozenset({Effect.READ}))
    def read_vector(self, path: str, layer: str | None = None) -> dict:
        """Return CRS, columns, row count, bounds and geometry types.

        ``layer`` selects one layer of a container dataset (see ``list_layers``);
        omit it for a single-layer file.
        """
        return vector.read_vector(self._rt.workspace, path, layer)

    @tool(effects=frozenset({Effect.READ}))
    def check_geometry(self, path: str, layer: str | None = None) -> dict:
        """Report invalid and empty geometries, with the reason GEOS gives.

        Run this before a dissolve, overlay, or buffer that fails: invalid
        geometry is the usual cause, and ``fix_geometry`` repairs what it finds.
        """
        return vector.check_geometry(self._rt.workspace, path, layer)

    # -- writers -----------------------------------------------------------

    @tool()
    def reproject_vector(self, path: str, out: str, dst_crs: str, layer: str | None = None) -> str:
        """Reproject a vector dataset; returns the absolute output path."""
        written = vector.reproject_vector(self._rt.workspace, path, out, dst_crs, layer)
        return self._rt.record_artifact(written)

    @tool()
    def buffer(
        self,
        path: str,
        out: str,
        distance: float,
        unit: str = "meters",
        dissolve: bool = False,
        layer: str | None = None,
    ) -> str:
        """Buffer geometries by ``distance`` (metric units use the UTM CRS).

        ``unit`` is ``meters`` (default, applied in a local metric projection) or
        ``degrees``. With ``dissolve`` set, the buffers are merged into a single
        feature with their overlaps removed.
        """
        written = vector.buffer(self._rt.workspace, path, out, distance, unit, dissolve, layer)
        return self._rt.record_artifact(written)

    @tool()
    def clip_vector(
        self,
        path: str,
        out: str,
        mask: str,
        layer: str | None = None,
        mask_layer: str | None = None,
    ) -> str:
        """Clip a vector dataset to a mask polygon layer.

        Pass ``layer``/``mask_layer`` when either dataset is one layer of a
        container such as a GeoPackage.
        """
        written = vector.clip_vector(self._rt.workspace, path, out, mask, layer, mask_layer)
        return self._rt.record_artifact(written)

    @tool()
    def dissolve(self, path: str, out: str, column: str | None = None, layer: str | None = None) -> str:
        """Merge features into one geometry, or one per distinct ``column`` value."""
        written = vector.dissolve(self._rt.workspace, path, out, column, layer)
        return self._rt.record_artifact(written)

    @tool()
    def overlay(
        self,
        path: str,
        out: str,
        overlay: str,
        operation: str = "intersection",
        layer: str | None = None,
        overlay_layer: str | None = None,
    ) -> str:
        """Combine two polygon layers: ``intersection``, ``difference``, or ``union``.

        ``intersection`` keeps the area both layers cover and carries both sets of
        attributes. ``difference`` keeps the input's area outside the overlay layer
        (input attributes, polygonal output only). ``union`` returns one combined
        geometry with no attributes. Pass ``layer``/``overlay_layer`` when either
        dataset is one layer of a container such as a GeoPackage.
        """
        written = vector.overlay(
            self._rt.workspace, path, out, overlay, operation, layer, overlay_layer
        )
        return self._rt.record_artifact(written)

    @tool()
    def spatial_join(
        self,
        path: str,
        out: str,
        join: str,
        predicate: str = "intersects",
        how: str = "inner",
        layer: str | None = None,
        join_layer: str | None = None,
    ) -> str:
        """Attach the ``join`` layer's attributes by a spatial test.

        ``predicate`` is ``intersects``, ``within``, or ``contains`` (input →
        join layer); ``how`` is ``inner`` or ``left``. Pass
        ``layer``/``join_layer`` when either dataset is one layer of a container.
        """
        written = vector.spatial_join(
            self._rt.workspace, path, out, join, predicate, how, layer, join_layer
        )
        return self._rt.record_artifact(written)

    @tool()
    def attribute_join(
        self,
        path: str,
        out: str,
        join: str,
        field: str,
        join_field: str | None = None,
        columns: list[str] | None = None,
        layer: str | None = None,
        join_layer: str | None = None,
    ) -> str:
        """Attach a table's columns onto the layer by matching key values.

        ``field`` is the key in the input layer, ``join_field`` the key in the
        joined table (defaults to the same name), and ``columns`` limits what is
        carried over. Unmatched rows are kept with empty joined values. Pass
        ``layer``/``join_layer`` when either dataset is one layer of a container.
        """
        written = vector.attribute_join(
            self._rt.workspace, path, out, join, field, join_field, columns, layer, join_layer
        )
        return self._rt.record_artifact(written)

    @tool()
    def select_by_value(
        self,
        path: str,
        out: str,
        column: str,
        operator: str,
        value: str | None = None,
        layer: str | None = None,
    ) -> str:
        """Keep the features whose ``column`` satisfies ``operator`` against ``value``.

        Operators: ``eq``, ``neq``, ``gt``, ``gte``, ``lt``, ``lte``, ``contains``,
        ``starts-with``, ``is-null``, ``is-not-null``. Ordering comparisons are
        numeric when both sides are numbers, string-based otherwise.
        """
        written = vector.select_by_value(self._rt.workspace, path, out, column, operator, value, layer)
        return self._rt.record_artifact(written)

    @tool()
    def select_by_location(
        self,
        path: str,
        out: str,
        mask: str,
        predicate: str = "intersects",
        layer: str | None = None,
        mask_layer: str | None = None,
    ) -> str:
        """Keep the features that relate to a mask layer by ``predicate``.

        ``predicate`` is ``intersects``, ``within``, ``contains``, or ``disjoint``;
        the mask's features are unioned first, so any overlap counts once. Pass
        ``layer``/``mask_layer`` when either dataset is one layer of a container.
        """
        written = vector.select_by_location(
            self._rt.workspace, path, out, mask, predicate, layer, mask_layer
        )
        return self._rt.record_artifact(written)

    @tool()
    def centroids(self, path: str, out: str, layer: str | None = None) -> str:
        """Replace each feature with its centroid (computed in a metric CRS)."""
        written = vector.centroids(self._rt.workspace, path, out, layer)
        return self._rt.record_artifact(written)

    @tool()
    def convex_hull(self, path: str, out: str, layer: str | None = None) -> str:
        """Compute the single convex hull enclosing every feature."""
        written = vector.convex_hull(self._rt.workspace, path, out, layer)
        return self._rt.record_artifact(written)

    @tool()
    def bounding_box(self, path: str, out: str, layer: str | None = None) -> str:
        """Compute the axis-aligned bounding box of every feature as one polygon."""
        written = vector.bounding_box(self._rt.workspace, path, out, layer)
        return self._rt.record_artifact(written)

    @tool()
    def simplify(
        self,
        path: str,
        out: str,
        tolerance: float = 0.001,
        preserve_topology: bool = True,
        layer: str | None = None,
    ) -> str:
        """Reduce vertices with Douglas-Peucker; ``tolerance`` is in layer units.

        For an ``EPSG:4326`` layer that is degrees (0.0001 ≈ 11 m at the equator),
        so reproject to a projected CRS first when the tolerance should mean meters.
        """
        written = vector.simplify(self._rt.workspace, path, out, tolerance, preserve_topology, layer)
        return self._rt.record_artifact(written)

    @tool()
    def explode(self, path: str, out: str, layer: str | None = None) -> str:
        """Split multipart geometries into one feature per part, keeping attributes."""
        written = vector.explode(self._rt.workspace, path, out, layer)
        return self._rt.record_artifact(written)

    @tool()
    def fix_geometry(self, path: str, out: str, layer: str | None = None) -> str:
        """Repair invalid geometries with ``make_valid``; valid ones pass through."""
        written = vector.fix_geometry(self._rt.workspace, path, out, layer)
        return self._rt.record_artifact(written)

    @tool()
    def aggregate(
        self,
        path: str,
        out: str,
        column: str,
        statistic: str = "count",
        field: str | None = None,
        layer: str | None = None,
    ) -> str:
        """Merge polygons by ``column`` and attach one summary statistic per group.

        ``statistic`` is ``count``, or a reduction of the numeric ``field``:
        ``sum``, ``mean``, ``min``, ``max``, ``median``. The output holds the group
        column plus one column named ``count`` or ``<field>_<statistic>``.
        """
        written = vector.aggregate(self._rt.workspace, path, out, column, statistic, field, layer)
        return self._rt.record_artifact(written)

    @tool()
    def voronoi(self, path: str, out: str, kind: str = "voronoi", layer: str | None = None) -> str:
        """Build a Voronoi diagram or Delaunay triangulation from a point layer.

        The cells are computed in a metric CRS and clipped to the points' extent.
        Needs at least three points that are not collinear.
        """
        written = vector.voronoi(self._rt.workspace, path, out, kind, layer)
        return self._rt.record_artifact(written)

    @tool()
    def points_along(
        self, path: str, out: str, interval: float, layer: str | None = None
    ) -> str:
        """Place points every ``interval`` along lines and polygon boundaries.

        ``interval`` is in layer units — degrees for ``EPSG:4326`` — measured from
        the start of each line and of each polygon ring.
        """
        written = vector.points_along(self._rt.workspace, path, out, interval, layer)
        return self._rt.record_artifact(written)

    @tool()
    def grid(
        self,
        out: str,
        path: str | None = None,
        bounds: list[float] | None = None,
        layer: str | None = None,
        cell_width: float = 0.01,
        cell_height: float | None = None,
        kind: str = "rectangle",
    ) -> str:
        """Generate a regular grid over ``bounds`` or the extent of ``path``.

        Pass ``bounds`` as ``[west, south, east, north]``, or ``path`` (plus
        ``layer`` for one layer of a container such as a GeoPackage) to grid that
        layer's extent; one of the two is required. ``cell_width``/``cell_height``
        are in those same units and ``cell_height`` defaults to ``cell_width``.
        ``kind`` is ``rectangle`` (a fishnet) or ``hexagon`` (``cell_width`` is the
        across-corners width).
        """
        written = vector.grid(
            self._rt.workspace, out, path, bounds, layer, cell_width, cell_height, kind
        )
        return self._rt.record_artifact(written)

    @tool()
    def export_vector(self, path: str, out: str, layer: str | None = None) -> str:
        """Write a vector layer in the format the output extension names.

        ``.geojson``/``.json``, ``.gpkg``, and ``.shp`` are supported, so this
        both converts to GeoJSON and packages a result as a GeoPackage or
        Shapefile. ``layer`` extracts one layer of a container (see
        ``list_layers``).
        """
        written = vector.export_vector(self._rt.workspace, path, out, layer)
        return self._rt.record_artifact(written)

    @tool(effects=frozenset({Effect.MAP_WRITE}))
    def add_heatmap(
        self,
        path: str,
        name: str = "Heatmap",
        radius: float = 30.0,
        intensity: float = 1.0,
        layer: str | None = None,
    ) -> str:
        """Add a point dataset to the live map as a heatmap layer.

        Use this for density questions ("where are the incidents concentrated")
        rather than mapping thousands of individual markers. ``radius`` is the
        kernel size in pixels and ``intensity`` its weight; ``layer`` picks one
        layer of a container such as a GeoPackage.
        """
        m = self._rt.require_map()
        layer_id = vector.add_heatmap_to_map(
            self._rt.workspace, m, path, name, radius, intensity, layer
        )
        document.persist_map(m, self._rt.workspace)
        self._rt.events.notify_map()
        return layer_id

    @tool(effects=frozenset({Effect.MAP_WRITE}))
    def add_vector_to_map(
        self,
        path: str,
        name: str,
        column: str | None = None,
        palette: str = "viridis",
        layer: str | None = None,
    ) -> str:
        """Load a vector dataset and add it to the live map; returns the layer id."""
        m = self._rt.require_map()
        layer_id = vector.add_vector_to_map(
            self._rt.workspace, m, path, name, column=column, palette=palette, layer=layer
        )
        document.persist_map(m, self._rt.workspace)
        self._rt.events.notify_map()
        return layer_id


__all__ = ["VectorPack"]
