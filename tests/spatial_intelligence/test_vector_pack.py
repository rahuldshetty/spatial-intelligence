"""Vector service and pack: GeoPandas processing, map layers, and confinement."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
import pyogrio
import shapely
from shapely.geometry import mapping

from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.contracts.errors import RuntimeNotBoundError, ToolInputError, WorkspaceError
from spatial_intelligence.geo import vector
from spatial_intelligence.map import document
from spatial_intelligence.tools import ToolRegistry, ToolRuntime
from spatial_intelligence.tools.packs.vector import VectorPack
from spatial_intelligence.tools.runtime import RuntimeEvents
from spatial_intelligence.workspace import Workspace

SIDE = 0.5


def square(x: float) -> dict:
    """Return a closed square of side ``SIDE`` with its lower-left at ``(x, 0)``."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [x, 0.0],
                [x + SIDE, 0.0],
                [x + SIDE, SIDE],
                [x, SIDE],
                [x, 0.0],
            ]
        ],
    }


def feature(name: str, value: float, geometry: dict) -> dict:
    return {
        "type": "Feature",
        "properties": {"name": name, "value": value},
        "geometry": geometry,
    }


def as_geojson(geometry, **properties) -> dict:
    """Wrap a shapely geometry as a one-feature FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": properties, "geometry": mapping(geometry)}
        ],
    }


#: Four squares in a row, so a clip against the first one keeps fewer features.
POINTS = {
    "type": "FeatureCollection",
    "features": [
        feature("a", 1.0, square(0.0)),
        feature("b", 2.0, square(1.0)),
        feature("c", 3.0, square(2.0)),
        feature("d", 4.0, square(3.0)),
    ],
}


def zoned(name: str, value: float, zone: str, x: float) -> dict:
    """A square carrying a group column, for dissolve/aggregate/join tests."""
    return {
        "type": "Feature",
        "properties": {"name": name, "value": value, "zone": zone},
        "geometry": square(x),
    }


#: The same squares split into two groups, for dissolve/aggregate/joins.
ZONES = {
    "type": "FeatureCollection",
    "features": [
        zoned("a", 1.0, "N", 0.0),
        zoned("b", 2.0, "N", 1.0),
        zoned("c", 3.0, "S", 2.0),
        zoned("d", 4.0, "S", 3.0),
    ],
}
#: A 2×2 point grid: spread out enough for Voronoi/Delaunay, unlike centroids.
POINT_LAYER = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"name": name},
         "geometry": {"type": "Point", "coordinates": [x, y]}}
        for name, x, y in (("sw", 0.5, 0.5), ("nw", 0.5, 1.5), ("se", 1.5, 0.5), ("ne", 1.5, 1.5))
    ],
}
#: Three collinear points, which cannot form a diagram.
LINE_POINTS = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Point", "coordinates": [index, 0.0]}}
        for index in range(3)
    ],
}
LINE_LAYER = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"name": "road"},
         "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 0.0]]}}
    ],
}
MULTI_LAYER = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"name": "pair"},
         "geometry": {"type": "MultiPolygon", "coordinates": [
             [[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.0, 0.0]]],
             [[[1.0, 0.0], [1.5, 0.0], [1.5, 0.5], [1.0, 0.5], [1.0, 0.0]]],
         ]}}
    ],
}
#: A self-intersecting ring: the canonical invalid geometry.
BOWTIE = as_geojson(
    shapely.from_wkt("POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))"), name="bowtie", value=0.0
)
#: A dense ring, so simplification has vertices to remove.
DENSE = as_geojson(shapely.Point(0.0, 0.0).buffer(1.0, quad_segs=16), name="disc", value=0.0)
#: A key/value table (as a vector layer) with only the northern group present.
LOOKUP = as_geojson(shapely.Point(0.0, 0.0), zone="N", label="north")
#: A small square overlapping the corner of the first fixture square.
OVERLAP = as_geojson(
    shapely.from_wkt("POLYGON ((0.4 0.4, 0.5 0.4, 0.5 0.5, 0.4 0.5, 0.4 0.4))"),
    name="overlap",
    value=0.0,
)
#: Covers only square "a" (x from -0.1 to 0.4).
MASK = {
    "type": "FeatureCollection",
    "features": [feature("mask", 0.0, square(-0.1))],
}
SQUARE_ONE = {
    "type": "FeatureCollection",
    "features": [feature("a", 1.0, square(0.0))],
}


class VectorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.write("data/points.geojson", POINTS)
        self.write("data/square.geojson", SQUARE_ONE)
        self.write("data/mask.geojson", MASK)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, relative: str, payload: dict) -> Path:
        target = self.workspace.resolve(relative, write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def outputs(self) -> list[str]:
        manifest = json.loads(
            (self.workspace.root / "workspace.json").read_text(encoding="utf-8")
        )
        return manifest["outputs"]


class VectorServiceTests(VectorTestCase):
    def test_read_vector_describes_the_dataset(self):
        info = vector.read_vector(self.workspace, "data/points.geojson")

        self.assertEqual(info["crs"], "EPSG:4326")
        self.assertEqual(info["columns"], ["name", "value", "geometry"])
        self.assertEqual(info["len"], 4)
        self.assertEqual(info["bounds"], [0.0, 0.0, 3.5, 0.5])
        self.assertEqual(info["geom_types"], ["Polygon"])

    def test_reproject_vector_changes_the_crs_of_the_written_output(self):
        written = vector.reproject_vector(
            self.workspace, "data/points.geojson", "results/web.geojson", "EPSG:3857"
        )

        self.assertEqual(self.workspace.relative(written), "results/web.geojson")
        self.assertEqual(
            vector.read_vector(self.workspace, "results/web.geojson")["crs"], "EPSG:3857"
        )
        self.assertEqual(
            vector.read_vector(self.workspace, "data/points.geojson")["crs"], "EPSG:4326"
        )

    def test_buffer_grows_the_geometry_and_keeps_the_source_crs(self):
        # 500 m in UTM must stay a sub-degree change in the stored EPSG:4326
        # layer, which is what tells metric buffering apart from degree buffering.
        written = vector.buffer(
            self.workspace, "data/square.geojson", "results/buffered.geojson", 500.0
        )
        before = vector.read_vector(self.workspace, "data/square.geojson")
        after = vector.read_vector(self.workspace, "results/buffered.geojson")

        self.assertEqual(self.workspace.relative(written), "results/buffered.geojson")
        self.assertEqual(after["crs"], before["crs"])
        self.assertEqual(after["geom_types"], ["Polygon"])
        self.assertLess(after["bounds"][0], before["bounds"][0])
        self.assertLess(after["bounds"][1], before["bounds"][1])
        self.assertGreater(after["bounds"][2], before["bounds"][2])
        self.assertGreater(after["bounds"][3], before["bounds"][3])
        self.assertLess(after["bounds"][2] - after["bounds"][0], 1.0)

    def test_buffer_in_degrees_uses_the_layer_units(self):
        # Buffering a geographic layer in its own units is what GeoPandas warns
        # about; that is exactly the documented non-metric behaviour.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            written = vector.buffer(
                self.workspace,
                "data/square.geojson",
                "results/wide.geojson",
                1.0,
                unit="degrees",
            )
        after = vector.read_vector(self.workspace, self.workspace.relative(written))

        self.assertLess(after["bounds"][0], -0.9)
        self.assertGreater(after["bounds"][2], 1.4)

    def test_clip_vector_keeps_only_the_intersecting_features(self):
        written = vector.clip_vector(
            self.workspace, "data/points.geojson", "results/clipped.geojson", "data/mask.geojson"
        )
        clipped = vector.read_vector(self.workspace, "results/clipped.geojson")

        self.assertEqual(self.workspace.relative(written), "results/clipped.geojson")
        self.assertEqual(clipped["len"], 1)
        self.assertLess(clipped["len"], vector.read_vector(self.workspace, "data/points.geojson")["len"])

    def test_export_vector_writes_a_valid_feature_collection(self):
        written = vector.export_vector(
            self.workspace, "data/points.geojson", "results/points.geojson"
        )
        payload = json.loads(written.read_text(encoding="utf-8"))

        self.assertEqual(payload["type"], "FeatureCollection")
        self.assertEqual(len(payload["features"]), 4)
        self.assertEqual(payload["features"][0]["geometry"]["type"], "Polygon")
        self.assertEqual(payload["features"][0]["properties"]["name"], "a")

    def test_reading_a_missing_file_is_refused(self):
        with self.assertRaises(WorkspaceError):
            vector.read_vector(self.workspace, "data/missing.geojson")

    def test_writers_are_confined_to_output_directories(self):
        with self.assertRaises(WorkspaceError):
            vector.export_vector(self.workspace, "data/points.geojson", "traces/points.geojson")
        with self.assertRaises(WorkspaceError):
            vector.reproject_vector(
                self.workspace, "data/points.geojson", "../escaped.geojson", "EPSG:3857"
            )
        with self.assertRaises(WorkspaceError):
            vector.buffer(
                self.workspace,
                "data/points.geojson",
                str(Path(self._tmp.name) / "absolute.geojson"),
                10.0,
            )


class VectorMapTests(VectorTestCase):
    def test_add_vector_to_map_returns_the_layer_id(self):
        m = document.create_map(self.workspace)

        layer_id = vector.add_vector_to_map(
            self.workspace, m, "data/points.geojson", "Points", "value"
        )

        self.assertTrue(layer_id)
        self.assertEqual([layer["id"] for layer in m.project["layers"]], [layer_id])
        self.assertEqual([layer["name"] for layer in m.project["layers"]], ["Points"])


    def test_export_vector_writes_a_geopackage_the_reader_can_list(self):
        written = vector.export_vector(
            self.workspace, "data/points.geojson", "results/points.gpkg"
        )

        self.assertEqual(self.workspace.relative(written), "results/points.gpkg")
        listing = vector.list_layers(self.workspace, "results/points.gpkg")
        self.assertEqual([entry["layer"] for entry in listing["layers"]], ["points"])
        self.assertEqual(listing["layers"][0]["features"], 4)

    def test_export_vector_refuses_an_unsupported_extension(self):
        with self.assertRaises(ToolInputError) as caught:
            vector.export_vector(
                self.workspace, "data/points.geojson", "results/points.xlsx"
            )

        self.assertIn(".gpkg", str(caught.exception))


    def test_grid_takes_a_layer_path_and_refuses_a_layer_name(self):
        written = vector.grid(
            self.workspace, "results/cells.geojson", path="data/points.geojson", cell_width=1.0
        )
        self.assertEqual(written, self.workspace.resolve("results/cells.geojson"))

        # ``layer`` names a layer *inside* a container; a path is not one, and the
        # old signature made the error message send the caller into a failure.
        with self.assertRaises(ToolInputError) as caught:
            vector.grid(self.workspace, "results/x.geojson", layer="data/points.geojson")

        self.assertIn("path", str(caught.exception))

    def test_buffer_rejects_a_unit_it_does_not_accept(self):
        with self.assertRaises(ToolInputError) as caught:
            vector.buffer(
                self.workspace, "data/points.geojson", "results/b.geojson", 5.0,
                unit="kilometers",
            )

        # Not a silent degrees buffer, which is what an unchecked else branch did.
        self.assertIn("meters", str(caught.exception))
        self.assertFalse((self.workspace.results / "b.geojson").exists())

    def test_attribute_join_attaches_a_plain_table(self):
        table = self.workspace.resolve("data/scores.csv", write=True)
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text("name,score\na,10\nb,20\n", encoding="utf-8")

        written = vector.attribute_join(
            self.workspace, "data/points.geojson", "results/joined.geojson",
            "data/scores.csv", field="name", columns=["score"],
        )
        frame = gpd.read_file(written)

        # The table's own columns arrive as pyogrio read them (text, for a CSV) with
        # nulls where the key did not match; casting for arithmetic is the caller's
        # job, in run_python.
        scores = [None if value != value else str(value) for value in frame["score"]]
        self.assertEqual(scores, ["10", "20", None, None])

        # A table is not a layer, and saying so is clearer than an AttributeError.
        with self.assertRaises(ToolInputError) as caught:
            vector.read_vector(self.workspace, "data/scores.csv")

        self.assertIn("no geometry", str(caught.exception))

    def test_points_along_refuses_an_interval_that_would_exceed_the_cap(self):
        self.write("data/line.geojson", LINE_LAYER)

        with self.assertRaises(ToolInputError) as caught:
            vector.points_along(self.workspace, "data/line.geojson", "results/p.geojson", 1e-9)

        message = str(caught.exception)
        self.assertIn(str(vector.MAX_ALONG_POINTS), message)
        self.assertIn("at least", message)
        # Refused before writing, rather than returning a truncated layer.
        self.assertFalse((self.workspace.results / "p.geojson").exists())

    def test_select_by_location_skips_a_null_geometry(self):
        self.write("data/nulls.geojson", {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"name": "none"}, "geometry": None},
                feature("hit", 1.0, square(0.0)),
                feature("miss", 2.0, square(3.0)),
            ],
        })

        written = vector.select_by_location(
            self.workspace, "data/nulls.geojson", "results/sel.geojson", "data/mask.geojson"
        )

        self.assertEqual(list(gpd.read_file(written)["name"]), ["hit"])

    def test_the_summarizing_tools_refuse_an_empty_layer(self):
        self.write("data/empty.geojson", {"type": "FeatureCollection", "features": []})

        for call in (
            lambda: vector.convex_hull(self.workspace, "data/empty.geojson", "results/h.geojson"),
            lambda: vector.dissolve(self.workspace, "data/empty.geojson", "results/d.geojson"),
        ):
            with self.subTest(call=call):
                with self.assertRaises(ToolInputError) as caught:
                    call()
                # Otherwise a summary of nothing is written as a feature whose
                # geometry is an empty collection.
                self.assertIn("no features", str(caught.exception))

    def test_a_numeric_argument_reports_the_same_shape_of_error_everywhere(self):
        self.write("data/line.geojson", LINE_LAYER)
        cases = {
            "interval": lambda: vector.points_along(
                self.workspace, "data/line.geojson", "results/x.geojson", 0
            ),
            "tolerance": lambda: vector.simplify(
                self.workspace, "data/line.geojson", "results/x.geojson", -1.0
            ),
        }

        for name, call in cases.items():
            with self.subTest(argument=name):
                with self.assertRaises(ToolInputError) as caught:
                    call()
                self.assertTrue(
                    str(caught.exception).startswith(f"{name} must be"),
                    str(caught.exception),
                )


class VectorHeatmapTests(VectorTestCase):
    def build_pack(self, map_obj=None) -> ToolRegistry:
        runtime = ToolRuntime(workspace=self.workspace, map=map_obj)
        registry = ToolRegistry()
        registry.add_pack(VectorPack, runtime)
        return registry

    def test_add_heatmap_adds_a_layer_and_persists_it(self):
        self.write("data/stations.geojson", POINT_LAYER)
        m = document.create_map(self.workspace)
        registry = self.build_pack(m)

        layer_id = registry.get("add_heatmap").callable(
            "data/stations.geojson", "Stations", 40.0, 2.0
        )

        layer = next(layer for layer in m.project["layers"] if layer["id"] == layer_id)
        self.assertEqual(layer["name"], "Stations")
        # GeoLibre renders a heatmap through the point renderer, and the app
        # resolves the project-level style copy, so both must carry it.
        self.assertEqual(layer["style"]["pointRenderer"], "heatmap")
        self.assertEqual(layer["style"]["heatmapRadius"], 40.0)
        self.assertEqual(layer["style"]["heatmapIntensity"], 2.0)
        self.assertEqual(m.project["styles"][layer_id]["pointRenderer"], "heatmap")
        snapshot = json.loads(document.snapshot_path(self.workspace).read_text(encoding="utf-8"))
        self.assertEqual([item["id"] for item in snapshot["layers"]], [layer_id])

    def test_add_heatmap_refuses_polygon_input_and_a_missing_map(self):
        registry = self.build_pack()
        with self.assertRaises(RuntimeNotBoundError):
            registry.get("add_heatmap").callable("data/points.geojson", "Squares")

        m = document.create_map(self.workspace)
        with self.assertRaises(ToolInputError) as caught:
            self.build_pack(m).get("add_heatmap").callable("data/points.geojson", "Squares")
        self.assertIn("point geometry", str(caught.exception))

class VectorPackTests(VectorTestCase):
    def build_pack(self, notifications: list[str], map_obj=None) -> ToolRegistry:
        runtime = ToolRuntime(
            workspace=self.workspace,
            map=map_obj,
            events=RuntimeEvents(
                files_changed=lambda: notifications.append("files"),
                map_changed=lambda: notifications.append("map"),
            ),
        )
        registry = ToolRegistry()
        registry.add_pack(VectorPack, runtime)
        return registry

    def test_writers_record_the_output_and_notify_files(self):
        notifications: list[str] = []
        registry = self.build_pack(notifications)

        absolute = registry.get("reproject_vector").callable(
            "data/points.geojson", "results/web.geojson", "EPSG:3857"
        )

        self.assertEqual(Path(absolute), self.workspace.resolve("results/web.geojson"))
        self.assertEqual(notifications, ["files"])
        self.assertEqual(self.outputs(), ["results/web.geojson"])

    def test_read_vector_reports_a_missing_file(self):
        with self.assertRaises(WorkspaceError):
            self.build_pack([]).get("read_vector").callable("data/missing.geojson")

    def test_writing_outside_the_output_directories_is_refused(self):
        registry = self.build_pack([])

        with self.assertRaises(WorkspaceError):
            registry.get("export_vector").callable("data/points.geojson", "traces/points.geojson")

        self.assertEqual(self.outputs(), [])

    def test_add_vector_to_map_persists_the_snapshot_and_notifies_map(self):
        notifications: list[str] = []
        m = document.create_map(self.workspace)
        registry = self.build_pack(notifications, map_obj=m)

        layer_id = registry.get("add_vector_to_map").callable(
            "data/points.geojson", "Points", "value"
        )

        snapshot = document.snapshot_path(self.workspace)
        self.assertTrue(snapshot.is_file())
        project = json.loads(snapshot.read_text(encoding="utf-8"))
        self.assertEqual([layer["id"] for layer in project["layers"]], [layer_id])
        self.assertEqual(
            [layer["name"] for layer in project["layers"]], ["Points"]
        )
        self.assertEqual(notifications, ["map"])

    def test_add_vector_to_map_requires_a_live_map(self):
        with self.assertRaises(RuntimeNotBoundError):
            self.build_pack([]).get("add_vector_to_map").callable(
                "data/points.geojson", "Points"
            )

    def test_registered_vector_tools_expose_expected_effects(self):
        registry = self.build_pack([])

        self.assertEqual(
            sorted(registry.names()),
            [
                "add_heatmap",
                "add_vector_to_map",
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
                "export_vector",
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
                "voronoi",
            ],
        )
        self.assertTrue(registry.replay_safe("read_vector"))
        self.assertTrue(registry.replay_safe("check_geometry"))
        self.assertTrue(registry.replay_safe("list_layers"))
        self.assertFalse(registry.replay_safe("buffer"))
        self.assertFalse(registry.replay_safe("dissolve"))
        self.assertFalse(registry.replay_safe("export_vector"))
        self.assertFalse(registry.replay_safe("add_vector_to_map"))
        self.assertEqual(registry.get("read_vector").effects, frozenset({Effect.READ}))
        self.assertEqual(registry.get("check_geometry").effects, frozenset({Effect.READ}))
        self.assertEqual(
            registry.get("add_vector_to_map").effects, frozenset({Effect.MAP_WRITE})
        )
        self.assertEqual(registry.get("buffer").category, "vector")


class VectorAnalysisTests(VectorTestCase):
    """The analysis toolbox: geometry, attributes, selection, and grids."""

    def setUp(self):
        super().setUp()
        self.write("data/zones.geojson", ZONES)
        self.write("data/point-layer.geojson", POINT_LAYER)
        self.write("data/line-points.geojson", LINE_POINTS)
        self.write("data/line.geojson", LINE_LAYER)
        self.write("data/multi.geojson", MULTI_LAYER)
        self.write("data/bowtie.geojson", BOWTIE)
        self.write("data/dense.geojson", DENSE)
        self.write("data/lookup.geojson", LOOKUP)
        self.write("data/overlap.geojson", OVERLAP)
        self.write("data/square-web.geojson", self.web_mercator_square())
        self.write_gpkg("data/container.gpkg")

    def web_mercator_square(self) -> dict:
        """The first fixture square, reprojected to EPSG:3857."""
        frame = gpd.GeoDataFrame(
            {"name": ["a"], "value": [1.0]},
            geometry=[shapely.from_wkt("POLYGON ((0 0, 0.5 0, 0.5 0.5, 0 0.5, 0 0))")],
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")
        return json.loads(frame.to_json())

    def write_gpkg(self, relative: str) -> Path:
        """A GeoPackage holding two named layers, as a real container would."""
        target = self.workspace.resolve(relative, write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        frames = {
            "parcels": gpd.GeoDataFrame(
                {"name": ["a", "b"]},
                geometry=[shapely.from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
                          shapely.from_wkt("POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))")],
                crs="EPSG:4326",
            ),
            "wells": gpd.GeoDataFrame(
                {"well": ["w1"]},
                geometry=[shapely.from_wkt("POINT (0.5 0.5)")],
                crs="EPSG:4326",
            ),
        }
        for layer, frame in frames.items():
            pyogrio.write_dataframe(frame, target, layer=layer, driver="GPKG")
        return target

    # -- geometry ---------------------------------------------------------

    def test_dissolve_by_column_merges_one_geometry_per_group(self):
        written = vector.dissolve(self.workspace, "data/zones.geojson", "results/zones.geojson", "zone")
        result = gpd.read_file(written)

        self.assertEqual(sorted(result["zone"]), ["N", "S"])
        self.assertEqual(len(result), 2)
        north = result[result["zone"] == "N"].geometry.iloc[0]
        self.assertEqual(north.bounds, (0.0, 0.0, 1.5, 0.5))

    def test_dissolve_without_a_column_returns_one_feature(self):
        written = vector.dissolve(self.workspace, "data/points.geojson", "results/one.geojson")
        result = gpd.read_file(written)

        self.assertEqual(len(result), 1)
        self.assertEqual(tuple(round(value, 3) for value in result.geometry.iloc[0].bounds),
                         (0.0, 0.0, 3.5, 0.5))

    def test_dissolve_reports_an_unknown_column_with_the_available_ones(self):
        with self.assertRaises(ToolInputError) as caught:
            vector.dissolve(self.workspace, "data/zones.geojson", "results/x.geojson", "nope")

        self.assertIn("available columns", str(caught.exception))

    def test_overlay_intersection_difference_and_union(self):
        intersection = gpd.read_file(
            vector.overlay(self.workspace, "data/square.geojson", "results/i.geojson", "data/overlap.geojson")
        )
        difference = gpd.read_file(
            vector.overlay(
                self.workspace, "data/square.geojson", "results/d.geojson", "data/overlap.geojson",
                operation="difference",
            )
        )
        union = gpd.read_file(
            vector.overlay(
                self.workspace, "data/square.geojson", "results/u.geojson", "data/overlap.geojson",
                operation="union",
            )
        )

        self.assertAlmostEqual(intersection.geometry.iloc[0].area, 0.1 * 0.1, places=4)
        self.assertAlmostEqual(difference.geometry.iloc[0].area, 0.25 - 0.01, places=4)
        self.assertEqual(len(union), 1)
        # A union carries no attributes, matching the upstream tool.
        self.assertNotIn("name", union.columns)

    def test_overlay_aligns_a_layer_that_is_in_another_crs(self):
        written = vector.overlay(
            self.workspace,
            "data/square.geojson",
            "results/aligned.geojson",
            "data/square-web.geojson",
        )
        result = gpd.read_file(written)

        self.assertEqual(str(result.crs), "EPSG:4326")
        self.assertEqual(len(result), 1)

    def test_centroids_land_in_the_middle_of_each_square(self):
        written = vector.centroids(self.workspace, "data/points.geojson", "results/mid.geojson")
        result = gpd.read_file(written)

        self.assertEqual(sorted(result.geom_type.unique()), ["Point"])
        self.assertAlmostEqual(result.geometry.iloc[0].x, 0.25, places=4)
        self.assertAlmostEqual(result.geometry.iloc[0].y, 0.25, places=4)
        self.assertEqual(list(result["name"]), ["a", "b", "c", "d"])

    def test_convex_hull_and_bounding_box_enclosing_everything(self):
        hull = gpd.read_file(
            vector.convex_hull(self.workspace, "data/points.geojson", "results/hull.geojson")
        )
        bounds = gpd.read_file(
            vector.bounding_box(self.workspace, "data/points.geojson", "results/bbox.geojson")
        )

        self.assertEqual(len(hull), 1)
        self.assertEqual(len(bounds), 1)
        self.assertEqual(tuple(bounds.geometry.iloc[0].bounds), (0.0, 0.0, 3.5, 0.5))
        self.assertTrue(hull.geometry.iloc[0].covers(bounds.geometry.iloc[0]))

    def test_simplify_reduces_the_vertex_count(self):
        before = len(gpd.read_file(self.workspace.resolve("data/dense.geojson", must_exist=True)).geometry.iloc[0].exterior.coords)
        written = vector.simplify(self.workspace, "data/dense.geojson", "results/simple.geojson", 0.01)
        after = len(gpd.read_file(written).geometry.iloc[0].exterior.coords)

        self.assertLess(after, before)

    def test_explode_splits_a_multipart_geometry_into_one_feature_per_part(self):
        written = vector.explode(self.workspace, "data/multi.geojson", "results/parts.geojson")
        result = gpd.read_file(written)

        self.assertEqual(len(result), 2)
        self.assertEqual(sorted(result.geom_type.unique()), ["Polygon"])
        self.assertEqual(list(result["name"]), ["pair", "pair"])

    def test_points_along_spaces_points_at_the_interval(self):
        written = vector.points_along(self.workspace, "data/line.geojson", "results/along.geojson", 0.25)
        result = gpd.read_file(written)

        # A 1.0-long line sampled every 0.25 from its start: 0, .25, .5, .75, 1.
        self.assertEqual(len(result), 5)
        self.assertEqual(sorted(result.geom_type.unique()), ["Point"])
        self.assertAlmostEqual(result.geometry.iloc[2].x, 0.5, places=6)

    def test_voronoi_and_delaunay_from_a_point_layer(self):
        cells = gpd.read_file(
            vector.voronoi(self.workspace, "data/point-layer.geojson", "results/voronoi.geojson")
        )
        triangles = gpd.read_file(
            vector.voronoi(
                self.workspace, "data/point-layer.geojson", "results/delaunay.geojson",
                kind="delaunay",
            )
        )

        self.assertGreaterEqual(len(cells), 3)
        self.assertTrue(all(cell.is_valid for cell in cells.geometry))
        self.assertTrue(
            all(len(triangle.exterior.coords) - 1 == 3 for triangle in triangles.geometry)
        )

    def test_grid_rectangle_and_hexagon(self):
        squares = gpd.read_file(
            vector.grid(self.workspace, "results/grid.geojson", bounds=[0.0, 0.0, 1.0, 1.0], cell_width=0.5)
        )
        hexes = gpd.read_file(
            vector.grid(
                self.workspace,
                "results/hex.geojson",
                bounds=[0.0, 0.0, 1.0, 1.0],
                cell_width=0.5,
                kind="hexagon",
            )
        )
        from_layer = gpd.read_file(
            vector.grid(
                self.workspace, "results/from-layer.geojson",
                path="data/point-layer.geojson", cell_width=1.0,
            )
        )

        self.assertEqual(len(squares), 4)
        self.assertAlmostEqual(squares.geometry.iloc[0].area, 0.25, places=6)
        self.assertGreaterEqual(len(hexes), 4)
        self.assertTrue(all(len(cell.exterior.coords) - 1 == 6 for cell in hexes.geometry))
        # The point layer spans 0.5…1.5 in both axes, so one 1.0 cell covers it.
        self.assertEqual(
            tuple(round(value, 3) for value in from_layer.total_bounds), (0.5, 0.5, 1.5, 1.5)
        )

    def test_grid_and_interval_reject_impossible_arguments(self):
        cases = [
            lambda: vector.grid(self.workspace, "results/x.geojson"),
            lambda: vector.grid(self.workspace, "results/x.geojson", bounds=[0.0, 0.0, 1.0], cell_width=0.5),
            lambda: vector.grid(self.workspace, "results/x.geojson", bounds=[0.0, 0.0, 1.0, 1.0], cell_width=0),
            lambda: vector.grid(
                self.workspace, "results/x.geojson", bounds=[0.0, 0.0, 1.0, 1.0], cell_width=0.00001
            ),
            lambda: vector.grid(
                self.workspace, "results/x.geojson", bounds=[0.0, 0.0, 1.0, 1.0], cell_width=0.5, kind="triangle"
            ),
            lambda: vector.points_along(self.workspace, "data/line.geojson", "results/x.geojson", 0),
        ]

        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ToolInputError):
                    case()

    def test_voronoi_refuses_collinear_points(self):
        with self.assertRaises(ToolInputError):
            vector.voronoi(self.workspace, "data/line-points.geojson", "results/x.geojson")

    # -- attributes and selection -----------------------------------------

    def test_spatial_join_inner_and_left(self):
        inner = gpd.read_file(
            vector.spatial_join(
                self.workspace, "data/points.geojson", "results/join.geojson", "data/square.geojson",
                predicate="within",
            )
        )
        left = gpd.read_file(
            vector.spatial_join(
                self.workspace, "data/points.geojson", "results/left.geojson", "data/square.geojson",
                predicate="within", how="left",
            )
        )

        self.assertEqual(len(inner), 1)
        # Only the containing square carries a name; the rest are empty.
        self.assertEqual(inner["name_right"].tolist(), ["a"])
        self.assertEqual(len(left), 4)
        self.assertEqual(left["name_right"].isna().sum(), 3)

    def test_attribute_join_carries_columns_and_keeps_unmatched_rows(self):
        written = vector.attribute_join(
            self.workspace, "data/zones.geojson", "results/attr.geojson", "data/lookup.geojson",
            field="zone", columns=["label"],
        )
        result = gpd.read_file(written)

        self.assertEqual(sorted(result.columns), ["geometry", "label", "name", "value", "zone"])
        self.assertEqual(result["label"].notna().sum(), 2)
        self.assertEqual(result["label"].isna().sum(), 2)
        self.assertEqual(len(result), 4)

    def test_attribute_join_reports_a_missing_key(self):
        with self.assertRaises(ToolInputError) as caught:
            vector.attribute_join(
                self.workspace, "data/zones.geojson", "results/attr.geojson",
                "data/lookup.geojson", field="missing",
            )

        self.assertIn("missing", str(caught.exception))

    def test_select_by_value_covers_numeric_string_and_null_operators(self):
        def count(column, operator, value=None):
            written = vector.select_by_value(
                self.workspace, "data/zones.geojson", "results/sel.geojson", column, operator, value
            )
            return len(gpd.read_file(written))

        self.assertEqual(count("value", "gt", "2"), 2)
        self.assertEqual(count("value", "gte", "2"), 3)
        self.assertEqual(count("zone", "eq", "N"), 2)
        self.assertEqual(count("name", "contains", "a"), 1)
        self.assertEqual(count("name", "is-null"), 0)
        self.assertEqual(count("name", "is-not-null"), 4)

    def test_select_by_value_rejects_unknown_operators_and_missing_values(self):
        with self.assertRaises(ToolInputError):
            vector.select_by_value(
                self.workspace, "data/zones.geojson", "results/x.geojson", "value", "like", "2"
            )
        with self.assertRaises(ToolInputError):
            vector.select_by_value(
                self.workspace, "data/zones.geojson", "results/x.geojson", "value", "gt"
            )

    def test_select_by_location_predicates(self):
        def select(predicate, mask="data/square.geojson"):
            written = vector.select_by_location(
                self.workspace, "data/points.geojson", "results/loc.geojson", mask, predicate
            )
            return [int(value) for value in gpd.read_file(written)["value"]]

        self.assertEqual(select("intersects"), [1])
        self.assertEqual(select("within"), [1])
        self.assertEqual(select("disjoint"), [2, 3, 4])

    def test_aggregate_counts_and_sums_per_group(self):
        counted = gpd.read_file(
            vector.aggregate(self.workspace, "data/zones.geojson", "results/count.geojson", "zone")
        )
        summed = gpd.read_file(
            vector.aggregate(
                self.workspace, "data/zones.geojson", "results/sum.geojson", "zone",
                statistic="sum", field="value",
            )
        )

        self.assertEqual(sorted(counted.columns), ["count", "geometry", "zone"])
        self.assertEqual(dict(zip(counted["zone"], counted["count"])), {"N": 2, "S": 2})
        self.assertAlmostEqual(dict(zip(summed["zone"], summed["value_sum"]))["N"], 3.0)
        self.assertAlmostEqual(dict(zip(summed["zone"], summed["value_sum"]))["S"], 7.0)

    def test_aggregate_rejects_an_unknown_statistic_and_a_missing_field(self):
        with self.assertRaises(ToolInputError):
            vector.aggregate(
                self.workspace, "data/zones.geojson", "results/x.geojson", "zone", statistic="mode"
            )
        with self.assertRaises(ToolInputError):
            vector.aggregate(
                self.workspace, "data/zones.geojson", "results/x.geojson", "zone",
                statistic="sum", field=None,
            )

    # -- topology and containers -------------------------------------------

    def test_check_geometry_reports_the_reason_and_fix_geometry_repairs_it(self):
        report = vector.check_geometry(self.workspace, "data/bowtie.geojson")

        self.assertFalse(report["valid"])
        self.assertEqual(report["invalid"], 1)
        self.assertEqual(report["len"], 1)

        written = vector.fix_geometry(self.workspace, "data/bowtie.geojson", "results/fixed.geojson")
        repaired = gpd.read_file(written)

        self.assertEqual(len(repaired), 1)
        self.assertTrue(all(geometry.is_valid for geometry in repaired.geometry))
        self.assertTrue(vector.check_geometry(self.workspace, "results/fixed.geojson")["valid"])

    def test_check_geometry_passes_a_valid_layer(self):
        report = vector.check_geometry(self.workspace, "data/square.geojson")

        self.assertTrue(report["valid"])
        self.assertEqual(report["invalid"], 0)
        self.assertEqual(report["details"], [])

    def test_list_layers_describes_a_container_dataset(self):
        listing = vector.list_layers(self.workspace, "data/container.gpkg")

        self.assertEqual(listing["path"], "data/container.gpkg")
        by_name = {entry["layer"]: entry for entry in listing["layers"]}
        self.assertEqual(sorted(by_name), ["parcels", "wells"])
        self.assertEqual(by_name["parcels"]["features"], 2)
        self.assertEqual(by_name["wells"]["features"], 1)
        self.assertEqual(by_name["wells"]["geometry_type"], "Point")

    def test_two_layer_tools_take_a_layer_name_for_each_side(self):
        joined = gpd.read_file(
            vector.spatial_join(
                self.workspace,
                "data/container.gpkg",
                "results/joined.geojson",
                "data/container.gpkg",
                predicate="within",
                layer="wells",
                join_layer="parcels",
            )
        )
        clipped = gpd.read_file(
            vector.clip_vector(
                self.workspace,
                "data/container.gpkg",
                "results/clipped.geojson",
                "data/container.gpkg",
                layer="wells",
                mask_layer="parcels",
            )
        )

        # The one well sits inside the first parcel, so joining by name finds it
        # and clipping the wells layer to the parcels keeps exactly that point.
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined["name"].tolist(), ["a"])
        self.assertEqual(len(clipped), 1)
        self.assertEqual(clipped["well"].tolist(), ["w1"])

    def test_reading_a_container_without_a_layer_name_is_refused(self):
        with self.assertRaises(ToolInputError) as caught:
            vector.read_vector(self.workspace, "data/container.gpkg")

        message = str(caught.exception)
        self.assertIn("parcels", message)
        self.assertIn("wells", message)
        self.assertIn("layer=", message)

    def test_read_vector_and_export_vector_take_a_named_layer(self):
        info = vector.read_vector(self.workspace, "data/container.gpkg", "wells")
        written = vector.export_vector(
            self.workspace, "data/container.gpkg", "results/wells.geojson", "wells"
        )
        extracted = gpd.read_file(written)

        self.assertEqual(info["len"], 1)
        self.assertEqual(info["layer"], "wells")
        self.assertEqual(info["geom_types"], ["Point"])
        self.assertEqual(list(extracted["well"]), ["w1"])


class VectorPackAnalysisTests(VectorTestCase):
    def test_analysis_tools_record_outputs_and_read_only_tools_do_not(self):
        notifications: list[str] = []
        runtime = ToolRuntime(
            workspace=self.workspace,
            events=RuntimeEvents(files_changed=lambda: notifications.append("files")),
        )
        registry = ToolRegistry()
        registry.add_pack(VectorPack, runtime)
        call = lambda name, *args, **kwargs: registry.get(name).callable(*args, **kwargs)  # noqa: E731

        absolute = call("centroids", "data/points.geojson", "results/mid.geojson")
        grid_path = call(
            "grid", "results/cells.geojson", None, [0.0, 0.0, 1.0, 1.0], None, 0.5
        )

        self.assertEqual(Path(absolute), self.workspace.resolve("results/mid.geojson"))
        self.assertEqual(Path(grid_path), self.workspace.resolve("results/cells.geojson"))
        self.assertEqual(notifications, ["files", "files"])
        self.assertEqual(
            self.outputs(), ["results/mid.geojson", "results/cells.geojson"]
        )

        call("check_geometry", "data/points.geojson")
        call("list_layers", "data/points.geojson")
        self.assertEqual(len(self.outputs()), 2)


if __name__ == "__main__":
    unittest.main()
