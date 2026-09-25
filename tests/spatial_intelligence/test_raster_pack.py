"""Raster service and pack: inspection, stretching, warping, and approval."""


import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.contracts.errors import ToolInputError, WorkspaceError
from spatial_intelligence.contracts.progress import JobState, Reporter
from spatial_intelligence.geo import raster
from spatial_intelligence.tools import ToolRegistry, ToolRuntime
from spatial_intelligence.tools.packs.raster import RasterPack
from spatial_intelligence.tools.runtime import RuntimeEvents
from spatial_intelligence.workspace import Workspace

SRC = "data/scene.tif"


class RecordingSink:
    def __init__(self):
        self.events = []
    def emit(self, event):
        self.events.append(event.as_dict())
    def for_job(self, job_id):
        return [event for event in self.events if event["job_id"] == job_id]
    def statuses(self):
        return [event["status"] for event in self.events]


class RasterTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
    def tearDown(self):
        self._tmp.cleanup()
    def write_raster(self, values=None, *, count=1, dtype="float32", nodata=None):
        """Write a 4x3 synthetic raster into ``data/`` and return its path."""
        if values is None:
            values = np.arange(1, 3 * 4 * count + 1, dtype=dtype).reshape(count, 3, 4)
        profile = {
            "driver": "GTiff",
            "width": 4,
            "height": 3,
            "count": count,
            "dtype": dtype,
            "crs": "EPSG:4326",
            "transform": from_origin(0.0, 1.0, 0.25, 0.25),
        }
        if nodata is not None:
            profile["nodata"] = nodata
        path = self.workspace.resolve(SRC, write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(np.asarray(values, dtype=dtype))
        return path
    def write_mask(self, geometry: dict, relative: str = "data/mask.geojson") -> str:
        path = self.workspace.resolve(relative, write=True)
        path.write_text(
            json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature",
                        "properties": {}, "geometry": geometry}]}),
            encoding="utf-8",
        )
        return relative
    def manifest_outputs(self) -> list[str]:
        manifest = json.loads(self.workspace.manifest_path.read_text(encoding="utf-8"))
        return manifest["outputs"]


class RasterInfoTests(RasterTestCase):

    def test_raster_info_reports_crs_size_bands_and_bounds(self):
        self.write_raster(count=2, dtype="uint16")

        info = raster.raster_info(self.workspace, SRC)

        self.assertEqual(info["crs"], "EPSG:4326")
        self.assertEqual((info["width"], info["height"]), (4, 3))
        self.assertEqual(info["count"], 2)
        self.assertEqual(info["dtypes"], ["uint16", "uint16"])
        self.assertEqual(info["bounds"], [0.0, 0.25, 1.0, 1.0])
        self.assertEqual(len(info["transform"]), 6)
        self.assertIsNone(info["nodata"])

    def test_raster_info_reports_a_missing_file(self):
        with self.assertRaises(WorkspaceError) as caught:
            raster.raster_info(self.workspace, "data/missing.tif")

        self.assertIn("missing.tif", str(caught.exception))

    def test_raster_info_refuses_a_path_outside_the_workspace(self):
        with self.assertRaises(WorkspaceError):
            raster.raster_info(self.workspace, "../outside.tif")


class RasterStatsTests(RasterTestCase):

    def test_stats_return_a_summary_and_a_256_bin_histogram(self):
        self.write_raster(values=np.arange(12, dtype="float32").reshape(1, 3, 4))

        stats = raster.raster_stats(self.workspace, SRC)

        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 11.0)
        self.assertEqual(stats["mean"], 5.5)
        self.assertEqual(len(stats["histogram"]), 256)
        self.assertEqual(sum(stats["histogram"]), 12)
        self.assertLess(stats["p2"], stats["p98"])
        self.assertEqual(stats["p2"], float(np.nanpercentile(np.arange(12), 2)))

    def test_stats_return_an_empty_summary_for_a_fully_masked_band(self):
        self.write_raster(
            values=np.full((1, 3, 4), -9999.0, dtype="float32"), nodata=-9999.0
        )

        stats = raster.raster_stats(self.workspace, SRC)

        self.assertIsNone(stats["min"])
        self.assertIsNone(stats["p98"])
        self.assertEqual(stats["histogram"], [])

    def test_sample_point_reads_the_pixel_value(self):
        self.write_raster(values=np.arange(12, dtype="float32").reshape(1, 3, 4))

        sampled = raster.sample_point(self.workspace, SRC, 0.375, 0.875)

        self.assertEqual(sampled, {"value": 1.0, "lng": 0.375, "lat": 0.875})


class RasterWriterTests(RasterTestCase):

    def test_to_cog_writes_a_readable_copy(self):
        self.write_raster(count=2)

        written = raster.to_cog(self.workspace, SRC, "results/scene_cog.tif")

        self.assertEqual(written, self.workspace.results / "scene_cog.tif")
        with rasterio.open(written) as ds:
            self.assertEqual((ds.count, ds.width, ds.height), (2, 4, 3))
            self.assertEqual(ds.crs.to_epsg(), 4326)

    def test_reproject_writes_the_destination_crs(self):
        self.write_raster()

        written = raster.reproject(self.workspace, SRC, "results/3857.tif", "EPSG:3857")

        with rasterio.open(written) as ds:
            self.assertEqual(ds.crs.to_epsg(), 3857)
            self.assertEqual(ds.count, 1)

    def test_clip_writes_only_the_requested_window(self):
        self.write_raster()

        written = raster.clip(
            self.workspace, SRC, "results/clip.tif", bounds=[0.0, 0.25, 0.5, 0.75]
        )

        with rasterio.open(written) as ds:
            self.assertEqual((ds.width, ds.height), (2, 2))
            self.assertEqual(list(ds.bounds), [0.0, 0.25, 0.5, 0.75])
            np.testing.assert_allclose(ds.read(1), [[5.0, 6.0], [9.0, 10.0]])

    def test_clip_masks_to_a_geojson_feature_collection(self):
        self.write_raster()
        mask_path = self.write_mask(
            {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.25], [0.5, 0.25], [0.5, 0.75], [0.0, 0.75], [0.0, 0.25]]
                ],
            }
        )

        written = raster.clip(self.workspace, SRC, "results/masked.tif", mask_geojson=mask_path)

        with rasterio.open(written) as ds:
            self.assertEqual((ds.width, ds.height), (2, 2))
            np.testing.assert_allclose(ds.read(1), [[5.0, 6.0], [9.0, 10.0]])

    def test_clip_without_bounds_or_mask_fails(self):
        self.write_raster()

        with self.assertRaises(WorkspaceError) as caught:
            raster.clip(self.workspace, SRC, "results/clip.tif")

        self.assertIn("bounds or mask_geojson", str(caught.exception))

    def test_rescale_stretches_the_band_to_uint8(self):
        self.write_raster(values=np.arange(12, dtype="float32").reshape(1, 3, 4))

        written = raster.rescale(self.workspace, SRC, "results/stretch.tif")

        with rasterio.open(written) as ds:
            self.assertEqual(ds.dtypes, ("uint8",))
            self.assertEqual(ds.count, 1)
            data = ds.read(1)
        self.assertEqual((int(data.min()), int(data.max())), (0, 255))

    def test_rescale_honors_an_explicit_range(self):
        self.write_raster(values=np.arange(12, dtype="float32").reshape(1, 3, 4))

        written = raster.rescale(self.workspace, SRC, "results/+stretch.tif", vmin=0.0, vmax=11.0)

        with rasterio.open(written) as ds:
            data = ds.read(1)
        self.assertEqual((int(data[0, 0]), int(data[2, 3])), (0, 255))

    def test_rescale_maps_source_nodata_to_zero_when_asked(self):
        values = np.arange(12, dtype="float32").reshape(1, 3, 4).copy()
        values[0, 2, 3] = -9999.0
        self.write_raster(values=values, nodata=-9999.0)

        written = raster.rescale(self.workspace, SRC, "results/masked.tif", nodata=0.0)

        with rasterio.open(written) as ds:
            self.assertEqual(ds.nodata, 0.0)
            self.assertEqual(int(ds.read(1)[2, 3]), 0)

    def test_band_math_evaluates_the_expression(self):
        values = np.arange(24, dtype="float32").reshape(2, 3, 4)
        self.write_raster(values=values, count=2)

        written = raster.band_math(
            self.workspace, SRC, "results/index.tif", "(b2 - b1) / 2", {"b1": 1, "b2": 2}
        )

        with rasterio.open(written) as ds:
            self.assertEqual(ds.count, 1)
            np.testing.assert_allclose(ds.read(1), np.full((3, 4), 6.0))

    def test_gdal_translate_applies_creation_options(self):
        # The escape hatch drives the GDAL library rasterio bundles, so it works
        # with nothing extra installed.
        self.write_raster()

        written = raster.gdal_translate(
            self.workspace, SRC, "results/translated.tif",
            {"TILED": "YES", "COMPRESS": "DEFLATE", "PREDICTOR": "2"},
        )

        with rasterio.open(written) as ds, rasterio.open(self.workspace.resolve(SRC, must_exist=True)) as src:
            self.assertTrue(ds.is_tiled)
            self.assertEqual(ds.compression.name.lower(), "deflate")
            self.assertEqual(ds.read(1).tolist(), src.read(1).tolist())

    def test_gdal_translate_can_write_a_cog(self):
        self.write_raster()

        written = raster.gdal_translate(
            self.workspace, SRC, "results/as-cog.tif", {"driver": "COG"}
        )

        self.assertTrue(raster.looks_cog(written))
        self.assertEqual(raster.raster_info(self.workspace, "results/as-cog.tif")["cog"], True)

    def test_gdal_translate_reports_a_driver_that_cannot_be_written(self):
        self.write_raster()

        with self.assertRaises(ToolInputError):
            raster.gdal_translate(
                self.workspace, SRC, "results/broken.tif", {"driver": "NotADriver"}
            )


class RasterAnalysisTests(RasterTestCase):
    """Derived products: indices, zonal summaries, terrain, vectorization."""
    def write_grid(
        self,
        relative: str,
        array,
        *,
        crs: str = "EPSG:3857",
        origin: tuple[float, float] = (0.0, 0.0),
        pixel: float = 10.0,
        nodata=None,
    ) -> str:
        """Write an array as a GeoTIFF whose pixel size is ``pixel`` in ``crs``."""
        raw = np.asarray(array)
        data = raw[None, ...] if raw.ndim == 2 else raw
        count, height, width = data.shape
        path = self.workspace.resolve(relative, write=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": count,
            "dtype": np.asarray(array).dtype.name,
            "crs": crs,
            "transform": from_origin(origin[0], origin[1], pixel, pixel),
        }
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(path, "w", **profile) as ds:
            ds.write(data)
        return relative
    def read_band(self, relative: str, band: int = 1):
        path = self.workspace.resolve(relative, must_exist=True)
        with rasterio.open(path) as ds:
            return ds.read(band, masked=True), ds

    def test_spectral_index_computes_ndvi_from_one_multi_band_raster(self):
        red = np.full((4, 4), 2000, dtype="uint16")
        nir = np.full((4, 4), 6000, dtype="uint16")
        self.write_grid("data/stack.tif", np.stack([red, nir]))

        written = raster.spectral_index(
            self.workspace, "data/stack.tif", "results/ndvi.tif", "ndvi",
            {"red": 1, "nir": 2}, scale=10000,
        )
        values, ds = self.read_band("results/ndvi.tif")

        self.assertEqual(self.workspace.relative(written), "results/ndvi.tif")
        self.assertEqual(ds.dtypes[0], "float32")
        self.assertAlmostEqual(float(values.mean()), 0.5, places=5)

    def test_spectral_index_reads_bands_from_separate_files(self):
        self.write_grid("data/B04.tif", np.full((4, 4), 0.2, dtype="float32"))
        self.write_grid("data/B08.tif", np.full((4, 4), 0.6, dtype="float32"))
        self.write_grid("data/elsewhere.tif", np.full((3, 3), 0.5, dtype="float32"))

        raster.spectral_index(
            self.workspace, "data/B04.tif", "results/ndvi.tif", "ndvi",
            {"red": "data/B04.tif", "nir": "data/B08.tif"},
        )
        values, _ = self.read_band("results/ndvi.tif")

        self.assertAlmostEqual(float(values.mean()), 0.5, places=5)
        with self.assertRaises(ToolInputError) as caught:
            raster.spectral_index(
                self.workspace, "data/B04.tif", "results/bad.tif", "ndvi",
                {"red": "data/B04.tif", "nir": "data/elsewhere.tif"},
            )
        self.assertIn("grid and CRS", str(caught.exception))

    def test_spectral_index_requires_every_band_an_index_needs(self):
        self.write_grid("data/stack.tif", np.zeros((3, 4, 4), dtype="float32"))

        with self.assertRaises(ToolInputError) as caught:
            raster.spectral_index(
                self.workspace, "data/stack.tif", "results/evi.tif", "evi",
                {"red": 1, "nir": 2},
            )
        self.assertIn("blue", str(caught.exception))
        with self.assertRaises(ToolInputError):
            raster.spectral_index(
                self.workspace, "data/stack.tif", "results/x.tif", "ndwi", {"green": 1}
            )
        with self.assertRaises(ToolInputError) as unknown:
            raster.spectral_index(
                self.workspace, "data/stack.tif", "results/x.tif", "ndgi", {"nir": 1, "red": 2}
            )
        self.assertIn("unknown index", str(unknown.exception))

    def test_spectral_index_scale_only_matters_for_additive_indices(self):
        stacked = np.stack(
            [
                np.full((4, 4), 2000, dtype="uint16"),  # red
                np.full((4, 4), 6000, dtype="uint16"),  # nir
                np.full((4, 4), 1000, dtype="uint16"),  # blue
            ]
        )
        self.write_grid("data/stack.tif", stacked)
        bands = {"red": 1, "nir": 2, "blue": 3}

        raster.spectral_index(
            self.workspace, "data/stack.tif", "results/evi-scaled.tif", "evi", bands, scale=10000
        )
        raster.spectral_index(
            self.workspace, "data/stack.tif", "results/evi-raw.tif", "evi", bands, scale=1
        )
        scaled, _ = self.read_band("results/evi-scaled.tif")
        raw, _ = self.read_band("results/evi-raw.tif")

        # 2.5 * (0.6 - 0.2) / (0.6 + 1.2 - 0.75 + 1)
        self.assertAlmostEqual(float(scaled.mean()), 1.0 / 2.05, places=5)
        self.assertNotAlmostEqual(float(raw.mean()), float(scaled.mean()), places=3)

    def test_spectral_index_marks_masked_pixels_as_nodata(self):
        red = np.full((4, 4), 0.2, dtype="float32")
        nir = np.full((4, 4), 0.6, dtype="float32")
        red[0, 0] = 0.0  # divided out below, leaving a masked pixel behind
        self.write_grid("data/red.tif", red, nodata=0.0)
        self.write_grid("data/nir.tif", nir)

        raster.spectral_index(
            self.workspace, "data/red.tif", "results/ndvi.tif", "ndvi",
            {"red": "data/red.tif", "nir": "data/nir.tif"},
        )
        values, _ = self.read_band("results/ndvi.tif")

        self.assertTrue(np.ma.getmaskarray(values)[0, 0])
        self.assertEqual(int(np.ma.getmaskarray(values).sum()), 1)

    def test_zonal_stats_summarizes_each_zone(self):
        values = np.full((4, 4), 4.0, dtype="float32")
        self.write_grid("data/constant.tif", values, crs="EPSG:4326", pixel=0.25)
        zones = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"name": "west"},
                 "geometry": {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.5, 0.0], [0.5, -1.0], [0.0, -1.0], [0.0, 0.0]]]}},
                {"type": "Feature", "properties": {"name": "east"},
                 "geometry": {"type": "Polygon", "coordinates": [[[0.5, 0.0], [1.0, 0.0], [1.0, -1.0], [0.5, -1.0], [0.5, 0.0]]]}},
                {"type": "Feature", "properties": {"name": "far"},
                 "geometry": {"type": "Polygon", "coordinates": [[[8.0, 8.0], [8.5, 8.0], [8.5, 8.5], [8.0, 8.5], [8.0, 8.0]]]}},
            ],
        }
        path = self.workspace.resolve("data/zones.geojson", write=True)
        path.write_text(json.dumps(zones), encoding="utf-8")

        written = raster.zonal_stats(
            self.workspace, "data/constant.tif", "data/zones.geojson", "results/zone-stats.geojson",
            ["count", "mean", "max"],
        )
        payload = json.loads(written.read_text(encoding="utf-8"))
        rows = {feature["properties"]["name"]: feature["properties"] for feature in payload["features"]}

        self.assertAlmostEqual(rows["west"]["raster_mean"], 4.0, places=5)
        self.assertAlmostEqual(rows["east"]["raster_mean"], 4.0, places=5)
        self.assertEqual(rows["west"]["raster_max"], 4.0)
        self.assertGreater(rows["west"]["raster_count"], 0)
        # A zone outside the raster is reported, not dropped: no pixels, no mean.
        self.assertEqual(rows["far"]["raster_count"], 0)
        self.assertIsNone(rows["far"]["raster_mean"])

    def test_spectral_index_refuses_a_remote_source_it_would_have_to_read_whole(self):
        # The guard is what stops a full catalog tile being pulled into memory for
        # an index; it lives in _source, so every whole-array reader has to ask for it.
        class Huge:
            width = 12_000
            height = 12_000
            count = 1

            def close(self):
                pass

        self.write_grid("data/red.tif", np.ones((4, 4), dtype="float32"))
        real_open = rasterio.open

        def open_remote(path, *args, **kwargs):
            """Stub only the URL opens: the local band still has to be readable."""
            if str(path).startswith("https://"):
                return Huge()
            return real_open(path, *args, **kwargs)

        with patch("spatial_intelligence.geo.raster.rasterio.open", side_effect=open_remote):
            cases = {
                "the index raster itself": ("https://example.com/huge.tif", {
                    "nir": "https://example.com/huge.tif",
                    "red": "https://example.com/huge.tif",
                }),
                "one per-band file": ("data/red.tif", {
                    "nir": "https://example.com/huge.tif",
                    "red": "data/red.tif",
                }),
            }
            for label, (path, bands) in cases.items():
                with self.subTest(case=label):
                    with self.assertRaises(ToolInputError) as caught:
                        raster.spectral_index(
                            self.workspace, path, "results/ndvi.tif", "ndvi", bands
                        )
                    self.assertIn("clip", str(caught.exception))

    def test_zonal_stats_refuses_a_container_with_several_layers(self):
        import geopandas as gpd
        from shapely.geometry import box

        self.write_grid("data/constant.tif", np.full((4, 4), 1.0, dtype="float32"))
        zones = gpd.GeoDataFrame(
            {"name": ["a"]}, geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
        )
        target = self.workspace.resolve("data/zones.gpkg", write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        zones.to_file(target, driver="GPKG", layer="parcels")
        zones.to_file(target, driver="GPKG", layer="wells")

        with self.assertRaises(ToolInputError) as caught:
            raster.zonal_stats(
                self.workspace, "data/constant.tif", "data/zones.gpkg", "results/x.geojson"
            )

        self.assertIn("parcels", str(caught.exception))
        self.assertIn("wells", str(caught.exception))

    def test_zonal_stats_refuses_zones_without_geometry(self):
        self.write_grid("data/constant.tif", np.full((4, 4), 1.0, dtype="float32"))
        target = self.workspace.resolve("data/zones.csv", write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("name,value\na,1\n", encoding="utf-8")

        with self.assertRaises(ToolInputError) as caught:
            raster.zonal_stats(
                self.workspace, "data/constant.tif", "data/zones.csv", "results/x.geojson"
            )

        self.assertIn("no geometry", str(caught.exception))

    def test_a_broken_zone_read_is_not_reported_as_an_empty_zone(self):
        self.write_grid("data/constant.tif", np.full((4, 4), 1.0, dtype="float32"))
        zones = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"name": "a"},
                "geometry": {"type": "Polygon", "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, -1.0], [0.0, -1.0], [0.0, 0.0]]]},
            }],
        }
        path = self.workspace.resolve("data/zones.geojson", write=True)
        path.write_text(json.dumps(zones), encoding="utf-8")

        with patch("spatial_intelligence.geo.raster.mask.mask", side_effect=OSError("disk gone")):
            with self.assertRaises(ToolInputError) as caught:
                raster.zonal_stats(
                    self.workspace, "data/constant.tif", "data/zones.geojson", "results/x.geojson"
                )

        # count = 0 is what a zone covering no data gets, so a read failure must not
        # be folded into that answer.
        self.assertIn("disk gone", str(caught.exception))

    def test_zonal_stats_aligns_zones_from_another_crs(self):
        self.write_grid("data/constant.tif", np.full((4, 4), 2.0, dtype="float32"))
        zones = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"name": "web"},
                 "geometry": {"type": "Polygon", "coordinates": [[[0.0, 0.0], [20.0, 0.0], [20.0, -20.0], [0.0, -20.0], [0.0, 0.0]]]}},
            ],
        }
        import geopandas as gpd
        from shapely.geometry import shape

        path = self.workspace.resolve("data/web-zones.geojson", write=True)
        frame = gpd.GeoDataFrame(
            [{"name": feature["properties"]["name"]} for feature in zones["features"]],
            geometry=[shape(feature["geometry"]) for feature in zones["features"]],
            crs="EPSG:3857",
        )
        frame.to_file(path, driver="GeoJSON")

        written = raster.zonal_stats(
            self.workspace, "data/constant.tif", "data/web-zones.geojson", "results/stats.geojson",
            ["mean"],
        )
        rows = json.loads(written.read_text(encoding="utf-8"))["features"][0]["properties"]

        self.assertAlmostEqual(rows["raster_mean"], 2.0, places=5)

    def test_zonal_stats_rejects_unknown_statistics(self):
        self.write_grid("data/constant.tif", np.full((4, 4), 1.0, dtype="float32"))

        with self.assertRaises(ToolInputError) as caught:
            raster.zonal_stats(
                self.workspace, "data/constant.tif", "data/constant.tif",
                "results/x.geojson", ["mean", "mode"],
            )
        self.assertIn("mode", str(caught.exception))

    def test_hillshade_of_a_flat_dem_is_the_altitude_term(self):
        self.write_grid("data/flat.tif", np.full((4, 4), 100.0, dtype="float32"))

        written = raster.hillshade(self.workspace, "data/flat.tif", "results/flat-shade.tif")
        values, ds = self.read_band("results/flat-shade.tif")

        self.assertEqual(ds.dtypes[0], "uint8")
        # A flat surface shades to 255 * cos(90 - altitude) = 180 at altitude 45.
        self.assertEqual(set(values.compressed().tolist()), {180})
        self.assertIsNone(ds.nodata)

    def test_hillshade_lights_the_slope_facing_the_sun(self):
        rows = np.tile(np.arange(4, dtype="float32") * 10.0, (4, 1))  # rises eastward
        self.write_grid("data/east-facing.tif", rows)

        west_lit = raster.hillshade(
            self.workspace, "data/east-facing.tif", "results/west.tif", azimuth=270.0
        )
        east_lit = raster.hillshade(
            self.workspace, "data/east-facing.tif", "results/east.tif", azimuth=90.0
        )
        lit, _ = self.read_band("results/west.tif")
        shadowed, _ = self.read_band("results/east.tif")

        # Terrain rising to the east faces west, so a western sun is brighter.
        self.assertGreater(float(lit.mean()), float(shadowed.mean()))
        self.assertLess(float(lit.mean()), 256.0)
        self.assertLess(float(shadowed.mean()), float(lit.mean()))
        self.assertTrue(west_lit and east_lit)

    def test_slope_reports_degrees_and_percent_for_a_known_plane(self):
        # 1 m rise per 10 m pixel is a 10 % grade, and 8 columns keep an interior
        # pixel away from the replicated edge, which is half as steep by design.
        rows = np.tile(np.arange(8, dtype="float32") * 1.0, (8, 1))
        self.write_grid("data/ten-percent.tif", rows, pixel=10.0)

        raster.slope(self.workspace, "data/ten-percent.tif", "results/deg.tif", units="degrees")
        raster.slope(self.workspace, "data/ten-percent.tif", "results/pct.tif", units="percent")
        degrees, _ = self.read_band("results/deg.tif")
        percent, _ = self.read_band("results/pct.tif")

        self.assertAlmostEqual(float(degrees[4, 4]), 5.7106, places=3)
        self.assertAlmostEqual(float(percent[4, 4]), 10.0, places=3)

    def test_aspect_reports_the_compass_direction(self):
        east_rises = np.tile(np.arange(4, dtype="float32") * 1.0, (4, 1))
        # Row 0 is the top of the raster (north), so counting down raises terrain
        # southward; reversing it makes the surface rise northward instead.
        north_rises = np.tile(np.arange(3, -1, -1, dtype="float32").reshape(4, 1), (1, 4))
        self.write_grid("data/east-rises.tif", east_rises)
        self.write_grid("data/north-rises.tif", north_rises)

        raster.aspect(self.workspace, "data/east-rises.tif", "results/west-facing.tif")
        raster.aspect(self.workspace, "data/north-rises.tif", "results/south-facing.tif")
        west_facing, _ = self.read_band("results/west-facing.tif")
        south_facing, _ = self.read_band("results/south-facing.tif")

        # Rising to the east means facing west (270), rising north means facing south (180).
        self.assertAlmostEqual(float(west_facing.mean()), 270.0, places=3)
        self.assertAlmostEqual(float(south_facing.mean()), 180.0, places=3)

    def test_terrain_tools_reject_impossible_parameters(self):
        self.write_grid("data/flat.tif", np.full((4, 4), 1.0, dtype="float32"))

        with self.assertRaises(ToolInputError):
            raster.hillshade(self.workspace, "data/flat.tif", "results/x.tif", azimuth=400.0)
        with self.assertRaises(ToolInputError):
            raster.hillshade(self.workspace, "data/flat.tif", "results/x.tif", altitude=-5.0)
        with self.assertRaises(ToolInputError):
            raster.slope(self.workspace, "data/flat.tif", "results/x.tif", z_factor=0.0)
        with self.assertRaises(ToolInputError):
            raster.slope(self.workspace, "data/flat.tif", "results/x.tif", units="radians")
        with self.assertRaises(ToolInputError):
            raster.aspect(self.workspace, "data/flat.tif", "results/x.tif", band=7)

    def test_contour_writes_one_line_per_level_in_the_raster_crs(self):
        rows = np.tile(np.arange(20, dtype="float32") * 5.0, (20, 1)) + 100.0
        self.write_grid(
            "data/ramp.tif", rows, crs="EPSG:32610", origin=(500_000.0, 4_180_000.0), pixel=10.0
        )

        written = raster.contour(
            self.workspace, "data/ramp.tif", "results/contours.geojson", interval=20.0
        )
        payload = json.loads(written.read_text(encoding="utf-8"))
        values = sorted(feature["properties"]["value"] for feature in payload["features"])

        self.assertEqual(values, [100.0, 120.0, 140.0, 160.0, 180.0])
        self.assertTrue(all(f["geometry"]["type"] == "LineString" for f in payload["features"]))
        # Contours come back in the raster's own CRS, so they line up on the map.
        first = payload["features"][0]["geometry"]["coordinates"][0]
        self.assertAlmostEqual(first[0], 500_005.0, places=3)
        self.assertGreaterEqual(first[1], 4_179_800.0)

    def test_contour_simplify_removes_marching_squares_stair_steps(self):
        grid = np.tile(np.arange(20, dtype="float32") * 5.0, (20, 1))
        self.write_grid("data/step.tif", grid)

        plain = raster.contour(
            self.workspace, "data/step.tif", "results/plain.geojson", interval=25.0
        )
        simple = raster.contour(
            self.workspace, "data/step.tif", "results/simple.geojson", interval=25.0, simplify=2.0
        )
        counts = [
            sum(len(f["geometry"]["coordinates"]) for f in json.loads(path.read_text(encoding="utf-8"))["features"])
            for path in (plain, simple)
        ]

        self.assertLess(counts[1], counts[0])

    def test_contour_refuses_nodata_flat_and_impossible_parameters(self):
        rows = np.tile(np.arange(20, dtype="float32") * 5.0, (20, 1))
        self.write_grid("data/ramp.tif", rows)
        self.write_grid("data/flat.tif", np.full((20, 20), 3.0, dtype="float32"))
        self.write_grid("data/holed.tif", rows, nodata=0.0)

        for kwargs, expected in (
            ({"interval": 0}, "positive"),
            # A base above the band's maximum leaves no level to draw, and an
            # interval that would draw millions of them is refused.
            ({"interval": 1e9, "base": 500.0}, "no contour level"),
            ({"interval": 1e-9}, "levels"),
            ({"simplify": -1}, "non-negative"),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ToolInputError) as caught:
                    raster.contour(self.workspace, "data/ramp.tif", "results/x.geojson", **kwargs)
                self.assertIn(expected, str(caught.exception))

        with self.assertRaises(ToolInputError) as flat:
            raster.contour(self.workspace, "data/flat.tif", "results/x.geojson")
        self.assertIn("flat", str(flat.exception))

        # A masked hole would be outlined as if it were data, so it is refused.
        with self.assertRaises(ToolInputError) as holed:
            raster.contour(self.workspace, "data/holed.tif", "results/x.geojson")
        self.assertIn("nodata", str(holed.exception))

    def test_raster_info_reports_whether_a_file_is_a_cog(self):
        # The map loads a COG, so "is this file loadable?" is worth answering.
        self.write_grid("data/striped.tif", np.zeros((40, 40), dtype="float32"))

        striped = raster.raster_info(self.workspace, "data/striped.tif")
        raster.to_cog(self.workspace, "data/striped.tif", "results/cogged.tif")
        cogged = raster.raster_info(self.workspace, "results/cogged.tif")

        self.assertEqual(striped["cog"], False)
        self.assertEqual(cogged["cog"], True)

    def test_looks_cog_is_none_for_something_that_is_not_a_raster(self):
        junk = self.workspace.resolve("data/junk.tif", write=True)
        junk.write_bytes(b"definitely not a tiff")

        self.assertIsNone(raster.looks_cog(junk))

    def test_every_submodule_imports_and_the_registry_builds(self):
        """A fresh interpreter imports every module and assembles the tool registry,
        so an import-time mistake anywhere is caught without the whole suite."""
        script = textwrap.dedent(
            """
            import pathlib, pkgutil, importlib, sys, tempfile

            import spatial_intelligence
            for found in pkgutil.walk_packages(
                spatial_intelligence.__path__, "spatial_intelligence."
            ):
                importlib.import_module(found.name)      # nothing may fail to import

            from spatial_intelligence.geo import raster
            from spatial_intelligence.tools.build import default_registry
            from spatial_intelligence.tools.runtime import ToolRuntime
            from spatial_intelligence.workspace import Workspace

            workspace = Workspace(pathlib.Path(tempfile.mkdtemp()) / "w").create()
            registry = default_registry(ToolRuntime(workspace=workspace))
            assert "gdal_translate" in registry, "the registry lost gdal_translate"
            print("IMPORTED-AND-REGISTERED")
            """
        )

        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )

        self.assertIn("IMPORTED-AND-REGISTERED", done.stdout, done.stderr)

    def test_gdal_translate_writes_the_requested_block_size(self):
        self.write_raster()

        written = raster.gdal_translate(
            self.workspace, SRC, "results/blocked.tif",
            {"TILED": "YES", "BLOCKSIZE": "256", "COMPRESS": "DEFLATE"},
        )

        with rasterio.open(written) as ds:
            self.assertEqual(ds.block_shapes[0], (256, 256))

    def test_gdal_translate_refuses_arguments_that_are_not_creation_options(self):
        # GDAL ignores a creation option it does not know, so the tool refuses
        # these and names the tool that does the job instead of writing a file
        # that quietly ignores the request.
        self.write_raster()
        cases = {
            "OUTPUTTYPE": "run_python",
            "PROJWIN": "clip",
            "SCALEPARAMS": "rescale",
            "out_size": "reproject",
        }

        for option, hint in cases.items():
            with self.subTest(option=option):
                with self.assertRaises(ToolInputError) as caught:
                    raster.gdal_translate(
                        self.workspace, SRC, "results/refused.tif", {option: "1"}
                    )
                self.assertIn(hint, str(caught.exception))
                self.assertFalse((self.workspace.results / "refused.tif").exists())

    def test_derived_rasters_are_written_as_cogs(self):
        # Everything the map may load is written tiled, so a COG client can fetch
        # the screenful it needs instead of the whole file.
        rows = np.tile(np.arange(20, dtype="float32") * 5.0, (20, 1)) + 100.0
        self.write_grid("data/two-band.tif", np.stack([rows, rows * 0.5]))
        for name in ("red", "green", "blue"):
            self.write_grid(f"data/{name}.tif", rows)

        outputs = [
            raster.spectral_index(
                self.workspace, "data/two-band.tif", "results/idx.tif", "ndvi",
                {"red": 1, "nir": 2},
            ),
            raster.slope(self.workspace, "data/two-band.tif", "results/slope.tif"),
            raster.aspect(self.workspace, "data/two-band.tif", "results/aspect.tif"),
            raster.hillshade(self.workspace, "data/two-band.tif", "results/hs.tif"),
            raster.compose_rgb(
                self.workspace, "data/red.tif", "data/green.tif", "data/blue.tif",
                "results/rgb.tif",
            ),
        ]

        for path in outputs:
            with self.subTest(path=path.name):
                with rasterio.open(path) as ds:
                    self.assertTrue(ds.is_tiled, f"{path.name} is striped, so the app cannot COG-read it")
                    # ``Compression.none`` is truthy, so assert the compressor.
                    self.assertEqual(ds.compression.name.lower(), "deflate")

    def test_to_cog_if_needed_converts_only_a_striped_raster(self):
        self.write_grid("data/striped.tif", np.zeros((40, 40), dtype="float32"))
        striped = self.workspace.resolve("data/striped.tif", must_exist=True)

        converted_path, converted = raster.to_cog_if_needed(self.workspace, "data/striped.tif")

        self.assertTrue(converted)
        # The copy lands in results/ with every other output, not beside the source
        # in data/, so it can be recorded and versioned like one.
        self.assertEqual(converted_path, self.workspace.results / "striped_cog.tif")
        self.assertFalse((self.workspace.data / "striped_cog.tif").exists())
        with rasterio.open(converted_path) as ds:
            self.assertTrue(ds.is_tiled)
        self.assertTrue(striped.is_file())

        again_path, again = raster.to_cog_if_needed(self.workspace, "results/striped_cog.tif")
        self.assertFalse(again)
        self.assertEqual(again_path, converted_path)

        # A file that is not a raster at all is passed through, not crashed on.
        junk = self.workspace.resolve("data/not-a-raster.tif", write=True)
        junk.write_bytes(b"not a tiff")
        junk_path, junk_converted = raster.to_cog_if_needed(self.workspace, "data/not-a-raster.tif")
        self.assertFalse(junk_converted)
        self.assertEqual(junk_path, junk)

    def test_polygonize_returns_one_polygon_per_class(self):
        classes = np.zeros((4, 4), dtype="uint8")
        classes[:, 2:] = 1
        self.write_grid("data/classes.tif", classes)

        written = raster.polygonize(self.workspace, "data/classes.tif", "results/classes.geojson")
        payload = json.loads(written.read_text(encoding="utf-8"))
        by_value = {feature["properties"]["value"]: feature for feature in payload["features"]}

        self.assertEqual(sorted(by_value), [0, 1])
        self.assertEqual(len(payload["features"]), 2)

    def test_compose_rgb_stacks_three_bands_into_uint8(self):
        grid = np.tile(np.arange(4, dtype="float32") * 10.0, (4, 1))
        for name in ("red", "green", "blue"):
            self.write_grid(f"data/{name}.tif", grid)

        written = raster.compose_rgb(
            self.workspace, "data/red.tif", "data/green.tif", "data/blue.tif", "results/rgb.tif"
        )
        with rasterio.open(written) as ds:
            data = ds.read()

        self.assertEqual(data.shape[0], 3)
        self.assertEqual(ds.dtypes[0], "uint8")
        self.assertEqual(int(data.min()), 0)
        self.assertEqual(int(data.max()), 255)

    def test_compose_rgb_refuses_bands_on_different_grids(self):
        grid = np.tile(np.arange(4, dtype="float32"), (4, 1))
        self.write_grid("data/red.tif", grid)
        self.write_grid("data/green.tif", grid)
        self.write_grid("data/small-blue.tif", np.zeros((3, 3), dtype="float32"))

        with self.assertRaises(ToolInputError) as caught:
            raster.compose_rgb(
                self.workspace, "data/red.tif", "data/green.tif", "data/small-blue.tif",
                "results/rgb.tif",
            )

        self.assertIn("grid and CRS", str(caught.exception))

    def test_clip_takes_lng_lat_bounds_for_a_projected_raster(self):
        # A UTM tile is what a catalog serves; the AOI is lng/lat, as everywhere else.
        self.write_grid(
            "data/utm.tif",
            np.arange(30 * 40, dtype="float32").reshape(30, 40),
            crs="EPSG:32610",
            origin=(500_000.0, 4_180_000.0),
            pixel=10.0,
        )
        from rasterio.warp import transform_bounds

        west, south, east, north = transform_bounds(
            "EPSG:32610", "EPSG:4326", 500_000.0, 4_179_700.0, 500_400.0, 4_180_000.0
        )

        written = raster.clip(
            self.workspace, "data/utm.tif", "results/utm-aoi.tif", [west, south, east, north]
        )
        with rasterio.open(written) as ds:
            self.assertEqual(str(ds.crs), "EPSG:32610")
            # A lng/lat box over the whole tile clips the whole tile, not 0x0.
            self.assertEqual((ds.width, ds.height), (40, 30))

    def test_clip_accepts_projected_bounds_when_told_the_crs(self):
        self.write_grid(
            "data/utm.tif", np.zeros((30, 40), dtype="float32"),
            crs="EPSG:32610", origin=(500_000.0, 4_180_000.0), pixel=10.0,
        )

        written = raster.clip(
            self.workspace, "data/utm.tif", "results/corner.tif",
            [500_000.0, 4_179_800.0, 500_200.0, 4_180_000.0],
            bounds_crs="EPSG:32610",
        )
        with rasterio.open(written) as ds:
            self.assertEqual((ds.width, ds.height), (20, 20))

    def test_clip_refuses_projected_numbers_left_as_lng_lat(self):
        self.write_grid(
            "data/utm.tif", np.zeros((30, 40), dtype="float32"),
            crs="EPSG:32610", origin=(500_000.0, 4_180_000.0), pixel=10.0,
        )

        with self.assertRaises(ToolInputError) as caught:
            raster.clip(
                self.workspace, "data/utm.tif", "results/x.tif",
                [500_000.0, 4_179_800.0, 500_200.0, 4_180_000.0],
            )

        self.assertIn("bounds_crs", str(caught.exception))

    def test_clip_refuses_bounds_that_miss_the_raster_and_clamps_partial_ones(self):
        self.write_raster()

        with self.assertRaises(ToolInputError) as caught:
            raster.clip(self.workspace, SRC, "results/x.tif", [20.0, 20.0, 21.0, 21.0])
        self.assertIn("do not overlap", str(caught.exception))

        # A box that hangs over the edge keeps the part that exists: this raster
        # spans y 0.25…1.0, so a box reaching down to -1.0 keeps its bottom row.
        written = raster.clip(self.workspace, SRC, "results/edge.tif", [-1.0, -1.0, 0.5, 0.5])
        with rasterio.open(written) as ds:
            self.assertEqual((ds.width, ds.height), (2, 1))
            self.assertEqual(list(ds.bounds), [0.0, 0.25, 0.5, 0.5])


class RemoteRasterTests(RasterTestCase):
    """Reading a COG by URL: GDAL range reads, and the whole-array guard."""
    class FakeDataset:
        """Just enough of a rasterio dataset for the metadata and guard paths."""

        def __init__(self, width: int, height: int):
            from rasterio.crs import CRS

            self.width = width
            self.height = height
            self.count = 2
            self.dtypes = ("uint16", "uint16")
            self.nodata = None
            self.crs = CRS.from_epsg(4326)
            self.transform = from_origin(0.0, 1.0, 1.0, 1.0)
            self.bounds = (0.0, 0.0, float(width), float(height))
            self.closed = False

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def test_a_url_is_opened_directly_and_a_relative_path_is_not(self):
        self.assertTrue(raster._is_remote("https://example.com/a.tif"))
        self.assertTrue(raster._is_remote("http://example.com/a.tif"))
        self.assertFalse(raster._is_remote("data/a.tif"))
        self.assertFalse(raster._is_remote("results/a.tif"))

        opened: list[str] = []
        with patch("rasterio.open", side_effect=lambda target, *a, **k: (opened.append(str(target)), self.FakeDataset(10, 10))[1]):
            with raster._source(self.workspace, "https://example.com/a.tif"):
                pass

        self.assertEqual(opened, ["https://example.com/a.tif"])

    def test_raster_info_reads_a_remote_cog_without_a_workspace_copy(self):
        dataset = self.FakeDataset(4, 3)
        with patch("rasterio.open", return_value=dataset):
            info = raster.raster_info(self.workspace, "https://example.com/a.tif")

        self.assertEqual((info["width"], info["height"]), (4, 3))
        self.assertEqual(info["crs"], "EPSG:4326")
        self.assertEqual(self.manifest_outputs(), [])

    def test_a_whole_array_read_of_a_large_remote_source_asks_for_a_clip(self):
        huge = self.FakeDataset(12000, 12000)
        with patch("rasterio.open", return_value=huge):
            with self.assertRaises(ToolInputError) as caught:
                raster.raster_stats(self.workspace, "https://example.com/tile.tif")

        message = str(caught.exception)
        self.assertIn("12000x12000", message)
        self.assertIn("clip", message)
        self.assertTrue(huge.closed)

    def test_a_whole_array_read_of_a_small_remote_source_is_allowed(self):
        small = self.FakeDataset(100, 100)
        small.read = lambda band, masked=True: __import__("numpy").ma.masked_array(
            __import__("numpy").full((100, 100), 3.0), mask=False
        )
        with patch("rasterio.open", return_value=small):
            stats = raster.raster_stats(self.workspace, "https://example.com/small.tif")

        self.assertEqual(stats["mean"], 3.0)

    def test_a_windowed_tool_is_never_blocked_by_the_size_guard(self):
        # clip reads only its window, so a huge remote tile is exactly its job.
        huge = self.FakeDataset(12000, 12000)
        with patch("rasterio.open", return_value=huge):
            with raster._source(self.workspace, "https://example.com/tile.tif") as src:
                self.assertEqual(src.width, 12000)


class RasterPackTests(RasterTestCase):
    def build_pack(self, sink=None, notifications=None, approved=False):
        runtime = ToolRuntime(
            workspace=self.workspace,
            reporter=Reporter(sink or RecordingSink(), parent_id="cell-7"),
            events=RuntimeEvents(
                files_changed=lambda: (notifications if notifications is not None else []).append(
                    "files"
                )
            ),
            approved=approved,
        )
        registry = ToolRegistry()
        registry.add_pack(RasterPack, runtime)
        return registry
    def done_events(self, sink):
        return [event for event in sink.events if event["status"] == JobState.DONE.value]
    def assert_recorded(self, relative: str) -> Path:
        path = self.workspace.resolve(relative)
        self.assertTrue(path.is_file(), f"{relative} was not written")
        self.assertIn(relative, self.manifest_outputs())
        return path

    def test_registered_raster_tools_expose_expected_effects(self):
        registry = self.build_pack()

        self.assertEqual(
            sorted(registry.names()),
            [
                "aspect",
                "band_math",
                "clip",
                "compose_rgb",
                "contour",
                "gdal_translate",
                "hillshade",
                "polygonize",
                "raster_info",
                "raster_stats",
                "reproject",
                "rescale",
                "sample_point",
                "slope",
                "spectral_index",
                "to_cog",
                "zonal_stats",
            ],
        )
        for name in ("raster_info", "raster_stats", "sample_point"):
            spec = registry.get(name)
            self.assertEqual(spec.effects, frozenset({Effect.READ}), name)
            self.assertEqual(spec.category, "raster")
            self.assertTrue(registry.replay_safe(name), name)
        for name in ("to_cog", "reproject", "clip", "rescale", "band_math", "gdal_translate"):
            spec = registry.get(name)
            self.assertEqual(spec.effects, frozenset({Effect.WORKSPACE_WRITE}), name)
            self.assertFalse(registry.replay_safe(name), name)
        for name in ("to_cog", "reproject", "clip", "rescale"):
            self.assertEqual(registry.get(name).kind.value, "reporting", name)
        self.assertTrue(registry.get("band_math").requires_approval)
        self.assertFalse(registry.get("to_cog").requires_approval)
        self.assertEqual(registry.category_of("rescale"), "raster")

    def test_to_cog_reports_band_progress_and_records_the_artifact(self):
        self.write_raster(count=2)
        sink = RecordingSink()
        notifications: list[str] = []
        registry = self.build_pack(sink, notifications)

        absolute = registry.get("to_cog").callable(SRC, "results/scene_cog.tif")

        written = self.assert_recorded("results/scene_cog.tif")
        self.assertEqual(Path(absolute), written)
        self.assertEqual(notifications, ["files"])
        with rasterio.open(written) as ds:
            self.assertEqual(ds.count, 2)

        events = sink.events
        self.assertEqual({event["kind"] for event in events}, {"raster"})
        self.assertEqual(events[0]["label"], "to_cog results/scene_cog.tif")
        self.assertEqual(events[0]["unit"], "steps")
        self.assertEqual(events[0]["total"], 2)
        self.assertEqual(events[0]["parent_id"], "cell-7")
        # One report per band, then the terminal event carrying the artifact.
        self.assertEqual(
            [event["completed"] for event in events if event["status"] == "running"],
            [0.0, 1.0, 2.0],
        )
        final = events[-1]
        self.assertEqual(final["status"], JobState.DONE.value)
        self.assertEqual(final["artifact"], "results/scene_cog.tif")
        self.assertEqual(final["completed"], 2.0)

    def test_reproject_reports_a_done_job_with_the_artifact(self):
        self.write_raster(count=2)
        sink = RecordingSink()
        registry = self.build_pack(sink)

        registry.get("reproject").callable(SRC, "results/scene_3857.tif", "EPSG:3857")

        written = self.assert_recorded("results/scene_3857.tif")
        with rasterio.open(written) as ds:
            self.assertEqual(ds.crs.to_epsg(), 3857)
        done = self.done_events(sink)
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["artifact"], "results/scene_3857.tif")
        self.assertEqual(done[0]["total"], 2)

    def test_clip_reports_a_done_job_with_the_artifact(self):
        self.write_raster()
        sink = RecordingSink()
        registry = self.build_pack(sink)

        registry.get("clip").callable(SRC, "results/clip.tif", [0.0, 0.25, 0.5, 0.75])

        written = self.assert_recorded("results/clip.tif")
        with rasterio.open(written) as ds:
            self.assertEqual((ds.width, ds.height), (2, 2))
        done = self.done_events(sink)
        self.assertEqual([event["artifact"] for event in done], ["results/clip.tif"])
        self.assertEqual(done[0]["total"], 1)

    def test_clip_without_bounds_or_mask_fails_the_job(self):
        self.write_raster()
        sink = RecordingSink()
        registry = self.build_pack(sink)

        with self.assertRaises(WorkspaceError):
            registry.get("clip").callable(SRC, "results/clip.tif")

        self.assertEqual(sink.statuses()[-1], JobState.ERROR.value)
        self.assertIn("bounds or mask_geojson", sink.events[-1]["error"])

    def test_rescale_reports_a_done_job_with_the_artifact(self):
        self.write_raster(values=np.arange(12, dtype="float32").reshape(1, 3, 4))
        sink = RecordingSink()
        registry = self.build_pack(sink)

        registry.get("rescale").callable(SRC, "results/stretch.tif")

        written = self.assert_recorded("results/stretch.tif")
        with rasterio.open(written) as ds:
            self.assertEqual(ds.dtypes, ("uint8",))
        done = self.done_events(sink)
        self.assertEqual([event["artifact"] for event in done], ["results/stretch.tif"])

    def test_missing_source_fails_the_rescale_job(self):
        sink = RecordingSink()
        registry = self.build_pack(sink)

        with self.assertRaises(WorkspaceError):
            registry.get("rescale").callable("data/missing.tif", "results/stretch.tif")

        self.assertEqual(sink.statuses()[-1], JobState.ERROR.value)
        self.assertIn("missing.tif", sink.events[-1]["error"])

    def test_band_math_refuses_to_run_without_approval(self):
        self.write_raster(count=2)
        registry = self.build_pack()

        with self.assertRaises(ToolInputError) as caught:
            registry.get("band_math").callable(
                SRC, "results/index.tif", "b2 - b1", {"b1": 1, "b2": 2}
            )

        self.assertIn("band_math", str(caught.exception))
        self.assertIn("run_python", str(caught.exception))
        self.assertFalse((self.workspace.results / "index.tif").exists())
        self.assertEqual(self.manifest_outputs(), [])

    def test_band_math_computes_the_index_once_approved(self):
        values = np.arange(24, dtype="float32").reshape(2, 3, 4)
        self.write_raster(values=values, count=2)
        registry = self.build_pack(approved=True)

        absolute = registry.get("band_math").callable(
            SRC, "results/ndvi.tif", "(b2 - b1) / (b2 + b1)", {"b1": 1, "b2": 2}
        )

        self.assert_recorded("results/ndvi.tif")
        with rasterio.open(absolute) as ds:
            bands = values.astype("float64")
            expected = (bands[1] - bands[0]) / (bands[1] + bands[0])
            np.testing.assert_allclose(ds.read(1), expected)

    def test_sample_point_returns_the_value_without_recording_anything(self):
        self.write_raster(values=np.arange(12, dtype="float32").reshape(1, 3, 4))
        notifications: list[str] = []
        registry = self.build_pack(notifications=notifications)

        sampled = registry.get("sample_point").callable(SRC, 0.375, 0.875)

        self.assertEqual(sampled, {"value": 1.0, "lng": 0.375, "lat": 0.875})
        self.assertEqual(notifications, [])
        self.assertEqual(self.manifest_outputs(), [])


if __name__ == "__main__":
    unittest.main()
