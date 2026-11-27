"""SEN2-SR: raw-DN contract, tiling, and the written four-band COG."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from spatial_intelligence.ai import catalog, superres
from spatial_intelligence.ai.manager import ModelSession
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.geo.raster import looks_cog
from spatial_intelligence.workspace import Workspace

MODEL = catalog.SEN2_SR


class Nearest4:
    """A 4x model that repeats all four channels, for geometry tests."""

    def __init__(self):
        self.calls = 0
        self.shapes = []

    def predict_step(self, tensor):
        self.calls += 1
        self.shapes.append(tuple(tensor.shape))
        return tensor.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)


def session(graph) -> ModelSession:
    return ModelSession(
        spec=MODEL,
        encoder=graph,
        decoder=None,
        provider="cpu",
        threads=1,
    )


def write_raster(
    path: Path,
    width: int,
    height: int,
    *,
    count: int = 4,
    dtype: str = "uint16",
    crs: str | None = "EPSG:32645",
    pixel: float = 10.0,
):
    rng = np.random.default_rng(7)
    shape = (count, height, width)
    if dtype == "uint16":
        array = rng.integers(100, 9000, size=shape, dtype=np.uint16)
    else:
        array = rng.integers(20, 230, size=shape, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=from_origin(500_000, 3_000_000, pixel, pixel),
    ) as dataset:
        dataset.write(array)
    return array


class GeometryTests(unittest.TestCase):
    def test_cores_partition_the_raster(self):
        cores = superres.plan_cores(300, 260, 128)
        covered = np.zeros((260, 300), dtype=np.uint8)

        for x, y, width, height in cores:
            covered[y : y + height, x : x + width] += 1

        self.assertTrue(np.all(covered == 1))
        self.assertEqual(cores[0], (0, 0, 128, 128))
        self.assertEqual(cores[-1], (256, 256, 44, 4))


class RasterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.path = "data/scene.tif"
        self.source = write_raster(self.workspace.resolve(self.path, write=True), 300, 260)

    def tearDown(self):
        self._tmp.cleanup()

    def test_tiles_reassemble_into_the_nearest_upsample(self):
        graph = Nearest4()

        target, timings, info = superres.super_resolve_raster(
            self.workspace, self.path, session(graph)
        )

        with rasterio.open(target) as dataset:
            output = dataset.read()
        expected = np.repeat(np.repeat(self.source, 4, axis=1), 4, axis=2)
        self.assertEqual(output.shape, expected.shape)
        self.assertLessEqual(int(np.abs(output.astype(np.int32) - expected).max()), 1)
        self.assertEqual(graph.calls, 9)
        self.assertEqual(graph.shapes[0], (1, 4, 160, 160))
        self.assertEqual(timings.tiles, 9)
        self.assertEqual(timings.skipped_tiles, 0)
        self.assertEqual(info["output_pixels"], [1200, 1040])

    def test_the_output_keeps_the_grid_and_pixel_size(self):
        target, _timings, info = superres.super_resolve_raster(
            self.workspace, self.path, session(Nearest4())
        )

        with rasterio.open(target) as dataset:
            self.assertEqual(dataset.crs.to_string(), "EPSG:32645")
            self.assertEqual((dataset.width, dataset.height), (1200, 1040))
            self.assertEqual(dataset.count, 4)
            self.assertEqual(dataset.dtypes, ("uint16",) * 4)
            self.assertEqual(dataset.nodata, 0)
            self.assertEqual(dataset.interleaving, rasterio.enums.Interleaving.band)
            self.assertAlmostEqual(dataset.transform.a, 10 / 4)
            self.assertAlmostEqual(dataset.transform.e, -10 / 4)
            self.assertTrue(dataset.is_tiled)
            self.assertGreaterEqual(min(dataset.block_shapes[0]), 256)
            self.assertTrue(dataset.overviews(1))
        self.assertIs(looks_cog(target), True)
        self.assertEqual(info["pixel_size"], [2.5, 2.5])
        self.assertEqual(info["map_bands"], [1, 2, 3])

    def test_a_three_band_or_uint8_raster_is_refused(self):
        three = "data/three.tif"
        write_raster(self.workspace.resolve(three, write=True), 64, 64, count=3)
        with self.assertRaises(ToolInputError) as caught:
            superres.super_resolve_raster(self.workspace, three, session(Nearest4()))
        self.assertIn("takes 4", str(caught.exception))

        eight = "data/eight.tif"
        write_raster(
            self.workspace.resolve(eight, write=True), 64, 64, count=4, dtype="uint8"
        )
        with self.assertRaises(ToolInputError) as caught:
            superres.super_resolve_raster(self.workspace, eight, session(Nearest4()))
        self.assertIn("uint16", str(caught.exception))

    def test_a_non_ten_metre_grid_is_refused(self):
        path = "data/twenty.tif"
        write_raster(self.workspace.resolve(path, write=True), 64, 64, pixel=20.0)

        with self.assertRaises(ToolInputError) as caught:
            superres.super_resolve_raster(self.workspace, path, session(Nearest4()))
        self.assertIn("10 m", str(caught.exception))

    def test_a_raster_without_a_crs_is_refused(self):
        path = "data/no-crs.tif"
        write_raster(self.workspace.resolve(path, write=True), 64, 64, crs=None)

        with self.assertRaises(ToolInputError) as caught:
            superres.super_resolve_raster(self.workspace, path, session(Nearest4()))
        self.assertIn("no CRS", str(caught.exception))

    def test_a_standard_s2_stack_needs_explicit_band_indexes(self):
        path = "data/twelve.tif"
        write_raster(self.workspace.resolve(path, write=True), 64, 64, count=12)

        with self.assertRaises(ToolInputError) as caught:
            superres.super_resolve_raster(self.workspace, path, session(Nearest4()))
        self.assertIn("Pass bands", str(caught.exception))

        target, _timings, _info = superres.super_resolve_raster(
            self.workspace, path, session(Nearest4()), bands=[4, 3, 2, 8]
        )
        with rasterio.open(target) as dataset:
            self.assertEqual(dataset.count, 4)

    def test_max_tiles_refuses_before_writing(self):
        out = "results/limited.tif"

        with self.assertRaises(ToolInputError):
            superres.super_resolve_raster(
                self.workspace,
                self.path,
                session(Nearest4()),
                max_tiles=2,
                out=out,
            )
        self.assertFalse(self.workspace.resolve(out, write=True).exists())


if __name__ == "__main__":
    unittest.main()
