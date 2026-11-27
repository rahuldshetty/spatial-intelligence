"""Super-resolution: tiling geometry, normalization, dates, and the written COG."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from spatial_intelligence.ai import superres
from spatial_intelligence.ai.catalog import ModelSpec
from spatial_intelligence.ai.manager import ModelSession
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.geo.raster import looks_cog
from spatial_intelligence.workspace import Workspace

RGB = ("tci", "tci", "tci")


def spec(*, dates: int = 1, bands: tuple[str, ...] = RGB) -> ModelSpec:
    return ModelSpec(
        id="fake-sr",
        repo="example/fake",
        revision="rev",
        files=(),
        task="super-resolution",
        prompts=(),
        scale=4,
        dates=dates,
        input_bands=bands,
        unload_when_idle=True,
    )


class Nearest4:
    """A 4x graph that repeats the first three channels, for geometry tests."""

    def __init__(self):
        self.calls = 0
        self.shapes = []

    def get_inputs(self):
        return (SimpleNamespace(name="input"),)

    def run(self, _outputs, feeds):
        self.calls += 1
        array = feeds["input"]
        self.shapes.append(array.shape)
        return [np.repeat(np.repeat(array[:, :3], 4, axis=2), 4, axis=3)]


def session(graph, model: ModelSpec) -> ModelSession:
    return ModelSession(
        spec=model, encoder=graph, decoder=graph, provider="CPUExecutionProvider", threads=1
    )


def write_raster(path: Path, width: int, height: int, *, count: int = 3, dtype: str = "uint8"):
    rng = np.random.default_rng(7)
    shape = (count, height, width)
    if dtype == "uint8":
        array = rng.integers(20, 230, size=shape, dtype=np.uint8)
    else:
        array = rng.integers(100, 8000, size=shape, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs="EPSG:32645",
        transform=from_origin(500_000, 3_000_000, 10, 10),
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

    def test_normalize_scales_by_dtype(self):
        self.assertAlmostEqual(float(superres.normalize(np.array([[255]], np.uint8))[0, 0]), 1.0)
        self.assertAlmostEqual(
            float(superres.normalize(np.array([[8160]], np.uint16))[0, 0]), 1.0
        )
        self.assertEqual(
            float(superres.normalize(np.array([[16320]], np.uint16))[0, 0]), 1.0
        )
        self.assertAlmostEqual(float(superres.normalize(np.array([[0.25]], np.float32))[0, 0]), 0.25)


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
            self.workspace, self.path, session(graph, spec()), tile_size=128, halo=16
        )

        with rasterio.open(target) as dataset:
            output = dataset.read()
        expected = np.repeat(np.repeat(self.source, 4, axis=1), 4, axis=2)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(graph.calls, 9)
        self.assertEqual(timings.tiles, 9)
        self.assertEqual(timings.skipped_tiles, 0)
        self.assertEqual(info["output_pixels"], [1200, 1040])

    def test_the_output_keeps_the_grid_and_pixel_size(self):
        target, _timings, info = superres.super_resolve_raster(
            self.workspace, self.path, session(Nearest4(), spec()), tile_size=128, halo=8
        )

        with rasterio.open(target) as dataset:
            self.assertEqual(dataset.crs.to_string(), "EPSG:32645")
            self.assertEqual((dataset.width, dataset.height), (1200, 1040))
            self.assertAlmostEqual(dataset.transform.a, 10 / 4)
            self.assertAlmostEqual(dataset.transform.e, -10 / 4)
            self.assertTrue(dataset.is_tiled)
            self.assertGreaterEqual(min(dataset.block_shapes[0]), 256)
            self.assertTrue(dataset.overviews(1))
        self.assertIs(looks_cog(target), True)
        self.assertEqual(info["pixel_size"], [2.5, 2.5])

    def test_bounds_narrow_the_output_to_the_window(self):
        target, _timings, info = superres.super_resolve_raster(
            self.workspace,
            self.path,
            session(Nearest4(), spec()),
            bounds=[500_100, 2_999_900, 500_300, 3_000_000],
            tile_size=128,
            halo=8,
        )

        with rasterio.open(target) as dataset:
            output = dataset.read()
            self.assertEqual((dataset.width, dataset.height), (80, 40))
            self.assertAlmostEqual(dataset.bounds.left, 500_100)
            self.assertAlmostEqual(dataset.bounds.bottom, 2_999_900)
            self.assertAlmostEqual(dataset.bounds.right, 500_300)
            self.assertAlmostEqual(dataset.bounds.top, 3_000_000)
        expected = np.repeat(np.repeat(self.source[:, :10, 10:30], 4, axis=1), 4, axis=2)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(info["source_pixels"], [20, 10])
        self.assertEqual(info["output_pixels"], [80, 40])

    def test_two_dates_are_concatenated_and_must_share_a_grid(self):
        other = "data/other.tif"
        write_raster(self.workspace.resolve(other, write=True), 300, 260)
        graph = Nearest4()

        superres.super_resolve_raster(
            self.workspace,
            self.path,
            session(graph, spec(dates=2)),
            dates=[other],
            tile_size=128,
            halo=8,
        )

        self.assertEqual(graph.shapes[0], (1, 6, 144, 144))

    def test_a_date_on_another_grid_is_refused(self):
        other = "data/other.tif"
        write_raster(self.workspace.resolve(other, write=True), 128, 128)

        with self.assertRaises(ToolInputError):
            superres.super_resolve_raster(
                self.workspace,
                self.path,
                session(Nearest4(), spec(dates=2)),
                dates=[other],
            )

    def test_the_model_date_count_is_enforced(self):
        with self.assertRaises(ToolInputError):
            superres.super_resolve_raster(
                self.workspace, self.path, session(Nearest4(), spec(dates=2))
            )

    def test_max_tiles_refuses_before_writing(self):
        out = "results/limited.tif"

        with self.assertRaises(ToolInputError):
            superres.super_resolve_raster(
                self.workspace,
                self.path,
                session(Nearest4(), spec()),
                tile_size=64,
                max_tiles=2,
                out=out,
            )
        self.assertFalse(self.workspace.resolve(out, write=True).exists())


if __name__ == "__main__":
    unittest.main()
