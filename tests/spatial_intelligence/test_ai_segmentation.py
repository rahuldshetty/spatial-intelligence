"""Segmentation: preprocessing, mask geometry, tiling, and the mask filters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from spatial_intelligence.ai import catalog
from spatial_intelligence.ai import segmentation as seg
from spatial_intelligence.ai.manager import EMBEDDING_CACHE_SIZE, ModelSession
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.workspace import Workspace

#: A 512x512 RGB scene at one metre per pixel, so a mask pixel is a metre.
SIZE = 512


class FakeEncoder:
    """An encoder that returns embeddings of the shape the pipeline expects."""

    def run(self, _outputs, feeds):
        batch = feeds["pixel_values"].shape[0]
        return (
            np.zeros((batch, 256, 64, 64), dtype=np.float32),
            np.zeros((batch, 256, 64, 64), dtype=np.float32),
        )


class FakeDecoder:
    """A decoder that answers every prompt with one rectangle on the mask grid."""

    def __init__(self, box=(80, 120, 100, 140), score=0.95, whole_tile=False):
        self.box = box
        self.score = score
        self.whole_tile = whole_tile

    def run(self, _outputs, feeds):
        prompts = feeds["input_points"].shape[1]
        iou = np.full((1, prompts, 3), 0.5, dtype=np.float32)
        iou[:, :, 0] = self.score
        masks = np.full((1, prompts, 3, 256, 256), -5.0, dtype=np.float32)
        if self.whole_tile:
            masks[:, :, 0, :, :] = 5.0
        else:
            top, bottom, left, right = self.box
            masks[:, :, 0, top:bottom, left:right] = 5.0
        return iou, masks


def make_session(decoder: FakeDecoder) -> ModelSession:
    """A session with fake ONNX sessions, sharing the real catalog spec."""
    return ModelSession(
        spec=catalog.find("slimsam-77"),
        encoder=FakeEncoder(),
        decoder=decoder,
        provider="CPUExecutionProvider",
        threads=1,
    )


class PreprocessTestCase(unittest.TestCase):
    def test_longest_edge_maps_onto_the_canvas(self):
        rgb = np.full((100, 200, 3), 128, dtype=np.uint8)

        pixel_values, scale = seg.preprocess(rgb, 1024)

        self.assertEqual(pixel_values.shape, (1, 3, 1024, 1024))
        self.assertAlmostEqual(scale, 1024 / 200)
        # The image occupies the top-left corner of the padded canvas...
        self.assertAlmostEqual(float(pixel_values[0, 0, 10, 10]), (128 / 255 - 0.485) / 0.229, places=5)
        # ...and the padding is the normalized zero the processor pads with.
        self.assertAlmostEqual(float(pixel_values[0, 2, 1000, 1000]), (0 - 0.406) / 0.225, places=5)

    def test_one_pixel_window_is_refused(self):
        with self.assertRaises(ToolInputError):
            seg.preprocess(np.zeros((0, 0, 3), dtype=np.uint8), 1024)

    def test_float_input_is_stretched_to_eight_bit(self):
        values = np.linspace(1000.0, 2000.0, 16 * 16 * 3).reshape(16, 16, 3)

        stretched = seg.stretch_to_uint8(values)

        self.assertEqual(stretched.dtype, np.uint8)
        self.assertLess(int(stretched.min()), 10)
        self.assertGreater(int(stretched.max()), 245)


class MaskGeometryTestCase(unittest.TestCase):
    def test_mask_lands_on_the_tile_pixels_it_came_from(self):
        mask = np.zeros((256, 256), dtype=bool)
        mask[64:128, 32:96] = True  # quarter-scale pixels 64..127 rows, 32..95 cols
        scale = 1024 / 512  # a 512-pixel tile

        tile = seg.mask_to_tile(mask, scale, 512, 512)

        self.assertEqual(tile.shape, (512, 512))
        rows, cols = np.nonzero(tile)
        self.assertEqual((rows.min(), rows.max()), (128, 255))
        self.assertEqual((cols.min(), cols.max()), (64, 191))

    def test_polygonize_places_the_polygon_in_the_raster_crs(self):
        tile_mask = np.zeros((100, 100), dtype=bool)
        tile_mask[10:30, 40:60] = True
        transform = from_origin(500_000, 3_000_000, 1, 1)  # 1 m pixels, UTM-like

        polygons = seg.polygonize(tile_mask, transform)

        self.assertEqual(len(polygons), 1)
        minx, miny, maxx, maxy = polygons[0].bounds
        self.assertAlmostEqual(minx, 500_040)
        self.assertAlmostEqual(maxx, 500_060)
        self.assertAlmostEqual(maxx - minx, 20)


class FilterTestCase(unittest.TestCase):
    def test_stability_is_high_for_a_confident_mask(self):
        logits = np.full((32, 32), 10.0, dtype=np.float32)

        self.assertAlmostEqual(seg.stability_score(logits), 1.0)

    def test_box_nms_keeps_the_best_of_overlapping_boxes(self):
        # Boxes 1 and 2 overlap by IoU 0.82, past the 0.7 suppression threshold.
        boxes = np.array([[0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5], [50, 50, 60, 60]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])

        self.assertEqual(seg.box_nms(boxes, scores), [0, 2])

    def test_manual_prompts_land_where_they_are_given(self):
        transform = from_origin(0, 1000, 1, 1)  # 1 m pixels, north-up

        points, labels = seg._point_prompts(
            [[100, 900]], transform, (0, 0), 2.0, "EPSG:32645", "EPSG:32645"
        )

        self.assertEqual(labels.tolist(), [[1]])
        self.assertAlmostEqual(points[0, 0, 0], 200.0)
        self.assertAlmostEqual(points[0, 0, 1], 200.0)

    def test_boxes_become_two_labelled_corners(self):
        transform = from_origin(0, 1000, 1, 1)

        points, labels = seg._box_points(
            [[0, 800, 100, 1000]], transform, (0, 0), 1.0, "EPSG:32645", "EPSG:32645"
        )

        self.assertEqual(points.shape, (1, 2, 2))
        self.assertEqual(labels.tolist(), [[2, 3]])


class TilingTestCase(unittest.TestCase):
    def test_a_small_raster_is_one_tile(self):
        self.assertEqual(seg.plan_tiles(300, 200, 1024, 128), [(0, 0, 300, 200)])

    def test_tiles_cover_the_raster_in_even_steps(self):
        tiles = seg.plan_tiles(3000, 2000, 1024, 128)

        xs = sorted({x for x, _, _, _ in tiles})
        self.assertEqual(xs[0], 0)
        self.assertEqual(max(x + w for x, _, w, _ in tiles), 3000)
        self.assertLessEqual(len(xs) * len({y for _, y, _, _ in tiles}), 16)

    def test_cores_partition_the_raster_exactly_once(self):
        width = height = 2600
        tile, overlap = 1024, 128
        xs = seg.axis_starts(width, tile, overlap)
        ys = seg.axis_starts(height, tile, overlap)
        x_edges = seg.axis_boundaries(xs, tile, width)
        y_edges = seg.axis_boundaries(ys, tile, height)

        covered = np.zeros((height, width), dtype=np.int16)
        for row in range(len(ys)):
            for column in range(len(xs)):
                x0, x1 = int(x_edges[column]), int(x_edges[column + 1])
                y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
                covered[y0:y1, x0:x1] += 1

        self.assertTrue(covered.all())
        self.assertEqual(int(covered.max()), 1)


class SegmentRasterTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.path = "data/scene.tif"
        self.write_scene(SIZE, SIZE)

    def tearDown(self):
        self._tmp.cleanup()

    def write_scene(self, width: int, height: int) -> None:
        """Write a textured RGB raster: a flat image gives the model nothing."""
        rng = np.random.default_rng(4)
        rgb = rng.integers(40, 220, size=(height, width, 3), dtype=np.uint8)
        target = self.workspace.resolve(self.path, write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            target, "w", driver="GTiff", width=width, height=height, count=3,
            dtype="uint8", crs="EPSG:32645", transform=from_origin(500_000, 3_000_000, 1, 1),
        ) as dataset:
            dataset.write(np.transpose(rgb, (2, 0, 1)))

    def test_auto_mode_returns_polygons_in_the_raster_crs(self):
        session = make_session(FakeDecoder())

        frame, timings = seg.segment_raster(
            self.workspace, self.path, session, mode="auto", points_per_side=4, min_area_px=0
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.crs.to_string(), "EPSG:32645")
        # Mask rows 80..119 and cols 100..139 of the 256 grid map onto tile
        # pixels 160..239 and 200..279 (scale 2), which is 500 200..500 280 east
        # and 2 999 760..2 999 840 north on a north-up 1 m grid.
        minx, miny, maxx, maxy = frame.geometry.iloc[0].bounds
        self.assertAlmostEqual(minx, 500_200)
        self.assertAlmostEqual(maxx, 500_280)
        self.assertAlmostEqual(miny, 2_999_760)
        self.assertAlmostEqual(maxy, 2_999_840)
        self.assertEqual(timings.tiles, 1)
        self.assertEqual(frame["score"].iloc[0], 0.95)

    def test_grid_points_are_pruned_once_a_mask_covers_them(self):
        session = make_session(FakeDecoder(whole_tile=True))
        points, labels = seg.grid_points(1024, 1024, 1024, 16)
        embeddings = (np.zeros((1, 256, 64, 64), np.float32), np.zeros((1, 256, 64, 64), np.float32))

        masks, decoded = seg.auto_masks(session, embeddings, points, labels)

        # The first batch of 64 covers everything, so the remaining 192 grid
        # points are never decoded — and never returned twice.
        self.assertEqual(len(masks), 1)
        self.assertEqual(decoded, seg.DECODE_BATCH)
        self.assertLess(decoded, len(points))

    def test_a_mask_covering_the_whole_tile_is_dropped(self):
        session = make_session(FakeDecoder(whole_tile=True))

        frame, _timings = seg.segment_raster(
            self.workspace, self.path, session, mode="auto", points_per_side=4, min_area_px=0
        )

        self.assertEqual(len(frame), 0)

    def test_small_masks_are_dropped_by_the_area_floor(self):
        session = make_session(FakeDecoder(box=(80, 90, 100, 110)))

        frame, _timings = seg.segment_raster(
            self.workspace, self.path, session, mode="auto", points_per_side=4, min_area_px=10_000
        )

        self.assertEqual(len(frame), 0)

    def test_weak_masks_are_dropped_by_the_iou_threshold(self):
        session = make_session(FakeDecoder(score=0.4))

        frame, _timings = seg.segment_raster(
            self.workspace, self.path, session, mode="auto", points_per_side=4,
            iou_threshold=0.85, min_area_px=0,
        )

        self.assertEqual(len(frame), 0)

    def test_prompt_mode_decodes_only_the_tile_the_prompt_falls_in(self):
        self.write_scene(2600, 1024)
        session = make_session(FakeDecoder())

        frame, timings = seg.segment_raster(
            self.workspace, self.path, session, mode="points",
            points=[[500_100, 2_999_900]], prompt_crs="EPSG:32645", min_area_px=0,
        )

        self.assertEqual(timings.tiles, 3)  # every tile is read...
        self.assertEqual(timings.prompts, 1)  # ...but only one is decoded
        self.assertGreaterEqual(len(frame), 1)

    def test_prompt_mode_without_prompts_is_refused(self):
        session = make_session(FakeDecoder())

        with self.assertRaises(ToolInputError):
            seg.segment_raster(self.workspace, self.path, session, mode="points")

    def test_unknown_mode_is_refused(self):
        session = make_session(FakeDecoder())

        with self.assertRaises(ToolInputError):
            seg.segment_raster(self.workspace, self.path, session, mode="everything")

    def test_too_many_tiles_is_refused_with_a_direction(self):
        self.write_scene(4000, 4000)
        session = make_session(FakeDecoder())

        with self.assertRaises(ToolInputError) as caught:
            seg.segment_raster(
                self.workspace, self.path, session, mode="auto", tile_size=1024, max_tiles=2
            )

        self.assertIn("bounds", str(caught.exception))

    def test_embeddings_are_reused_for_a_second_run(self):
        session = make_session(FakeDecoder())

        seg.segment_raster(self.workspace, self.path, session, mode="auto", points_per_side=2)
        cached = dict(session.embeddings)
        seg.segment_raster(self.workspace, self.path, session, mode="auto", points_per_side=2)

        self.assertTrue(cached)
        self.assertLessEqual(len(session.embeddings), EMBEDDING_CACHE_SIZE)


if __name__ == "__main__":
    unittest.main()
