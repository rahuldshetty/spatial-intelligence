"""Detection: the canvas mapping, the score/class filters, and box ownership."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from spatial_intelligence.ai import catalog
from spatial_intelligence.ai import detection as det
from spatial_intelligence.ai.manager import ModelSession
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.workspace import Workspace

LABELS = ("N/A", "person", "car", "boat")

#: A 512x512 RGB scene at one metre per pixel.
SIZE = 512


class FakeDetector:
    """A detector that answers each query with one configured box.

    Logits are built so the softmax puts ``score`` on the class: the class slot
    gets ``log(score)`` and the rest share ``log(1 - score)``. The graph's last
    slot is the reserved "no object" one, as in the real export.
    """

    def __init__(self, detections=((2, 0.9, 0.25, 0.25, 0.5, 0.5),), classes=92, queries=100):
        self.detections = detections
        self.classes = classes
        self.queries = queries
        self.calls = 0
        self.last_shape = None

    def run(self, _outputs, feeds):
        self.calls += 1
        self.last_shape = feeds["pixel_values"].shape
        logits = np.zeros((1, self.queries, self.classes), dtype=np.float32)
        boxes = np.zeros((1, self.queries, 4), dtype=np.float32)
        for query, (class_id, score, cx, cy, box_w, box_h) in enumerate(self.detections):
            logits[0, query] = np.log((1.0 - score) / (self.classes - 1))
            logits[0, query, class_id] = np.log(score)
            boxes[0, query] = (cx, cy, box_w, box_h)
        return logits, boxes


def make_session(decoder, labels=LABELS, shortest_edge=512, longest_edge=1333) -> ModelSession:
    """A session over the real catalog entry shape, with a fake graph."""
    spec = catalog.YOLOS_TINY if labels is catalog.COCO_LABELS else catalog.ModelSpec(
        id="fake-detector",
        repo="example/fake",
        revision="rev",
        files=(),
        task="detection",
        prompts=(),
        labels=labels,
        shortest_edge=shortest_edge,
        longest_edge=longest_edge,
    )
    return ModelSession(
        spec=spec, encoder=decoder, decoder=decoder, provider="CPUExecutionProvider", threads=1
    )


class CanvasTestCase(unittest.TestCase):
    def test_the_short_edge_sets_the_scale_when_the_long_edge_fits(self):
        canvas = det.canvas_for(1000, 2000, shortest_edge=512, longest_edge=1333, max_side=1333)

        self.assertEqual(canvas.height, 512)
        self.assertEqual(canvas.width, 1024)
        self.assertAlmostEqual(canvas.scale, 0.512)

    def test_the_long_edge_caps_the_scale(self):
        canvas = det.canvas_for(500, 5000, shortest_edge=512, longest_edge=1333, max_side=1333)

        self.assertEqual(canvas.width, 1328)  # 1333 rounded down to a multiple of 16
        self.assertEqual(canvas.height, 128)
        self.assertLess(canvas.scale, 512 / 500)

    def test_max_side_lowers_the_short_edge_too(self):
        reference = det.canvas_for(1000, 2000, 512, 1333, 1333)
        capped = det.canvas_for(1000, 2000, 512, 1333, 800)

        self.assertLess(capped.width, reference.width)
        self.assertLessEqual(capped.width, 800)
        self.assertAlmostEqual(capped.width / capped.height, reference.width / reference.height, places=1)

    def test_the_canvas_is_a_multiple_of_the_patch(self):
        for height, width in ((333, 777), (1000, 40), (16, 16), (1023, 2047)):
            canvas = det.canvas_for(height, width, 512, 1333, 800)
            self.assertEqual(canvas.width % det.PATCH, 0)
            self.assertEqual(canvas.height % det.PATCH, 0)
            self.assertGreaterEqual(canvas.width, det.PATCH)


def softmax_logits(pairs, classes: int) -> np.ndarray:
    """Build ``(1, queries, classes)`` logits whose softmax puts each score on its class."""
    logits = np.zeros((1, len(pairs), classes), dtype=np.float32)
    for query, (class_id, score) in enumerate(pairs):
        logits[0, query] = np.log((1.0 - score) / (classes - 1))
        logits[0, query, class_id] = np.log(score)
    return logits


class DecodeTestCase(unittest.TestCase):
    def test_a_detection_is_read_with_its_class_and_score(self):
        logits = softmax_logits([(2, 0.9)], classes=4)
        boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)

        found = det.decode((logits, boxes), confidence=0.5, labels=LABELS)

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].label, "car")
        self.assertEqual(found[0].class_id, 2)
        self.assertAlmostEqual(found[0].score, 0.9, places=4)
        for actual, expected in zip(found[0].box, (0.4, 0.4, 0.6, 0.6)):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_the_reserved_slot_is_not_a_class(self):
        labels = ("N/A", "person", "car")
        logits = softmax_logits([(0, 0.9), (1, 0.8)], classes=3)
        boxes = np.zeros((1, 2, 4), dtype=np.float32)

        found = det.decode((logits, boxes), confidence=0.5, labels=labels)

        self.assertEqual([item.label for item in found], ["person"])

    def test_a_query_that_answers_nothing_is_dropped(self):
        # A uniform query spends a slice of its probability everywhere, which is
        # what "no object" looks like once the reserved slot is removed.
        logits = np.zeros((1, 1, 92), dtype=np.float32)
        boxes = np.zeros((1, 1, 4), dtype=np.float32)

        self.assertEqual(det.decode((logits, boxes), confidence=0.5, labels=LABELS), [])

    def test_the_confidence_floor_drops_weak_queries(self):
        logits = softmax_logits([(2, 0.27)], classes=4)
        boxes = np.zeros((1, 1, 4), dtype=np.float32)

        self.assertEqual(len(det.decode((logits, boxes), confidence=0.25, labels=LABELS)), 1)
        self.assertEqual(len(det.decode((logits, boxes), confidence=0.5, labels=LABELS)), 0)

    def test_the_class_filter_keeps_only_what_was_asked_for(self):
        logits = softmax_logits([(2, 0.9), (3, 0.8)], classes=4)
        boxes = np.zeros((1, 2, 4), dtype=np.float32)

        found = det.decode((logits, boxes), confidence=0.5, labels=LABELS, classes=[" boat "])

        self.assertEqual([item.label for item in found], ["boat"])

    def test_a_mismatched_graph_is_reported(self):
        with self.assertRaises(ToolInputError):
            det.decode((np.zeros((1, 4), dtype=np.float32),), confidence=0.25, labels=LABELS)
        with self.assertRaises(ToolInputError):
            det.decode(
                (np.zeros((1, 2, 2), np.float32), np.zeros((1, 2, 4), np.float32)),
                confidence=0.25,
                labels=LABELS,
            )


class MappingTestCase(unittest.TestCase):
    def test_normalized_boxes_become_tile_pixels(self):
        # A canvas with scale exactly 1, so the mapping has one correct answer.
        canvas = det.canvas_for(512, 512, 512, 512, 512)
        self.assertEqual((canvas.width, canvas.height, canvas.scale), (512, 512, 1.0))
        found = [
            det.Detection("car", 2, 0.9, (0.25, 0.25, 0.75, 0.75)),
        ]

        mapped = det.to_tile_pixels(found, canvas, 512, 512)

        self.assertEqual(mapped[0].box, (128.0, 128.0, 384.0, 384.0))

    def test_boxes_are_clamped_to_the_tile_and_thin_ones_dropped(self):
        canvas = det.canvas_for(512, 512, 512, 512, 512)
        found = [
            det.Detection("car", 2, 0.9, (-0.5, -0.5, 1.5, 1.5)),
            det.Detection("boat", 3, 0.8, (0.5, 0.5, 0.5005, 0.9)),  # under a pixel wide
        ]

        mapped = det.to_tile_pixels(found, canvas, 512, 512)

        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].box, (0.0, 0.0, 512.0, 512.0))

    def test_nms_suppresses_within_a_class_only(self):
        overlapping_car = det.Detection("car", 2, 0.9, (0, 0, 10, 10))
        weaker_car = det.Detection("car", 2, 0.7, (1, 1, 11, 11))
        boat_same_place = det.Detection("boat", 3, 0.6, (1, 1, 11, 11))

        kept = det.box_nms([weaker_car, boat_same_place, overlapping_car])

        self.assertEqual([item.label for item in kept], ["car", "boat"])


class PerTileDetector:
    """A detector whose answer depends on which tile it is asked about.

    Used to stage the cross-tile duplicate: two tiles see one object at slightly
    different places, each with its centre inside its own tile's core, so only a
    suppression pass over the assembled result can collapse them.
    """

    def __init__(self, by_call: dict[int, tuple[int, float, float, float, float]]):
        self.by_call = by_call
        self.calls = 0

    def run(self, _outputs, feeds):
        self.calls += 1
        logits = np.zeros((1, 2, 92), dtype=np.float32)
        boxes = np.zeros((1, 2, 4), dtype=np.float32)
        answer = self.by_call.get(self.calls)
        if answer is not None:
            class_id, score, cx, cy, size = answer
            logits[0, 0] = np.log((1 - score) / 91)
            logits[0, 0, class_id] = np.log(score)
            boxes[0, 0] = (cx, cy, size, size)
        return logits, boxes


class CrossTileTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        rng = np.random.default_rng(11)
        rgb = rng.integers(40, 220, size=(SIZE, SIZE, 3), dtype=np.uint8)
        target = self.workspace.resolve("data/scene.tif", write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            target, "w", driver="GTiff", width=SIZE, height=SIZE, count=3, dtype="uint8",
            crs="EPSG:32645", transform=from_origin(500_000, 3_000_000, 1, 1),
        ) as dataset:
            dataset.write(np.transpose(rgb, (2, 0, 1)))
        self.path = "data/scene.tif"

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_object_seen_by_two_tiles_is_reported_once(self):
        # Tiles of 256 with 128 overlap: starts 0/128/256, cores split at 192 and
        # 320. The first tile reports a box centred at raster (150, 150), the
        # tile below it one centred at (150, 180) — both inside the first core,
        # overlapping by more than the threshold.
        graph = PerTileDetector(
            {
                1: (2, 0.95, 150 / 256, 150 / 256, 0.5),
                4: (2, 0.90, 150 / 256, 52 / 256, 0.5),
            }
        )
        session = make_session(graph, shortest_edge=256, longest_edge=256)

        frame, _timings = det.detect_raster(
            self.workspace, self.path, session, confidence=0.5, max_side=256,
            tile_size=256, overlap=128,
        )

        self.assertEqual(len(frame), 1, frame[["label", "score"]].to_string())
        self.assertEqual(frame["score"].iloc[0], 0.95)  # the stronger box survives

    def test_nms_indices_keeps_the_best_and_respects_class(self):
        kept = det.nms_indices(
            [2, 2, 3, 2],
            [(0, 0, 10, 10), (0.5, 0.5, 10.5, 10.5), (0.5, 0.5, 10.5, 10.5), (50, 50, 60, 60)],
            [0.9, 0.8, 0.7, 0.6],
        )

        self.assertEqual(kept, [0, 2, 3])
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.path = "data/scene.tif"
        rng = np.random.default_rng(7)
        rgb = rng.integers(40, 220, size=(SIZE, SIZE, 3), dtype=np.uint8)
        target = self.workspace.resolve(self.path, write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            target, "w", driver="GTiff", width=SIZE, height=SIZE, count=3, dtype="uint8",
            crs="EPSG:32645", transform=from_origin(500_000, 3_000_000, 1, 1),
        ) as dataset:
            dataset.write(np.transpose(rgb, (2, 0, 1)))

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_detection_lands_on_the_ground_it_boxed(self):
        # Centre of the canvas, a tenth of the canvas wide. The spec resizes
        # 512 to 512, so a normalized box maps to the same tile pixels.
        graph = FakeDetector(detections=[(2, 0.9, 0.5, 0.5, 0.1, 0.1)])
        session = make_session(graph, shortest_edge=512, longest_edge=512)

        frame, timings = det.detect_raster(
            self.workspace, self.path, session, confidence=0.25, max_side=512
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["label"].iloc[0], "car")
        self.assertEqual(frame.crs.to_string(), "EPSG:32645")
        minx, miny, maxx, maxy = frame.geometry.iloc[0].bounds
        self.assertAlmostEqual((minx + maxx) / 2, 500_256, delta=0.5)
        self.assertAlmostEqual((miny + maxy) / 2, 2_999_744, delta=0.5)
        self.assertAlmostEqual(maxx - minx, 51.2, delta=0.5)
        self.assertEqual(timings.tiles, 1)
        self.assertEqual(timings.objects, 1)

    def test_the_class_filter_reaches_the_graph_output(self):
        graph = FakeDetector(
            detections=[(2, 0.9, 0.4, 0.4, 0.1, 0.1), (3, 0.8, 0.6, 0.6, 0.1, 0.1)]
        )
        session = make_session(graph)

        frame, _timings = det.detect_raster(
            self.workspace, self.path, session, classes=["boat"], max_side=512
        )

        self.assertEqual(frame["label"].tolist(), ["boat"])

    def test_a_detection_is_owned_by_the_core_that_holds_its_centre(self):
        core = (100.0, 200.0, 300.0, 400.0)

        self.assertTrue(det.owned_by_core(250.0, 300.0, core))
        self.assertTrue(det.owned_by_core(100.0, 200.0, core))  # the corner is inclusive
        self.assertFalse(det.owned_by_core(300.0, 300.0, core))  # the far edge belongs next door
        self.assertFalse(det.owned_by_core(99.0, 300.0, core))
        self.assertFalse(det.owned_by_core(250.0, 400.0, core))

    def test_overlapping_tiles_do_not_multiply_detections(self):
        # Every tile answers with a box at its own centre, so only the tiles whose
        # centre lies in their own core survive the ownership test.
        graph = FakeDetector(detections=[(2, 0.9, 0.5, 0.5, 0.02, 0.02)])
        session = make_session(graph, shortest_edge=256, longest_edge=256)

        frame, timings = det.detect_raster(
            self.workspace, self.path, session, tile_size=256, overlap=64, max_side=256
        )

        self.assertGreater(timings.tiles, 1)
        self.assertGreater(len(frame), 0)
        self.assertLessEqual(len(frame), timings.tiles)

    def test_a_model_without_labels_is_refused(self):
        session = make_session(FakeDetector(), labels=())

        with self.assertRaises(ToolInputError) as caught:
            det.detect_raster(self.workspace, self.path, session)

        self.assertIn("labels", str(caught.exception))

    def test_too_many_tiles_is_refused(self):
        graph = FakeDetector()
        session = make_session(graph)

        with self.assertRaises(ToolInputError):
            det.detect_raster(
                self.workspace, self.path, session, tile_size=256, overlap=0, max_tiles=2
            )


class CatalogTestCase(unittest.TestCase):
    def test_the_detection_entry_is_pinned_and_labelled(self):
        spec = catalog.find("yolos-tiny")

        self.assertEqual(spec.task, "detection")
        self.assertEqual(len(spec.labels), 91)
        self.assertEqual(spec.labels[1], "person")
        self.assertEqual(spec.labels[9], "boat")
        self.assertEqual(len(spec.files), 1)
        self.assertNotEqual(spec.revision, "main")  # a revision must be immutable
        self.assertEqual(len(spec.files[0].sha256), 64)
        self.assertGreater(spec.shortest_edge, 0)
        self.assertGreater(spec.longest_edge, spec.shortest_edge)

    def test_the_two_catalog_entries_serve_different_tasks(self):
        tasks = {spec.task for spec in catalog.MODELS}

        self.assertEqual(tasks, {"segmentation", "detection"})
        self.assertEqual([spec.id for spec in catalog.iter_specs("detection")], ["yolos-tiny"])


if __name__ == "__main__":
    unittest.main()
