"""The AI packs: three packs over three problems, and the shared publish step."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from spatial_intelligence.contracts.progress import NULL_REPORTER
from spatial_intelligence.tools.build import default_registry
from spatial_intelligence.tools.packs.ai_common import publish, task_job
from spatial_intelligence.tools.packs.detection import DetectionPack
from spatial_intelligence.tools.packs.segmentation import SegmentationPack
from spatial_intelligence.tools.runtime import RuntimeEvents, ToolRuntime
from spatial_intelligence.workspace import Workspace


class PackSplitTestCase(unittest.TestCase):
    """Each problem keeps its own pack, and the lifecycle stays separate."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.runtime = ToolRuntime(
            workspace=self.workspace, reporter=NULL_REPORTER, events=RuntimeEvents()
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_three_packs_hold_their_own_tools(self):
        lifecycle = {spec.name for spec in default_registry(self.runtime).specs()}
        registry = default_registry(self.runtime)

        self.assertEqual(
            [spec.name for spec in registry.by_category("ai")],
            ["ai_models", "ai_pull_model", "ai_fetch_model", "ai_unload_model"],
        )
        self.assertEqual(
            [spec.name for spec in registry.by_category("segmentation")], ["segment_image"]
        )
        self.assertEqual(
            [spec.name for spec in registry.by_category("detection")], ["detect_objects"]
        )
        self.assertTrue({"ai_models", "segment_image", "detect_objects"} <= lifecycle)

    def test_each_task_pack_declares_its_own_effects_and_kind(self):
        registry = default_registry(self.runtime)

        for name in ("segment_image", "detect_objects"):
            spec = registry.get(name)
            self.assertEqual(spec.kind.value, "reporting")
            self.assertEqual(spec.timeout, 1800)
            self.assertIn("workspace_write", spec.effects)
            self.assertFalse(spec.replay_safe)  # it writes a layer

    def test_each_task_tool_defaults_to_its_own_model(self):
        # The first default of each tool is its model_id: the tools do not share
        # a default, because they do not share a problem.
        self.assertEqual(
            SegmentationPack.__dict__["segment_image"].__defaults__[0], "slimsam-77"
        )
        self.assertEqual(
            DetectionPack.__dict__["detect_objects"].__defaults__[0], "yolos-tiny"
        )


class PublishTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.runtime = ToolRuntime(
            workspace=self.workspace, reporter=NULL_REPORTER, events=RuntimeEvents()
        )
        self.target = self.workspace.resolve("data/scene.tif", write=True)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            self.target, "w", driver="GTiff", width=8, height=8, count=3, dtype="uint8",
            crs="EPSG:32645", transform=from_origin(500_000, 3_000_000, 1, 1),
        ) as dataset:
            dataset.write(np.zeros((3, 8, 8), dtype="uint8"))

    def tearDown(self):
        self._tmp.cleanup()

    def frame(self) -> gpd.GeoDataFrame:
        geometry = box(500_000, 2_999_992, 500_004, 2_999_996)
        return gpd.GeoDataFrame(
            [{"geometry": geometry, "score": 0.9, "label": "car"}],
            geometry="geometry",
            crs="EPSG:32645",
        )

    def test_publish_writes_wgs84_and_records_the_artifact(self):
        summary = publish(self.runtime, self.frame(), out="results/x.geojson", keep_crs=False)

        written = Path(summary["path"])
        self.assertTrue(written.is_file())
        self.assertEqual(summary["crs"], "EPSG:4326")
        self.assertEqual(summary["source_crs"], "EPSG:32645")
        self.assertEqual(summary["score"], {"min": 0.9, "max": 0.9})
        self.assertEqual(len(summary["bounds"]), 4)
        manifest = json.loads(self.workspace.manifest_path.read_text(encoding="utf-8"))
        self.assertIn("results/x.geojson", manifest["outputs"])

    def test_keep_crs_leaves_the_layer_in_the_raster_crs(self):
        summary = publish(self.runtime, self.frame(), out="results/y.geojson", keep_crs=True)

        self.assertEqual(summary["crs"], "EPSG:32645")
        self.assertEqual(summary["source_crs"], "EPSG:32645")


class TaskJobTestCase(unittest.TestCase):
    """The shared job/reservation helper: the part both task packs rely on."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.runtime = ToolRuntime(
            workspace=self.workspace, reporter=NULL_REPORTER, events=RuntimeEvents()
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_helper_hands_back_a_job_and_a_loaded_session(self):
        session = object()
        seen: list[str] = []

        class FakeManager:
            @contextmanager
            def reserve(self, model_id, *, progress=None):
                seen.append(model_id)
                yield session

        with task_job(self.runtime, FakeManager(), "slimsam-77", "segment x.geojson") as (
            job,
            handed,
        ):
            self.assertIs(handed, session)
            self.assertTrue(job.id)
            self.assertEqual(job.state.value, "running")

        self.assertEqual(seen, ["slimsam-77"])

    def test_the_reservation_is_released_when_the_block_exits(self):
        released: list[bool] = []

        class FakeManager:
            @contextmanager
            def reserve(self, model_id, *, progress=None):
                try:
                    yield object()
                finally:
                    released.append(True)

        with task_job(self.runtime, FakeManager(), "slimsam-77", "segment x.geojson"):
            self.assertEqual(released, [])

        self.assertEqual(released, [True])


class WrongModelTestCase(unittest.TestCase):
    """A task tool asked for the wrong model fails with a direction, not a stack."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.runtime = ToolRuntime(
            workspace=self.workspace, reporter=NULL_REPORTER, events=RuntimeEvents()
        )
        self.target = self.workspace.resolve("data/scene.tif", write=True)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            self.target, "w", driver="GTiff", width=8, height=8, count=3, dtype="uint8",
            crs="EPSG:32645", transform=from_origin(500_000, 3_000_000, 1, 1),
        ) as dataset:
            dataset.write(np.zeros((3, 8, 8), dtype="uint8"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_segmenter_refuses_a_detector(self):
        pack = SegmentationPack(self.runtime)

        with self.assertRaises(Exception) as caught:
            pack.segment_image("data/scene.tif", model_id="yolos-tiny")

        self.assertIn("not a segmenter", str(caught.exception))

    def test_the_detector_refuses_a_segmenter(self):
        pack = DetectionPack(self.runtime)

        with self.assertRaises(Exception) as caught:
            pack.detect_objects("data/scene.tif", model_id="slimsam-77")

        self.assertIn("no classes", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
