"""User-drawn AOI retrieval: the browser bridge and saved GeoJSON summary."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest

from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.contracts.services import MAP_BRIDGE_SERVICE
from spatial_intelligence.session.bus import EventBus
from spatial_intelligence.session.map_bridge import MapBridgeBroker
from spatial_intelligence.tools.packs.layers import LayersPack
from spatial_intelligence.tools.runtime import ToolRuntime
from spatial_intelligence.workspace import Workspace

POINT = {
    "type": "Feature",
    "id": "user-point",
    "properties": {"label": "camp"},
    "geometry": {"type": "Point", "coordinates": [85.1, 27.8]},
}
POLYGON = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[85.0, 27.7], [85.2, 27.7], [85.2, 27.9], [85.0, 27.7]]],
    },
}


class FakeBridge:
    def __init__(self, features):
        self.features = features
        self.method = None

    def request(self, method, params=None, *, workspace=None):
        self.method = method
        return self.features


class MapBridgeBrokerTests(unittest.TestCase):
    def test_a_query_round_trips_through_the_browser(self):
        bus = EventBus()
        subscriber = bus.subscribe()
        broker = MapBridgeBroker(bus, timeout=2)
        outcome: dict = {}

        def worker():
            outcome["value"] = broker.request(
                "getDrawnFeatures", {}, workspace="demo"
            )

        thread = threading.Thread(target=worker)
        thread.start()
        event = subscriber.get(timeout=1)
        self.assertEqual(event["event"], "map_bridge")
        data = event["data"]
        self.assertEqual(data["method"], "getDrawnFeatures")
        self.assertEqual(data["workspace"], "demo")
        self.assertTrue(
            broker.complete(
                data["id"],
                data["token"],
                result=[POINT],
            )
        )
        thread.join(timeout=2)
        bus.unsubscribe(subscriber)
        self.assertEqual(outcome["value"], [POINT])

    def test_a_bad_token_is_ignored(self):
        bus = EventBus()
        subscriber = bus.subscribe()
        broker = MapBridgeBroker(bus, timeout=2)
        outcome: dict = {}

        def worker():
            try:
                outcome["value"] = broker.request("getDrawnFeatures")
            except ToolInputError as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        data = subscriber.get(timeout=1)["data"]
        self.assertFalse(
            broker.complete(data["id"], "wrong-token", result=[POINT])
        )
        self.assertTrue(broker.complete(data["id"], data["token"], result=[POINT]))
        thread.join(timeout=2)
        bus.unsubscribe(subscriber)
        self.assertEqual(outcome["value"], [POINT])

    def test_no_sse_client_fails_fast(self):
        with self.assertRaises(ToolInputError):
            MapBridgeBroker(EventBus()).request("getDrawnFeatures")


class DrawnFeatureToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(self.temp.name).create()

    def tearDown(self):
        self.temp.cleanup()

    def runtime(self, features) -> tuple[ToolRuntime, FakeBridge]:
        bridge = FakeBridge(features)
        runtime = ToolRuntime(
            workspace=self.workspace,
            services={MAP_BRIDGE_SERVICE: bridge},
        )
        return runtime, bridge

    def test_features_are_summarized_and_saved(self):
        runtime, bridge = self.runtime([POINT, POLYGON])
        result = LayersPack(runtime).get_drawn_features()

        self.assertEqual(bridge.method, "getDrawnFeatures")
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["geometry_types"], {"Point": 1, "Polygon": 1})
        self.assertEqual(result["bbox"], [85.0, 27.7, 85.2, 27.9])
        self.assertEqual(result["features"][0]["id"], "user-point")
        self.assertEqual(result["features"][0]["bbox"], [85.1, 27.8, 85.1, 27.8])
        self.assertEqual(result["features"][0]["center"], [85.1, 27.8])
        self.assertEqual(result["path"], "results/drawn_aoi.geojson")

        saved = json.loads(
            (self.workspace.root / "results" / "drawn_aoi.geojson").read_text("utf-8")
        )
        self.assertEqual(saved["type"], "FeatureCollection")
        self.assertEqual(len(saved["features"]), 2)
        manifest = json.loads(self.workspace.manifest_path.read_text("utf-8"))
        self.assertIn("results/drawn_aoi.geojson", manifest["outputs"])

    def test_an_empty_sketch_layer_saves_nothing(self):
        runtime, _ = self.runtime([])
        result = LayersPack(runtime).get_drawn_features()

        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["path"])
        self.assertIn("No shapes", result["note"])
        self.assertFalse((self.workspace.root / "results" / "drawn_aoi.geojson").exists())

    def test_path_empty_skips_the_file(self):
        runtime, _ = self.runtime([POINT])
        result = LayersPack(runtime).get_drawn_features(path="")

        self.assertEqual(result["count"], 1)
        self.assertIsNone(result["path"])
        self.assertFalse((self.workspace.root / "results" / "drawn_aoi.geojson").exists())

    def test_invalid_geometry_is_reported(self):
        broken = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1]]]},
        }
        runtime, _ = self.runtime([broken])
        with self.assertRaises(ToolInputError):
            LayersPack(runtime).get_drawn_features()


if __name__ == "__main__":
    unittest.main()
