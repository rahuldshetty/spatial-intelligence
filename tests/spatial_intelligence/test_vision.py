"""Image-input probing, map capture, and the conditional image tool."""

from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic_ai import BinaryContent, ToolReturn

from spatial_intelligence.agent.vision import VisionService, probe_now
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.contracts.services import CAPTURE_SERVICE, VISION_SERVICE
from spatial_intelligence.session.bus import EventBus
from spatial_intelligence.session.capture import CaptureBroker, MapCapture
from spatial_intelligence.session.jobs import JobRegistry
from spatial_intelligence.session.services import SessionServices
from spatial_intelligence.tools.build import default_registry
from spatial_intelligence.tools.packs.vision import VisionPack
from spatial_intelligence.tools.runtime import ToolRuntime
from spatial_intelligence.workspace import Workspace


class FakeResponse:
    def __init__(self, status: int, text: str = "", payload: dict | None = None) -> None:
        self.status_code = status
        self.text = text
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def png_bytes(width: int = 12, height: int = 10) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (180, 40, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


class VisionProbeTests(unittest.TestCase):
    def _probe(self, response: FakeResponse, model: str = "openai-chat:vision-test") -> dict:
        with patch("spatial_intelligence.agent.vision.httpx.post", return_value=response):
            with patch.dict(
                os.environ,
                {
                    "OPENAI_BASE_URL": "http://vision.test/v1",
                    "OPENAI_API_KEY": "test-key",
                },
            ):
                return probe_now(model)

    def test_a_two_hundred_means_image_input_is_supported(self):
        result = self._probe(
            FakeResponse(200, "{}", {"choices": [{"message": {"content": "OK"}}]})
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["state"], "supported")

    def test_an_image_rejection_means_unsupported(self):
        result = self._probe(
            FakeResponse(400, '{"error":{"message":"This model does not support image input"}}')
        )
        self.assertFalse(result["available"])
        self.assertEqual(result["state"], "unsupported")

    def test_a_credential_failure_stays_an_error_not_unsupported(self):
        result = self._probe(FakeResponse(401, '{"error":{"message":"bad key"}}'))
        self.assertFalse(result["available"])
        self.assertEqual(result["state"], "error")
        self.assertIn("credentials", result["detail"])

    def test_an_unrelated_bad_request_stays_an_error(self):
        result = self._probe(FakeResponse(400, '{"error":{"message":"unknown model"}}'))
        self.assertEqual(result["state"], "error")

    def test_non_openai_providers_are_not_probed(self):
        with patch("spatial_intelligence.agent.vision.httpx.post") as post:
            result = probe_now("anthropic:claude-sonnet-4-5")
        self.assertEqual(result["state"], "unsupported")
        post.assert_not_called()


class CaptureBrokerTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.subscriber = self.bus.subscribe()
        self.broker = CaptureBroker(self.bus, timeout=2)

    def tearDown(self):
        self.bus.unsubscribe(self.subscriber)

    def test_a_request_round_trips_through_the_browser(self):
        outcome: dict = {}

        def worker():
            outcome["capture"] = self.broker.request(workspace="demo")

        thread = threading.Thread(target=worker)
        thread.start()
        event = self.subscriber.get(timeout=1)
        self.assertEqual(event["event"], "map_capture")
        data = event["data"]
        self.assertEqual(data["workspace"], "demo")
        encoded = base64.b64encode(png_bytes()).decode()
        accepted = self.broker.complete(
            data["id"],
            data["token"],
            image="data:image/png;base64," + encoded,
            view={"bbox": [1, 2, 3, 4], "zoom": 12.0, "pitch": 0, "bearing": 0},
        )
        thread.join(timeout=2)
        self.assertTrue(accepted)
        capture = outcome["capture"]
        self.assertIsInstance(capture, MapCapture)
        self.assertEqual((capture.width, capture.height), (12, 10))
        self.assertEqual(capture.view["bbox"], [1, 2, 3, 4])

    def test_a_bad_token_does_not_resolve_the_request(self):
        outcome: dict = {}

        def worker():
            try:
                outcome["capture"] = self.broker.request(workspace="demo")
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        event = self.subscriber.get(timeout=1)["data"]
        self.assertFalse(
            self.broker.complete(
                event["id"],
                "wrong-token",
                image="data:image/png;base64," + base64.b64encode(png_bytes()).decode(),
                view={},
            )
        )
        self.broker.complete(
            event["id"],
            event["token"],
            image="data:image/png;base64," + base64.b64encode(png_bytes()).decode(),
            view={},
        )
        thread.join(timeout=2)
        self.assertIn("capture", outcome)

    def test_no_sse_client_fails_fast(self):
        bus = EventBus()
        broker = CaptureBroker(bus, timeout=2)
        with self.assertRaises(ToolInputError):
            broker.request(workspace="demo")


class FakeVision:
    def __init__(self, supported: bool = True) -> None:
        self.supported = supported


class FakeBroker:
    def __init__(self) -> None:
        self.capture = MapCapture(
            image=png_bytes(20, 16),
            media_type="image/png",
            width=20,
            height=16,
            view={"bbox": [85.0, 27.0, 85.1, 27.1], "zoom": 11.0, "pitch": 0, "bearing": 0},
        )

    def request(self, **_kwargs):
        return self.capture


class VisionGatingTests(unittest.TestCase):
    def test_the_tool_appears_when_a_background_probe_succeeds(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root).create()
            bus = EventBus()
            jobs = JobRegistry(bus)
            holder: dict = {}
            vision = VisionService(
                "openai:gpt-4o",
                on_change=lambda: holder["sync"](),
                enabled=True,
            )
            services = SessionServices(
                bus=bus,
                jobs=jobs,
                settings=lambda: {"model": "openai:gpt-4o"},
                files_provider=lambda: [],
                map_provider=lambda: {},
                vision=vision,
                capture=CaptureBroker(bus),
            )
            holder["sync"] = services.sync_vision
            services.bind(workspace, None)
            try:
                self.assertNotIn("analyze_image", services.registry)
                with patch(
                    "spatial_intelligence.agent.vision._probe",
                    return_value=("supported", ""),
                ):
                    vision.start()
                    vision._thread.join(timeout=2)
                self.assertIn("analyze_image", services.registry)
            finally:
                services.unbind()


class VisionToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(self.temp.name)
        self.workspace.data.mkdir(parents=True, exist_ok=True)
        self.workspace.results.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp.cleanup()

    def runtime(self, supported: bool = True) -> ToolRuntime:
        return ToolRuntime(
            workspace=self.workspace,
            services={
                VISION_SERVICE: FakeVision(supported),
                CAPTURE_SERVICE: FakeBroker(),
            },
        )

    def test_map_source_returns_a_jpeg_and_view_metadata(self):
        result = VisionPack(self.runtime()).analyze_image("what is shown?")
        self.assertIsInstance(result, ToolReturn)
        self.assertIn("bbox", result.return_value)
        self.assertEqual(len(result.content), 1)
        self.assertIsInstance(result.content[0], BinaryContent)
        self.assertEqual(result.content[0].media_type, "image/jpeg")

    def test_a_workspace_image_is_read_and_resized(self):
        path = self.workspace.data / "shot.png"
        path.write_bytes(png_bytes(1200, 600))
        result = VisionPack(self.runtime()).analyze_image(
            "describe it", source="data/shot.png", max_side=300
        )
        self.assertIn("data/shot.png", result.return_value)
        self.assertIn("1200x600", result.return_value)

    def test_the_tool_is_registered_only_when_supported(self):
        supported = default_registry(self.runtime(True))
        unsupported = default_registry(self.runtime(False))
        self.assertIn("analyze_image", supported)
        self.assertNotIn("analyze_image", unsupported)


if __name__ == "__main__":
    unittest.main()
