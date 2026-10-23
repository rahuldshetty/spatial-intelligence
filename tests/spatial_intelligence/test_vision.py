"""Image-input probing, map capture, and the conditional image tool."""

from __future__ import annotations

import base64
import io
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from PIL import Image

from spatial_intelligence.agent import imaging
from spatial_intelligence.agent.vision import VisionService, probe_now
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.contracts.services import CAPTURE_SERVICE, VISION_SERVICE
from spatial_intelligence.session.bus import EventBus
from spatial_intelligence.session.capture import CaptureBroker, MapCapture
from spatial_intelligence.session.jobs import JobRegistry
from spatial_intelligence.session.services import SessionServices
from spatial_intelligence.tools.build import default_registry
from spatial_intelligence.tools.packs.python_ import PythonPack
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


class VisionAskTests(unittest.TestCase):
    def _service(self, model: str, base_url: str) -> VisionService:
        with patch.dict(
            os.environ,
            {"OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": "test-key"},
        ):
            service = VisionService(model, enabled=True)
        service._state = "supported"
        return service

    def test_deepseek_direct_mode_is_sent_without_thinking(self):
        service = self._service(
            "openai:deepseek-flash", "https://api.deepseek.com"
        )
        response = FakeResponse(
            200,
            "{}",
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "red"}}
                ]
            },
        )
        with patch("spatial_intelligence.agent.vision.httpx.post", return_value=response) as post:
            with patch.dict(
                os.environ,
                {"OPENAI_BASE_URL": "https://api.deepseek.com", "OPENAI_API_KEY": "test-key"},
            ):
                self.assertEqual(service.ask("what color?", png_bytes()), "red")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_openai_does_not_receive_deepseek_thinking(self):
        service = self._service("openai:gpt-4o", "https://api.openai.com/v1")
        response = FakeResponse(
            200,
            "{}",
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "red"}}
                ]
            },
        )
        with patch("spatial_intelligence.agent.vision.httpx.post", return_value=response) as post:
            with patch.dict(
                os.environ,
                {"OPENAI_BASE_URL": "", "OPENAI_API_KEY": "test-key"},
            ):
                service.ask("what color?", png_bytes())
        self.assertNotIn("thinking", post.call_args.kwargs["json"])

    def test_reasoning_exhaustion_is_reported_not_returned_as_empty(self):
        service = self._service(
            "openai:deepseek-flash", "https://api.deepseek.com"
        )
        response = FakeResponse(
            200,
            "{}",
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "",
                            "reasoning_content": "thinking about boxes",
                        },
                    }
                ]
            },
        )
        with patch("spatial_intelligence.agent.vision.httpx.post", return_value=response):
            with patch.dict(
                os.environ,
                {"OPENAI_BASE_URL": "https://api.deepseek.com", "OPENAI_API_KEY": "test-key"},
            ):
                with self.assertRaises(ToolInputError) as raised:
                    service.ask("give boxes", png_bytes())
        self.assertIn("reasoning", str(raised.exception).lower())


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
        self.calls = []

    def ask(
        self,
        question,
        data,
        media_type="image/jpeg",
        *,
        thinking=False,
        max_tokens=2048,
    ):
        self.calls.append((question, data, media_type, thinking, max_tokens))
        return "a red square"


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


class PythonVisionTests(unittest.TestCase):
    def test_the_python_facade_gets_the_session_vision_service(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = Workspace(root).create()
            (workspace.data / "shot.png").write_bytes(png_bytes(8, 8))
            vision = FakeVision(True)
            runtime = ToolRuntime(
                workspace=workspace,
                services={VISION_SERVICE: vision},
            )
            pack = PythonPack(runtime)
            namespace = pack.executor.namespace()
            facade = namespace["vision"]

            self.assertTrue(facade.supported)
            self.assertEqual(
                facade.ask("what is it?", image="data/shot.png"),
                "a red square",
            )
            self.assertEqual(vision.calls[0][0], "what is it?")
            self.assertTrue(vision.calls[0][1])


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

    def test_map_source_returns_the_vision_answer(self):
        runtime = self.runtime()
        result = VisionPack(runtime).analyze_image("what is shown?")
        self.assertEqual(result, "a red square")
        vision = runtime.services[VISION_SERVICE]
        self.assertEqual(vision.calls[0][0], "what is shown?")
        self.assertEqual(vision.calls[0][2], "image/jpeg")
        self.assertTrue(vision.calls[0][1])

    def test_a_workspace_image_is_read_and_resized(self):
        path = self.workspace.data / "shot.png"
        path.write_bytes(png_bytes(1200, 600))
        runtime = self.runtime()
        result = VisionPack(runtime).analyze_image(
            "describe it", source="data/shot.png", max_side=300
        )
        self.assertEqual(result, "a red square")
        data = runtime.services[VISION_SERVICE].calls[0][1]
        with Image.open(io.BytesIO(data)) as image:
            self.assertLessEqual(max(image.size), 300)

    def test_a_raster_can_select_its_rgb_bands(self):
        path = self.workspace.data / "false_color.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=16,
            height=16,
            count=4,
            dtype="uint8",
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(0, 1, 0.01, 0.01),
        ) as dataset:
            for band in range(1, 5):
                dataset.write(
                    np.full((16, 16), band * 50, dtype="uint8"), band
                )

        runtime = self.runtime()
        with patch(
            "spatial_intelligence.agent.image_analysis.load_image",
            wraps=imaging.load_image,
        ) as loader:
            result = VisionPack(runtime).analyze_image(
                "describe it", source="data/false_color.tif", bands=[4, 2, 3]
            )

        self.assertEqual(result, "a red square")
        self.assertEqual(loader.call_args.args[2], [4, 2, 3])

    def test_tool_and_python_facade_share_the_vision_answer(self):
        path = self.workspace.data / "shot.png"
        path.write_bytes(png_bytes(40, 30))
        runtime = self.runtime()

        tool_answer = VisionPack(runtime).analyze_image(
            "describe it", source="data/shot.png"
        )
        facade = PythonPack(runtime).executor.namespace()["vision"]
        python_answer = facade.ask("describe it", image="data/shot.png")

        self.assertEqual(tool_answer, "a red square")
        self.assertEqual(python_answer, "a red square")
        self.assertEqual(
            [call[0] for call in runtime.services[VISION_SERVICE].calls],
            ["describe it", "describe it"],
        )

    def test_an_out_of_range_band_is_rejected(self):
        path = self.workspace.data / "three_band.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=8,
            height=8,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(0, 1, 0.01, 0.01),
        ) as dataset:
            dataset.write(np.zeros((3, 8, 8), dtype="uint8"))

        with self.assertRaises(ToolInputError):
            VisionPack(self.runtime()).analyze_image(
                "describe", source="data/three_band.tif", bands=[4]
            )

    def test_bands_are_rejected_for_the_map_capture(self):
        with self.assertRaises(ToolInputError):
            VisionPack(self.runtime()).analyze_image("describe", bands=[1, 2, 3])

    def test_the_tool_is_registered_only_when_supported(self):
        supported = default_registry(self.runtime(True))
        unsupported = default_registry(self.runtime(False))
        self.assertIn("analyze_image", supported)
        self.assertNotIn("analyze_image", unsupported)


if __name__ == "__main__":
    unittest.main()
