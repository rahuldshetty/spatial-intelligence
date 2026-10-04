"""The model store: fetching, verifying, and refusing to leave a half model."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from spatial_intelligence.ai import models
from spatial_intelligence.contracts.errors import ToolInputError


def make_spec(payloads: dict[str, bytes], **overrides) -> models.ModelSpec:
    """Build a catalog entry whose files are ``payloads`` keyed by repo path."""
    files = tuple(
        models.ModelFile(
            path=name,
            sha256=hashlib.sha256(body).hexdigest(),
            size=len(body),
        )
        for name, body in payloads.items()
    )
    return models.ModelSpec(
        id="tiny-model",
        repo="example/tiny-model",
        revision="rev1",
        files=files,
        **overrides,
    )


def make_client(payloads: dict[str, bytes], seen: list[str] | None = None) -> httpx.Client:
    """Return a client serving ``payloads`` by repo path, recording requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Key on the repo path, not the file name: a spec may carry nested
        # paths, and two files can share a leaf name.
        tail = request.url.path.split("/resolve/", 1)[-1]
        name = tail.split("/", 1)[1] if "/" in tail else tail
        if seen is not None:
            seen.append(name)
        if name not in payloads:
            return httpx.Response(404)
        return httpx.Response(200, content=payloads[name])

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


class ModelStoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "models"
        self._env = patch.dict(
            os.environ,
            {"GEOAI_MODELS_DIR": str(self.cache)},
            clear=False,
        )
        self._env.start()
        os.environ.pop("GEOAI_OFFLINE", None)

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def test_pull_downloads_verifies_and_manifests(self):
        payloads = {"encoder.onnx": b"encoder-bytes", "decoder.onnx": b"decoder-bytes"}
        spec = make_spec(payloads)

        result = models.pull(spec, client=make_client(payloads))

        self.assertEqual(result["downloaded"], 2)
        self.assertEqual(sorted(result["files"]), ["decoder.onnx", "encoder.onnx"])
        self.assertEqual(models.file_path(spec, spec.files[0]).read_bytes(), b"encoder-bytes")
        self.assertTrue(models.is_downloaded(spec))
        self.assertEqual(models.verify(spec), [])

        manifest = json.loads(models.manifest_path(spec).read_text(encoding="utf-8"))
        self.assertEqual(manifest["revision"], "rev1")
        self.assertEqual(manifest["files"]["encoder.onnx"]["sha256"], spec.files[0].sha256)

    def test_pull_skips_files_already_present(self):
        payloads = {"encoder.onnx": b"encoder-bytes", "decoder.onnx": b"decoder-bytes"}
        spec = make_spec(payloads)
        models.pull(spec, client=make_client(payloads))

        seen: list[str] = []
        second = models.pull(spec, client=make_client(payloads, seen))

        self.assertEqual(second["downloaded"], 0)
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(seen, [])

    def test_pull_rejects_a_file_that_does_not_match(self):
        spec = make_spec({"encoder.onnx": b"expected"})

        with self.assertRaises(ToolInputError) as caught:
            models.pull(spec, client=make_client({"encoder.onnx": b"tampered"}))

        self.assertIn("failed verification", str(caught.exception))
        self.assertFalse(models.file_path(spec, spec.files[0]).exists())
        self.assertEqual(list(models.spec_dir(spec).glob("*.part")), [])
        self.assertFalse(models.is_downloaded(spec))

    def test_pull_refuses_when_offline(self):
        spec = make_spec({"encoder.onnx": b"payload"})

        with patch.dict("os.environ", {"GEOAI_OFFLINE": "1"}):
            with self.assertRaises(ToolInputError) as caught:
                models.pull(spec, client=make_client({"encoder.onnx": b"payload"}))

        self.assertIn("GEOAI_OFFLINE", str(caught.exception))

    def test_is_downloaded_notices_a_truncated_file(self):
        payloads = {"encoder.onnx": b"encoder-bytes"}
        spec = make_spec(payloads)
        models.pull(spec, client=make_client(payloads))

        models.file_path(spec, spec.files[0]).write_bytes(b"short")

        self.assertFalse(models.is_downloaded(spec))
        self.assertEqual(models.verify(spec), ["encoder.onnx"])

    def test_external_weight_files_stay_beside_their_graph(self):
        payloads = {"onnx/vision.onnx": b"graph", "onnx/vision.onnx_data": b"weights"}
        spec = make_spec(payloads)

        models.pull(spec, client=make_client(payloads))

        self.assertEqual(
            models.file_path(spec, spec.files[1]).parent,
            models.file_path(spec, spec.files[0]).parent,
        )
        self.assertEqual(models.verify(spec), [])

    def test_unknown_model_names_the_known_ones(self):
        with self.assertRaises(ToolInputError) as caught:
            models.find("does-not-exist")

        self.assertIn("slimsam-77", str(caught.exception))

    def test_summary_reports_disk_state(self):
        payloads = {"encoder.onnx": b"x" * 32}
        spec = make_spec(payloads)

        before = models.summary(spec)
        self.assertFalse(before["downloaded"])
        self.assertEqual(before["on_disk_bytes"], 0)

        models.pull(spec, client=make_client(payloads))

        after = models.summary(spec)
        self.assertTrue(after["downloaded"])
        self.assertEqual(after["on_disk_bytes"], 32)
        self.assertEqual(after["bytes"], 32)
        self.assertEqual(after["task"], "segmentation")

    def test_shipped_catalog_points_at_verifiable_files(self):
        """The real entry must stay hash-pinned and inside the cache directory."""
        slimsam = models.find("slimsam-77")

        self.assertGreater(slimsam.total_bytes, 0)
        self.assertTrue(all(len(file.sha256) == 64 for file in slimsam.files))
        self.assertTrue(all(file.size > 0 for file in slimsam.files))
        self.assertEqual(
            {file.path for file in slimsam.files},
            {"onnx/vision_encoder.onnx", "onnx/prompt_encoder_mask_decoder.onnx"},
        )
        self.assertTrue(models.spec_dir(slimsam).is_relative_to(models.models_dir()))


if __name__ == "__main__":
    unittest.main()
