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

from spatial_intelligence.ai import catalog, store
from spatial_intelligence.contracts.errors import ToolInputError


def make_spec(payloads: dict[str, bytes], **overrides) -> catalog.ModelSpec:
    """Build a catalog entry whose files are ``payloads`` keyed by repo path."""
    files = tuple(
        catalog.ModelFile(
            path=name,
            sha256=hashlib.sha256(body).hexdigest(),
            size=len(body),
        )
        for name, body in payloads.items()
    )
    return catalog.ModelSpec(
        id="tiny-model",
        repo="example/tiny-model",
        revision="rev1",
        files=files,
        **overrides,
    )


def git_blob_sha1(body: bytes) -> str:
    """The id the Hub publishes for a file it stores as a plain git object.

    Not sha1 of the bytes: git hashes ``blob <size>\\0`` first, which is why the
    empty file is ``e69de29b…`` rather than ``da39a3ee…``.
    """
    digest = hashlib.sha1()
    digest.update(b"blob %d\0" % len(body))
    digest.update(body)
    return digest.hexdigest()


def plain_spec(payloads: dict[str, bytes]) -> catalog.ModelSpec:
    """A spec whose files carry only the git blob id, as the Hub reports a plain file."""
    files = tuple(
        catalog.ModelFile(path=name, sha256="", size=len(body), sha1=git_blob_sha1(body))
        for name, body in payloads.items()
    )
    return catalog.ModelSpec(
        id="plain-model", repo="example/plain-model", revision="rev1", files=files
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

        result = store.pull(spec, client=make_client(payloads))

        self.assertEqual(result["downloaded"], 2)
        self.assertEqual(sorted(result["files"]), ["decoder.onnx", "encoder.onnx"])
        self.assertEqual(store.file_path(spec, spec.files[0]).read_bytes(), b"encoder-bytes")
        self.assertTrue(store.is_downloaded(spec))
        self.assertEqual(store.verify(spec), [])

        manifest = json.loads(store.manifest_path(spec).read_text(encoding="utf-8"))
        self.assertEqual(manifest["revision"], "rev1")
        self.assertEqual(manifest["files"]["encoder.onnx"]["sha256"], spec.files[0].sha256)

    def test_pull_skips_files_already_present(self):
        payloads = {"encoder.onnx": b"encoder-bytes", "decoder.onnx": b"decoder-bytes"}
        spec = make_spec(payloads)
        store.pull(spec, client=make_client(payloads))

        seen: list[str] = []
        second = store.pull(spec, client=make_client(payloads, seen))

        self.assertEqual(second["downloaded"], 0)
        self.assertEqual(second["skipped"], 2)
        self.assertEqual(seen, [])

    def test_pull_rejects_a_file_that_does_not_match(self):
        spec = make_spec({"encoder.onnx": b"expected"})

        with self.assertRaises(ToolInputError) as caught:
            store.pull(spec, client=make_client({"encoder.onnx": b"tampered"}))

        self.assertIn("failed verification", str(caught.exception))
        self.assertFalse(store.file_path(spec, spec.files[0]).exists())
        self.assertEqual(list(store.spec_dir(spec).glob("*.part")), [])
        self.assertFalse(store.is_downloaded(spec))

    def test_pull_refuses_when_offline(self):
        spec = make_spec({"encoder.onnx": b"payload"})

        with patch.dict("os.environ", {"GEOAI_OFFLINE": "1"}):
            with self.assertRaises(ToolInputError) as caught:
                store.pull(spec, client=make_client({"encoder.onnx": b"payload"}))

        self.assertIn("GEOAI_OFFLINE", str(caught.exception))

    def test_is_downloaded_notices_a_truncated_file(self):
        payloads = {"encoder.onnx": b"encoder-bytes"}
        spec = make_spec(payloads)
        store.pull(spec, client=make_client(payloads))

        store.file_path(spec, spec.files[0]).write_bytes(b"short")

        self.assertFalse(store.is_downloaded(spec))
        self.assertEqual(store.verify(spec), ["encoder.onnx"])

    def test_external_weight_files_stay_beside_their_graph(self):
        payloads = {"onnx/vision.onnx": b"graph", "onnx/vision.onnx_data": b"weights"}
        spec = make_spec(payloads)

        store.pull(spec, client=make_client(payloads))

        self.assertEqual(
            store.file_path(spec, spec.files[1]).parent,
            store.file_path(spec, spec.files[0]).parent,
        )
        self.assertEqual(store.verify(spec), [])

    def test_the_git_blob_id_matches_git_for_an_empty_file(self):
        """A real Hub value: the empty config.yaml of the Prithvi crop repo."""
        self.assertEqual(
            git_blob_sha1(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        )

    def test_pull_accepts_a_file_pinned_only_by_its_git_blob_id(self):
        """Every non-LFS file arrives this way — sha256 is absent, not wrong."""
        payloads = {".gitattributes": b"*.png filter=lfs diff=lfs merge=lfs -text\n"}
        spec = plain_spec(payloads)

        store.pull(spec, client=make_client(payloads))

        self.assertEqual(store.verify(spec), [])
        self.assertTrue(store.is_downloaded(spec))

        # and the re-hash path sees a tampered plain file too
        store.file_path(spec, spec.files[0]).write_bytes(b"swapped")
        self.assertEqual(store.verify(spec), [".gitattributes"])

    def test_pull_accepts_an_empty_file_pinned_by_its_git_blob_id(self):
        payloads = {"config.yaml": b""}
        spec = plain_spec(payloads)

        store.pull(spec, client=make_client(payloads))

        self.assertEqual(store.verify(spec), [])
        self.assertEqual(store.file_path(spec, spec.files[0]).read_bytes(), b"")

    def test_pull_rejects_a_plain_file_whose_bytes_do_not_match(self):
        spec = plain_spec({"README.md": b"expected"})

        with self.assertRaises(ToolInputError) as caught:
            store.pull(spec, client=make_client({"README.md": b"tampered"}))

        self.assertIn("failed verification", str(caught.exception))
        self.assertIn("git-blob-sha1", str(caught.exception))
        self.assertFalse(store.file_path(spec, spec.files[0]).exists())

    def test_a_file_with_no_published_hash_is_checked_by_size(self):
        body = b"payload"
        files = (catalog.ModelFile(path="a.txt", sha256="", size=len(body)),)
        spec = catalog.ModelSpec(
            id="bare", repo="example/bare", revision="rev1", files=files
        )

        store.pull(spec, client=make_client({"a.txt": body}))
        self.assertEqual(store.file_path(spec, files[0]).read_bytes(), body)

        short = catalog.ModelFile(path="a.txt", sha256="", size=len(body) + 1)
        with self.assertRaises(ToolInputError) as caught:
            store.pull(
                catalog.ModelSpec(
                    id="bare", repo="example/bare", revision="rev1", files=(short,)
                ),
                client=make_client({"a.txt": body}),
            )
        self.assertIn("expected 8 bytes, got 7 bytes", str(caught.exception))

    def test_unknown_model_names_the_known_ones(self):
        with self.assertRaises(ToolInputError) as caught:
            catalog.find("does-not-exist")

        self.assertIn("slimsam-77", str(caught.exception))

    def test_summary_reports_disk_state(self):
        payloads = {"encoder.onnx": b"x" * 32}
        spec = make_spec(payloads)

        before = store.summary(spec)
        self.assertFalse(before["downloaded"])
        self.assertEqual(before["on_disk_bytes"], 0)

        store.pull(spec, client=make_client(payloads))

        after = store.summary(spec)
        self.assertTrue(after["downloaded"])
        self.assertEqual(after["on_disk_bytes"], 32)
        self.assertEqual(after["bytes"], 32)
        self.assertEqual(after["task"], "segmentation")

    def test_fetch_pulls_a_pinned_revision_without_a_catalog_entry(self):
        payloads = {"model.pt": b"weights"}
        spec = make_spec(payloads)
        files = list(spec.files)

        result = store.fetch("example/tiny-model", "rev9", files, client=make_client(payloads))

        self.assertEqual(result["downloaded"], 1)
        fetched = store.models_dir() / "example/tiny-model" / "rev9" / "model.pt"
        self.assertEqual(fetched.read_bytes(), b"weights")

    def test_fetch_verifies_the_hash_it_was_given(self):
        files = (catalog.ModelFile(path="model.pt", sha256="0" * 64, size=7),)

        with self.assertRaises(ToolInputError):
            store.fetch(
                "example/tiny-model", "rev1", files, client=make_client({"model.pt": b"weights"})
            )

    def test_shipped_catalog_points_at_verifiable_files(self):
        """The real entry must stay hash-pinned and inside the cache directory."""
        slimsam = catalog.find("slimsam-77")

        self.assertGreater(slimsam.total_bytes, 0)
        self.assertTrue(all(len(file.sha256) == 64 for file in slimsam.files))
        self.assertTrue(all(file.size > 0 for file in slimsam.files))
        self.assertEqual(
            {file.path for file in slimsam.files},
            {"onnx/vision_encoder.onnx", "onnx/prompt_encoder_mask_decoder.onnx"},
        )
        self.assertTrue(store.spec_dir(slimsam).is_relative_to(store.models_dir()))


if __name__ == "__main__":
    unittest.main()
