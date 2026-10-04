"""Files service and pack: confinement, downloads, and progress reporting."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_intelligence.contracts.errors import ToolInputError, WorkspaceError
from spatial_intelligence.contracts.progress import JobState, Reporter
from spatial_intelligence.tools import ToolRegistry, ToolRuntime
from spatial_intelligence.tools.packs.files import FilesPack
from spatial_intelligence.tools.runtime import RuntimeEvents
from spatial_intelligence.workspace import Workspace
from spatial_intelligence.workspace import files as fileops


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event.as_dict())

    def for_job(self, job_id):
        return [event for event in self.events if event["job_id"] == job_id]

    def statuses(self):
        return [event["status"] for event in self.events]


class FakeResponse:
    def __init__(self, body: bytes, *, content_length: str | None = None):
        self._body = body
        self._offset = 0
        self.headers = {
            "Content-Length": str(len(body)) if content_length is None else content_length
        }

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._body)
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FilesTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()

    def tearDown(self):
        self._tmp.cleanup()

    def data_file(self, relative: str, content: str = "hello") -> Path:
        target = self.workspace.resolve(relative, write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
        return target


class WorkspaceFileTests(FilesTestCase):
    def test_the_read_cap_scales_with_the_context_window(self):
        """A read must not be able to spend the window before compaction sees it."""
        self.assertEqual(fileops.read_limit_bytes(0), fileops.MAX_READ_BYTES)  # unknown window
        self.assertEqual(fileops.read_limit_bytes(200_000), 100_000)
        self.assertEqual(fileops.read_limit_bytes(1_000_000), 500_000)
        # A window too small to scale still allows a useful read...
        self.assertEqual(fileops.read_limit_bytes(32_000), fileops.MIN_READ_BYTES)
        # ...and a huge one still stops at the ceiling.
        self.assertEqual(fileops.read_limit_bytes(4_000_000), fileops.MAX_READ_BYTES)

    def test_listing_is_workspace_relative_and_sorted(self):
        self.data_file("data/b.txt")
        self.data_file("data/a.txt")
        self.data_file("results/out.tif", "x")

        self.assertEqual(
            fileops.list_files(self.workspace),
            ["data/a.txt", "data/b.txt", "results/out.tif", "workspace.json"],
        )
        self.assertEqual(fileops.list_files(self.workspace, "data"), ["data/a.txt", "data/b.txt"])
        self.assertEqual(fileops.find_files(self.workspace, "*.txt"), ["data/a.txt", "data/b.txt"])

    def test_read_returns_the_whole_file_or_a_slice(self):
        self.data_file("data/notes.txt", "alpha\nbeta\n")

        self.assertEqual(fileops.read_text(self.workspace, "data/notes.txt"), "alpha\nbeta\n")
        self.assertEqual(
            fileops.read_text(self.workspace, "data/notes.txt", offset=6, limit=4), "beta"
        )

    def test_read_refuses_a_file_larger_than_the_cap(self):
        self.data_file("data/big.txt", "x" * 64)

        with self.assertRaises(ToolInputError) as caught:
            fileops.read_text(self.workspace, "data/big.txt", max_bytes=8)

        self.assertIn("file too large", str(caught.exception))

    def test_read_reports_a_missing_file(self):
        with self.assertRaises(WorkspaceError):
            fileops.read_text(self.workspace, "data/missing.txt")

    def test_write_is_confined_to_output_directories(self):
        written = fileops.write_text(self.workspace, "results/a.geojson", "{}")

        self.assertEqual(written.read_text(encoding="utf-8"), "{}")
        with self.assertRaises(WorkspaceError):
            fileops.write_text(self.workspace, "traces/a.jsonl", "{}")
        with self.assertRaises(WorkspaceError):
            fileops.write_text(self.workspace, "../outside.txt", "x")
        with self.assertRaises(WorkspaceError):
            fileops.write_text(self.workspace, str(Path(self._tmp.name) / "abs.txt"), "x")

    def test_import_copies_files_and_directories_without_overwriting(self):
        source_dir = Path(self._tmp.name) / "incoming"
        source_dir.mkdir()
        (source_dir / "one.txt").write_text("1", encoding="utf-8")

        first = fileops.import_path(self.workspace, str(source_dir))
        second = fileops.import_path(self.workspace, str(source_dir))

        self.assertEqual(self.workspace.relative(first), "data/incoming")
        self.assertEqual(self.workspace.relative(second), "data/incoming_1")
        self.assertTrue((second / "one.txt").is_file())

    def test_import_reports_a_missing_source(self):
        with self.assertRaises(WorkspaceError):
            fileops.import_path(self.workspace, str(Path(self._tmp.name) / "nope"))


class DownloadTests(FilesTestCase):
    def opener(self, response: FakeResponse):
        def open_url(request, timeout=0):
            return response

        return open_url

    def test_download_streams_into_data_and_reports_bytes(self):
        sink = RecordingSink()
        reporter = Reporter(sink, parent_id="cell-1")
        job = reporter.job("download", "scene.tif", unit="bytes")

        with patch.object(fileops, "PROGRESS_INTERVAL_BYTES", 4), patch.object(
            fileops, "PROGRESS_INTERVAL_SECONDS", 3600
        ):
            path = fileops.download_file(
                self.workspace,
                "https://example.com/scene.tif",
                "scene.tif",
                job=job,
                opener=self.opener(FakeResponse(b"asset")),
            )

        self.assertEqual(path.read_bytes(), b"asset")
        self.assertEqual(self.workspace.relative(path), "data/scene.tif")
        # Opening event, then the size-known event, then one byte report.
        self.assertEqual(sink.statuses(), ["running", "running", "running"])
        self.assertIsNone(sink.events[0]["total"])
        self.assertEqual(sink.events[1]["total"], 5)
        self.assertEqual(sink.events[1]["completed"], 0)
        self.assertEqual(sink.events[2]["completed"], 5)
        self.assertEqual(sink.events[0]["kind"], "download")
        self.assertEqual(sink.events[0]["unit"], "bytes")
        self.assertEqual(sink.events[0]["parent_id"], "cell-1")

    def test_download_does_not_report_again_for_a_body_under_the_interval(self):
        sink = RecordingSink()
        job = Reporter(sink, parent_id="cell-1").job("download", "tiny.tif", unit="bytes")

        fileops.download_file(
            self.workspace,
            "https://example.com/tiny.tif",
            "tiny.tif",
            job=job,
            opener=self.opener(FakeResponse(b"abcd")),
        )

        self.assertEqual(sink.statuses(), ["running", "running"])
        self.assertEqual(sink.events[1]["total"], 4)

    def test_download_rejects_a_non_http_url(self):
        with self.assertRaises(ToolInputError):
            fileops.download_file(self.workspace, "ftp://example.com/a.tif", "a.tif")

    def test_download_refuses_an_oversized_content_length_before_reading(self):
        response = FakeResponse(b"tiny", content_length=str(3 * 1024**3))

        with self.assertRaises(ToolInputError) as caught:
            fileops.download_file(
                self.workspace,
                "https://example.com/huge.tif",
                "huge.tif",
                opener=self.opener(response),
            )

        self.assertIn("2 GB cap", str(caught.exception))
        self.assertEqual(list(self.workspace.data.iterdir()), [])

    def test_failed_download_leaves_no_partial_file(self):
        def explode(request, timeout=0):
            raise OSError("connection reset")

        with self.assertRaises(OSError):
            fileops.download_file(
                self.workspace,
                "https://example.com/a.tif",
                "a.tif",
                opener=explode,
            )

        self.assertEqual(list(self.workspace.data.iterdir()), [])

    def test_prepare_downloads_validates_input(self):
        with self.assertRaises(ToolInputError):
            fileops.prepare_downloads([])
        with self.assertRaises(ToolInputError):
            fileops.prepare_downloads([{"url": "https://example.com/a.tif"}] * 21)
        with self.assertRaises(ToolInputError):
            fileops.prepare_downloads([{"filename": "a.tif"}])
        with self.assertRaises(ToolInputError):
            fileops.prepare_downloads([{"url": "file:///etc/passwd"}])

        prepared = fileops.prepare_downloads(
            [{"url": "https://example.com/a.tif"}, {"url": "https://other.example/a.tif"}]
        )

        self.assertEqual([name for _, name in prepared], ["a.tif", "a_1.tif"])

    def test_download_many_isolates_sibling_failures(self):
        sink = RecordingSink()
        reporter = Reporter(sink, parent_id="cell-3")

        def open_url(request, timeout=0):
            if "bad" in request.full_url:
                raise OSError("boom")
            return FakeResponse(request.full_url.encode("utf-8"))

        with patch(
            "spatial_intelligence.workspace.files.urllib.request.urlopen",
            side_effect=open_url,
        ):
            results = fileops.download_many(
                self.workspace,
                [
                    {"url": "https://example.com/one.tif"},
                    {"url": "https://example.com/bad.tif"},
                    {"url": "https://example.com/two.tif"},
                ],
                reporter=reporter,
            )

        self.assertEqual([result["status"] for result in results], ["done", "error", "done"])
        self.assertEqual(Path(results[0]["path"]).name, "one.tif")
        self.assertIn("boom", results[1]["error"])
        self.assertTrue((self.workspace.data / "two.tif").is_file())

        jobs = {event["job_id"] for event in sink.events}
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(event["parent_id"] == "cell-3" for event in sink.events))
        self.assertIn("error", sink.statuses())
        completed = [
            event for event in sink.events if event["status"] == JobState.DONE.value
        ]
        self.assertEqual(
            sorted(event["artifact"] for event in completed),
            ["data/one.tif", "data/two.tif"],
        )


class FilesPackTests(FilesTestCase):
    def build_pack(self, sink: RecordingSink, notifications: list[str]):
        runtime = ToolRuntime(
            workspace=self.workspace,
            reporter=Reporter(sink, parent_id="cell-42"),
            events=RuntimeEvents(files_changed=lambda: notifications.append("files")),
        )
        registry = ToolRegistry()
        registry.add_pack(FilesPack, runtime)
        return registry

    def test_read_file_refuses_more_than_the_deployment_cap(self):
        """The cap is the deployment's, not the model's to raise."""
        runtime = ToolRuntime(workspace=self.workspace, max_read_bytes=200)
        registry = ToolRegistry()
        registry.add_pack(FilesPack, runtime)
        self.data_file("data/notes.txt", "x" * 400)
        read = registry.get("read_file").callable

        with self.assertRaises(ToolInputError) as caught:
            read("data/notes.txt")
        self.assertIn("read a slice with offset/limit", str(caught.exception))

        # A slice within the cap still reads.
        self.assertEqual(read("data/notes.txt", 200, 0, 10), "x" * 10)

        with self.assertRaises(ToolInputError) as caught:
            read("data/notes.txt", 5000)
        self.assertIn("above this deployment's read cap", str(caught.exception))

    def test_write_file_records_the_output_and_notifies(self):
        notifications: list[str] = []
        registry = self.build_pack(RecordingSink(), notifications)

        absolute = registry.get("write_file").callable("results/notes.md", "# Notes")

        self.assertEqual(Path(absolute).read_text(encoding="utf-8"), "# Notes")
        self.assertEqual(notifications, ["files"])
        manifest = self.workspace._read_manifest()
        self.assertIn("results/notes.md", manifest["outputs"])

    def test_download_reports_one_parent_scoped_job(self):
        sink = RecordingSink()
        registry = self.build_pack(sink, [])

        with patch(
            "spatial_intelligence.workspace.files.urllib.request.urlopen",
            return_value=FakeResponse(b"asset"),
        ):
            relative = registry.get("download").callable(
                "https://example.com/scene.tif", "scene.tif"
            )

        self.assertEqual(relative, "data/scene.tif")
        self.assertEqual(sink.statuses(), ["running", "running", "done"])
        self.assertEqual(sink.events[-1]["artifact"], "data/scene.tif")
        self.assertEqual(sink.events[-1]["parent_id"], "cell-42")
        self.assertEqual({event["kind"] for event in sink.events}, {"download"})

    def test_failed_download_reports_a_failed_job(self):
        sink = RecordingSink()
        registry = self.build_pack(sink, [])

        def explode(request, timeout=0):
            raise OSError("connection reset")

        with patch(
            "spatial_intelligence.workspace.files.urllib.request.urlopen",
            side_effect=explode,
        ):
            with self.assertRaises(OSError):
                registry.get("download").callable("https://example.com/a.tif", "a.tif")

        self.assertIs(sink.statuses()[-1], JobState.ERROR.value)
        self.assertIn("connection reset", sink.events[-1]["error"])

    def test_download_files_notifies_files_once(self):
        sink = RecordingSink()
        notifications: list[str] = []
        registry = self.build_pack(sink, notifications)

        def open_url(request, timeout=0):
            return FakeResponse(request.full_url.encode("utf-8"))

        with patch(
            "spatial_intelligence.workspace.files.urllib.request.urlopen",
            side_effect=open_url,
        ):
            results = registry.get("download_files").callable(
                [{"url": "https://example.com/one.tif"}, {"url": "https://example.com/two.tif"}]
            )

        self.assertEqual([result["status"] for result in results], ["done", "done"])
        self.assertEqual(notifications, ["files"])

    def test_registered_file_tools_expose_expected_effects(self):
        registry = self.build_pack(RecordingSink(), [])

        self.assertEqual(
            sorted(registry.names()),
            ["download", "download_files", "find_files", "list_files", "read_file", "write_file"],
        )
        self.assertEqual(
            registry.core_names(), frozenset({"list_files", "find_files", "read_file", "write_file"})
        )
        self.assertTrue(registry.replay_safe("read_file"))
        self.assertFalse(registry.replay_safe("download"))
        self.assertEqual(registry.get("download").kind.value, "reporting")


if __name__ == "__main__":
    unittest.main()
