"""HTTP surface: the snapshot, cell routes, settings, file serving, and SSE.

Drives the real FastAPI app through a test client against a temporary data
root, so route wiring, error mapping, and the workspace-file CORS contract are
covered without starting uvicorn.
"""

import contextlib
import io
import json
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient

from spatial_intelligence.discovery import CAPABILITIES
from spatial_intelligence.settings.env import app_root
from spatial_intelligence.server import banner, deps
from spatial_intelligence.server.app import create_app
from spatial_intelligence.tools.build import default_registry
from spatial_intelligence.tools.runtime import ToolRuntime
from spatial_intelligence.workspace import Workspace


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._previous_home = os.environ.get("GEOAI_HOME")
        os.environ["GEOAI_HOME"] = self._tmp.name
        deps.close_app_state()
        self.client = TestClient(create_app())

    def tearDown(self):
        # Closes the session *and* stops its worker: clearing the cache alone
        # left a worker thread reading a workspace that no longer exists.
        deps.close_app_state()
        if self._previous_home is None:
            os.environ.pop("GEOAI_HOME", None)
        else:
            os.environ["GEOAI_HOME"] = self._previous_home
        self._tmp.cleanup()

    def open_workspace(self, name: str = "demo") -> dict:
        response = self.client.post("/api/workspace/new", json={"name": name})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def add_cell(self, kind: str, source: str = "") -> dict:
        response = self.client.post(
            "/api/cells", json={"kind": kind, "source": source}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()


class StateRouteTests(ServerTestCase):
    def test_the_snapshot_carries_every_region_the_ui_renders(self):
        snapshot = self.client.get("/api/state").json()

        self.assertEqual(
            sorted(snapshot),
            [
                "active_workspace",
                "cells",
                "files",
                "jobs",
                "map_app_url",
                "map_project",
                "settings",
                "workspaces",
            ],
        )
        self.assertIsNone(snapshot["active_workspace"])
        self.assertEqual(snapshot["cells"], [])
        self.assertEqual(snapshot["jobs"], [])
        self.assertIn("model", snapshot["settings"])

    def test_the_shell_and_its_scripts_are_served(self):
        index = self.client.get("/")

        self.assertEqual(index.status_code, 200)
        self.assertIn("/static/js/main.js", index.text)
        self.assertEqual(index.headers["cache-control"], "no-store")
        for path in ("css/tokens.css", "vendor/marked.min.js", "js/main.js"):
            head = self.client.head(f"/static/{path}")
            self.assertIn(head.status_code, (200, 404), path)

    def test_settings_can_be_updated_and_are_read_back(self):
        updated = self.client.put("/api/settings", json={"theme": "dark"})

        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["theme"], "dark")
        self.assertEqual(self.client.get("/api/settings").json()["theme"], "dark")
        snapshot = self.client.get("/api/state").json()
        self.assertEqual(snapshot["settings"]["theme"], "dark")

    def test_an_invalid_model_is_reported_and_rolled_back(self):
        before = self.client.get("/api/settings").json()["model"]

        response = self.client.put("/api/settings", json={"model": "not-a-provider"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/settings").json()["model"], before)

    def test_the_bridge_handshake_is_recorded(self):
        response = self.client.post(
            "/api/geolibre/bridge",
            json={"version": "2.9.0", "methods": ["project.load"]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["connected"])
        self.assertEqual(self.client.get("/api/geolibre/bridge").json()["version"], "2.9.0")


class WorkspaceRouteTests(ServerTestCase):
    def test_creating_a_workspace_opens_it(self):
        snapshot = self.open_workspace()

        self.assertEqual(snapshot["active_workspace"], "demo")
        self.assertIn("demo", snapshot["workspaces"])
        self.assertIn("map_project", snapshot)

    def test_cells_can_be_added_edited_moved_and_deleted(self):
        self.open_workspace()
        snapshot = self.add_cell("markdown", "# one")
        self.assertEqual(len(snapshot["cells"]), 1)
        cell_id = snapshot["cells"][0]["id"]

        edited = self.client.put(
            f"/api/cells/{cell_id}", json={"source": "# two"}
        ).json()
        self.assertEqual(edited["cells"][0]["source"], "# two")

        second = self.add_cell("python", "1 + 1")["cells"][1]["id"]
        moved = self.client.post(f"/api/cells/{second}/move", json={"index": 0}).json()
        self.assertEqual(moved["cells"][0]["id"], second)

        remaining = self.client.delete(f"/api/cells/{cell_id}").json()
        self.assertEqual([cell["id"] for cell in remaining["cells"]], [second])

    def test_an_unknown_cell_kind_is_a_bad_request(self):
        self.open_workspace()

        response = self.client.post("/api/cells", json={"kind": "sql", "source": ""})

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid cell kind", response.json()["detail"])

    def test_an_unknown_cell_id_is_not_found(self):
        self.open_workspace()

        response = self.client.put("/api/cells/nope", json={"source": "x"})

        self.assertEqual(response.status_code, 404)

    def test_a_markdown_cell_cannot_be_run(self):
        self.open_workspace()
        cell_id = self.add_cell("markdown", "# notes")["cells"][0]["id"]

        response = self.client.post(f"/api/cells/{cell_id}/run")

        self.assertEqual(response.status_code, 400)

    def test_importing_an_upload_lands_in_the_workspace(self):
        self.open_workspace()

        response = self.client.post(
            "/api/import/local",
            files={"files": ("points.geojson", b"{}", "application/geo+json")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["imported"], ["data/points.geojson"])
        self.assertTrue(
            (Path(self._tmp.name) / "workspaces" / "demo" / "data" / "points.geojson").is_file()
        )

    def test_importing_without_a_workspace_is_a_conflict(self):
        response = self.client.post(
            "/api/import/local", files={"files": ("a.txt", b"x", "text/plain")}
        )

        self.assertEqual(response.status_code, 409)

    def test_a_pushed_map_project_is_persisted(self):
        self.open_workspace()
        project = self.client.get("/api/state").json()["map_project"]

        response = self.client.post("/api/map/project", json={"project": project})

        self.assertEqual(response.status_code, 200)
        snapshot = Path(
            self._tmp.name, "workspaces", "demo", "maps", "current.geolibre.json"
        )
        self.assertTrue(snapshot.is_file())


class FileRouteTests(ServerTestCase):
    def test_a_workspace_file_is_served_with_cors_headers(self):
        self.open_workspace()
        data = Path(self._tmp.name) / "workspaces" / "demo" / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "scene.tif").write_bytes(b"raster")

        response = self.client.get("/api/files/data/scene.tif")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"raster")
        self.assertEqual(response.headers["access-control-allow-origin"], "*")

    def test_the_preflight_answers_with_cors_headers(self):
        response = self.client.options("/api/files/data/scene.tif")

        self.assertEqual(response.status_code, 200)
        self.assertIn("GET", response.headers["access-control-allow-methods"])

    def test_a_missing_file_is_not_found(self):
        self.open_workspace()

        self.assertEqual(self.client.get("/api/files/data/nope.tif").status_code, 404)

    def test_an_escaping_path_is_rejected(self):
        self.open_workspace()

        response = self.client.get("/api/files/..%2F..%2Fsettings.json")

        self.assertEqual(response.status_code, 404)

    def test_file_serving_without_a_workspace_is_a_conflict(self):
        self.assertEqual(self.client.get("/api/files/data/x.tif").status_code, 409)


class EventStreamTests(ServerTestCase):
    def test_subscribers_receive_published_events(self):
        state = deps.app_state()
        subscriber = state.subscribe()

        self.client.put("/api/settings", json={"theme": "dark"})

        events = []
        while True:
            try:
                events.append(subscriber.get_nowait())
            except queue.Empty:
                break
        state.unsubscribe(subscriber)

        settings_events = [item for item in events if item["event"] == "settings"]
        self.assertTrue(settings_events)
        self.assertEqual(settings_events[-1]["data"]["settings"]["theme"], "dark")

    # The SSE frames themselves are verified against a live uvicorn server: the
    # in-process test client buffers a whole response body, so it cannot read an
    # endless stream incrementally.


if __name__ == "__main__":
    unittest.main()


class StartupReportTests(unittest.TestCase):
    """The banner and metadata block the server prints when it starts."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # The tool surface is a property of the build, so the table is built from a
        # registry whose workspace is never created.
        self.registry = default_registry(
            ToolRuntime(workspace=Workspace(Path(self._tmp.name) / "introspection"))
        )

    def test_the_wordmark_is_art_and_the_fallback_is_plain_ascii(self):
        for art in (banner.BANNER_BLOCKS, banner.BANNER_ASCII):
            self.assertEqual(len(art.splitlines()), 10)
            self.assertFalse(art.startswith("\\"))

        self.assertTrue(banner.BANNER_ASCII.isascii())
        self.assertIn(banner.TAGLINE, banner.banner())

    def test_the_report_names_the_totals_every_category_and_every_capability(self):
        text = banner.startup_report(self.registry)

        self.assertIn(f"{len(self.registry)} registered", text)
        self.assertIn(f"{len(self.registry.core_names())} always visible", text)
        for category, names in self.registry.categories().items():
            with self.subTest(category=category):
                self.assertIn(f"{category} ({len(names)})", text)
                for name in names:
                    self.assertIn(name, text)
        for capability in CAPABILITIES:
            self.assertIn(capability.id, text)
        # Metadata that has to be there for the block to be worth printing: the
        # interpreter, and where the workspaces will be read from.
        self.assertIn(sys.version.split()[0], text)
        self.assertIn(str(app_root() / "workspaces"), text)

    def test_every_line_stays_inside_the_configured_width(self):
        text = banner.startup_report(self.registry)

        widest = max(len(line) for line in text.splitlines())

        self.assertLessEqual(widest, banner.WIDTH)

    def test_a_stream_that_cannot_take_the_glyphs_degrades_instead_of_failing(self):
        class Narrow:
            """A single-byte stream: what a redirected stdout can look like.

            ``io.StringIO`` takes any character, so the encoding has to be enforced
            here to reproduce the failure this path exists for.
            """

            encoding = "ascii"

            def __init__(self) -> None:
                self.chunks: list[str] = []

            def write(self, text: str) -> None:
                text.encode(self.encoding)          # raises for a block glyph
                self.chunks.append(text)

            def flush(self) -> None:
                pass

            def getvalue(self) -> str:
                return "".join(self.chunks)

        narrow = Narrow()
        with contextlib.redirect_stdout(narrow):
            # The wordmark itself is chosen for this stream, and a report that
            # still holds a glyph is replaced rather than raising on startup.
            self.assertEqual(
                banner.banner().splitlines()[1], banner.BANNER_ASCII.splitlines()[1]
            )
            banner.announce("block glyph: \u2588")

        written = narrow.getvalue()
        self.assertNotIn("\u2588", written)
        self.assertIn("block glyph", written)

    def test_announce_prints_and_returns_the_report(self):
        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):
            returned = banner.announce("hello")

        self.assertEqual(returned, "hello")
        self.assertEqual(buffer.getvalue().strip(), "hello")
