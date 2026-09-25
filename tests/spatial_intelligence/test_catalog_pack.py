"""Catalog services and pack: canned catalog JSON, the scene cache, downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote, urlencode

from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.contracts.progress import JobState, Reporter
from spatial_intelligence.geo.catalogs import openaerialmap, vantor
from spatial_intelligence.geo.catalogs.scenes import CACHE_FILENAME, SceneCache
from spatial_intelligence.map import document, layers
from spatial_intelligence.tools import ToolRegistry, ToolRuntime
from spatial_intelligence.tools.packs.catalog import CatalogPack
from spatial_intelligence.tools.runtime import RuntimeEvents
from spatial_intelligence.workspace import Workspace

VANTOR_EVENT_URL = f"{vantor.VANTOR_CATALOG.rsplit('/', 1)[0]}/Nepal/catalog.json"
VANTOR_ITEM_URL = f"{vantor.VANTOR_CATALOG.rsplit('/', 1)[0]}/Nepal/post.json"
OAM_ASSET_URL = "https://example.com/aerial.tif"


def vantor_child(href: str, title: str) -> dict:
    return {"rel": "child", "href": href, "title": title}


def vantor_item(item_id: str = "scene-1", phase: str = "post-event") -> dict:
    return {
        "id": item_id,
        "bbox": [85.0, 27.0, 86.0, 28.0],
        "properties": {
            "datetime": "2026-08-27T00:00:00Z",
            "phase": phase,
            "vehicle_name": "WV03",
            "eo:cloud_cover": 5,
            "pan_gsd": 0.3,
        },
        "assets": {
            "visual": {"href": "https://example.com/post.tif", "type": "image/tiff"},
            "thumbnail": {"href": "https://example.com/post.jpg"},
        },
    }


def oam_meta_url(limit: int = 20, page: int = 1) -> str:
    query = urlencode(
        {
            "limit": limit,
            "page": page,
            "order_by": "acquisition_end",
            "sort": "desc",
            "bbox": "85.0,27.0,86.0,28.0",
        }
    )
    return f"{openaerialmap.OAM_API}/meta?{query}"


def oam_body() -> dict:
    return {
        "meta": {"found": "1"},
        "results": [
            {
                "_id": "oam-1",
                "uuid": OAM_ASSET_URL,
                "title": "Flood survey",
                "provider": "Example",
                "acquisition_end": "2026-08-28T00:00:00Z",
                "bbox": [85.0, 27.0, 86.0, 28.0],
                "properties": {"thumbnail": "https://example.com/thumb.png"},
            }
        ],
    }


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body
        self._offset = 0
        self.headers = {"Content-Length": str(len(body))}

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


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event.as_dict())

    def statuses(self):
        return [event["status"] for event in self.events]

    def job_ids(self):
        return sorted({event["job_id"] for event in self.events})

    def completed(self):
        return [event for event in self.events if event["status"] == JobState.DONE.value]


class CatalogTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()

    def tearDown(self):
        self._tmp.cleanup()

    def serve(self, responses: dict):
        """Return a ``urlopen`` replacement serving one canned body per URL."""

        def open_url(request, timeout=0):
            url = getattr(request, "full_url", request)
            if url not in responses:
                raise AssertionError(f"unexpected catalog URL: {url}")
            body = responses[url]
            if isinstance(body, Exception):
                raise body
            if isinstance(body, bytes):
                return FakeResponse(body)
            return FakeResponse(json.dumps(body).encode("utf-8"))

        return open_url


class VantorServiceTests(CatalogTestCase):
    def test_event_search_filters_by_title(self):
        catalog = {
            "links": [
                vantor_child("Nepal-Flooding-Aug-2026/catalog.json", "Nepal-Flooding-Aug-2026"),
                vantor_child("Other-Earthquake/catalog.json", "Other-Earthquake"),
                vantor_child("Nepal-Flooding-Jul-2026/catalog.json", "Nepal-Flooding-Jul-2026"),
            ]
        }

        with patch("urllib.request.urlopen", side_effect=self.serve({vantor.VANTOR_CATALOG: catalog})):
            found = vantor.search_vantor_events("load Nepal flood data")
            unfiltered = vantor.search_vantor_events()

        self.assertEqual(
            [event["id"] for event in found],
            ["Nepal-Flooding-Aug-2026", "Nepal-Flooding-Jul-2026"],
        )
        self.assertEqual(found[0]["provider"], "Vantor Open Data")
        self.assertEqual(len(unfiltered), 3)

    def test_event_search_clamps_limit(self):
        catalog = {
            "links": [
                vantor_child(f"Event-{index:03d}/catalog.json", f"Event-{index:03d}")
                for index in range(150)
            ]
        }

        with patch("urllib.request.urlopen", side_effect=self.serve({vantor.VANTOR_CATALOG: catalog})):
            capped = vantor.search_vantor_events(limit=999)
            minimum = vantor.search_vantor_events(limit=0)

        self.assertEqual(len(capped), 100)
        self.assertEqual(len(minimum), 1)

    def test_imagery_normalizes_scenes_and_filters_by_phase(self):
        cache = SceneCache(self.workspace)
        responses = {
            vantor.VANTOR_CATALOG: {
                "links": [vantor_child("Nepal/catalog.json", "Nepal")]
            },
            VANTOR_EVENT_URL: {"links": [{"rel": "item", "href": "post.json"}]},
            VANTOR_ITEM_URL: vantor_item(),
        }

        with patch("urllib.request.urlopen", side_effect=self.serve(responses)):
            scenes = vantor.search_vantor_imagery(
                "Nepal", bounds=[85.2, 27.2, 85.8, 27.8], phase="post", cache=cache
            )
            pre_event = vantor.search_vantor_imagery("Nepal", phase="pre", cache=cache)

        self.assertEqual(len(scenes), 1)
        scene = scenes[0]
        self.assertEqual(
            sorted(scene),
            [
                "asset_url",
                "bbox",
                "cloud_cover",
                "datetime",
                "event",
                "gsd",
                "id",
                "phase",
                "provider",
                "render",
                "scene_key",
                "sensor",
                "thumbnail_url",
                "title",
            ],
        )
        self.assertEqual(scene["id"], "scene-1")
        self.assertEqual(scene["sensor"], "WV03")
        self.assertEqual(scene["cloud_cover"], 5)
        self.assertEqual(scene["gsd"], 0.3)
        self.assertEqual(scene["phase"], "post")
        self.assertEqual(scene["render"], "cog")
        self.assertEqual(scene["asset_url"], "https://example.com/post.tif")
        self.assertEqual(scene["thumbnail_url"], "https://example.com/post.jpg")
        self.assertTrue(scene["scene_key"].startswith("vantor:scene-1:"))
        self.assertEqual(pre_event, [])
        self.assertEqual(cache.get(scene["scene_key"])["scene_key"], scene["scene_key"])

    def test_imagery_rejects_bad_bounds_phase_and_unknown_event(self):
        cache = SceneCache(self.workspace)
        responses = {
            vantor.VANTOR_CATALOG: {
                "links": [vantor_child("Nepal/catalog.json", "Nepal")]
            }
        }

        with patch("urllib.request.urlopen", side_effect=self.serve(responses)):
            with self.assertRaises(ToolInputError):
                vantor.search_vantor_imagery(
                    "Nepal", bounds=[85.8, 27.2, 85.2, 27.8], cache=cache
                )
            with self.assertRaises(ToolInputError):
                vantor.search_vantor_imagery("Nepal", bounds=[85.0, 27.0, 86.0], cache=cache)
            with self.assertRaises(ToolInputError):
                vantor.search_vantor_imagery("Nepal", phase="during", cache=cache)
            with self.assertRaises(ToolInputError) as caught:
                vantor.search_vantor_imagery("Nowhere", cache=cache)

        self.assertIn("Vantor event not found", str(caught.exception))

    def test_imagery_skips_an_item_that_fails_to_fetch(self):
        cache = SceneCache(self.workspace)
        responses = {
            vantor.VANTOR_CATALOG: {
                "links": [vantor_child("Nepal/catalog.json", "Nepal")]
            },
            VANTOR_EVENT_URL: {
                "links": [
                    {"rel": "item", "href": "post.json"},
                    {"rel": "item", "href": "broken.json"},
                ]
            },
            VANTOR_ITEM_URL: vantor_item(),
            f"{vantor.VANTOR_CATALOG.rsplit('/', 1)[0]}/Nepal/broken.json": OSError("truncated"),
        }

        with patch("urllib.request.urlopen", side_effect=self.serve(responses)):
            scenes = vantor.search_vantor_imagery("Nepal", cache=cache)

        self.assertEqual([scene["id"] for scene in scenes], ["scene-1"])

    def test_imagery_rejects_a_non_https_catalog_url(self):
        with self.assertRaises(ToolInputError):
            vantor.fetch_json("http://example.com/catalog.json")


class OpenAerialMapServiceTests(CatalogTestCase):
    def test_search_returns_the_documented_result_and_caches_scenes(self):
        bounds = [85.0, 27.0, 86.0, 28.0]
        cache = SceneCache(self.workspace)

        with patch(
            "urllib.request.urlopen",
            side_effect=self.serve({oam_meta_url(): oam_body()}),
        ):
            result = openaerialmap.search_openaerialmap(bounds, cache=cache)

        self.assertEqual(sorted(result), ["found", "limit", "page", "scenes"])
        self.assertEqual(result["found"], 1)
        self.assertEqual(result["page"], 1)
        self.assertEqual(result["limit"], 20)
        scene = result["scenes"][0]
        self.assertEqual(scene["id"], "oam-1")
        self.assertEqual(scene["render"], "xyz")
        self.assertIn("titiler.hotosm.org", scene["tile_url"])
        self.assertIn(quote(OAM_ASSET_URL, safe=""), scene["tile_url"])
        self.assertTrue(scene["scene_key"].startswith("oam:oam-1:"))
        self.assertEqual(scene["thumbnail_url"], "https://example.com/thumb.png")
        self.assertEqual(cache.get(scene["scene_key"])["tile_url"], scene["tile_url"])

    def test_search_rejects_bad_bounds(self):
        cache = SceneCache(self.workspace)

        with self.assertRaises(ToolInputError):
            openaerialmap.search_openaerialmap([85.0, 27.0, 84.0, 28.0], cache=cache)
        with self.assertRaises(ToolInputError):
            openaerialmap.search_openaerialmap([85.0, 27.0, 86.0], cache=cache)


class SceneCacheTests(CatalogTestCase):
    def test_cache_survives_a_new_instance(self):
        cache = SceneCache(self.workspace)
        cache.remember(
            {"scene_key": "oam:oam-1:abc123", "id": "oam-1", "asset_url": OAM_ASSET_URL}
        )

        stored = json.loads((self.workspace.traces / CACHE_FILENAME).read_text(encoding="utf-8"))
        fresh = SceneCache(self.workspace)

        self.assertEqual(sorted(stored), ["oam:oam-1:abc123"])
        self.assertEqual(fresh.get("oam:oam-1:abc123")["asset_url"], OAM_ASSET_URL)
        self.assertEqual(fresh.get("oam:unknown:zzz"), {})

    def test_cache_keeps_only_the_newest_scenes(self):
        cache = SceneCache(self.workspace)
        for index in range(510):
            cache.remember(
                {
                    "scene_key": f"oam:scene-{index}:k{index}",
                    "id": f"scene-{index}",
                    "asset_url": f"https://example.com/{index}.tif",
                }
            )

        stored = json.loads((self.workspace.traces / CACHE_FILENAME).read_text(encoding="utf-8"))
        fresh = SceneCache(self.workspace)

        self.assertEqual(len(stored), 500)
        self.assertEqual(fresh.get("oam:scene-9:k9"), {})
        self.assertEqual(fresh.get("oam:scene-10:k10")["id"], "scene-10")
        self.assertEqual(fresh.get("oam:scene-509:k509")["id"], "scene-509")

    def test_cache_tolerates_a_corrupt_file(self):
        cache = SceneCache(self.workspace)
        (self.workspace.traces / CACHE_FILENAME).write_text("{not json", encoding="utf-8")

        self.assertEqual(cache.get("oam:oam-1:abc123"), {})
        cache.remember({"scene_key": "oam:oam-1:abc123", "id": "oam-1"})
        self.assertEqual(cache.get("oam:oam-1:abc123")["id"], "oam-1")


class CatalogPackTests(CatalogTestCase):
    def setUp(self):
        super().setUp()
        self.sink = RecordingSink()
        self.notifications: list[str] = []
        self.map = document.create_map(self.workspace)
        runtime = ToolRuntime(
            workspace=self.workspace,
            map=self.map,
            reporter=Reporter(self.sink, parent_id="cell-7"),
            events=RuntimeEvents(
                files_changed=lambda: self.notifications.append("files"),
                map_changed=lambda: self.notifications.append("map"),
            ),
        )
        self.registry = ToolRegistry()
        self.registry.add_pack(CatalogPack, runtime)

    def call(self, name: str, *args, **kwargs):
        return self.registry.get(name).callable(*args, **kwargs)

    def search_one_scene(self) -> dict:
        with patch(
            "urllib.request.urlopen",
            side_effect=self.serve({oam_meta_url(): oam_body()}),
        ):
            return self.call("search_openaerialmap", [85.0, 27.0, 86.0, 28.0])["scenes"][0]

    def test_registered_catalog_tools_expose_expected_effects(self):
        self.assertEqual(
            sorted(self.registry.names()),
            [
                "add_catalog_scene",
                "download_catalog_scene",
                "list_stac_catalogs",
                "search_openaerialmap",
                "search_stac_collections",
                "search_stac_scenes",
                "search_vantor_events",
                "search_vantor_imagery",
            ],
        )
        self.assertEqual(self.registry.categories()["catalog"], tuple(self.registry.names()))
        self.assertEqual(
            self.registry.get("search_openaerialmap").effects,
            frozenset({Effect.READ, Effect.NETWORK}),
        )
        self.assertTrue(self.registry.replay_safe("search_vantor_events"))
        self.assertFalse(self.registry.replay_safe("add_catalog_scene"))
        self.assertEqual(self.registry.get("download_catalog_scene").kind.value, "reporting")

    def test_add_catalog_scene_requires_a_searched_key(self):
        with self.assertRaises(ToolInputError) as caught:
            self.call("add_catalog_scene", "oam:missing:1234")

        self.assertIn("search again", str(caught.exception))
        with self.assertRaises(ToolInputError):
            self.call("download_catalog_scene", "oam:missing:1234")

    def test_add_catalog_scene_downloads_and_adds_the_layer(self):
        scene = self.search_one_scene()

        with patch(
            "urllib.request.urlopen",
            side_effect=self.serve({OAM_ASSET_URL: b"COG"}),
        ):
            result = self.call("add_catalog_scene", scene["scene_key"])

        self.assertEqual(result["status"], "added")
        self.assertEqual(result["name"], "Flood survey")
        self.assertTrue(result["local_path"].startswith("data/"))
        self.assertTrue(result["local_path"].endswith(".tif"))
        written = self.workspace.root / result["local_path"]
        self.assertEqual(written.read_bytes(), b"COG")
        self.assertEqual([layer["name"] for layer in layers.layers(self.map)], ["Flood survey"])

        added = layers.find_layer(self.map, result["layer_id"])
        catalog = added["metadata"]["geoaiCatalog"]
        self.assertEqual(catalog["scene_key"], scene["scene_key"])
        self.assertEqual(catalog["id"], "oam-1")
        self.assertEqual(catalog["asset_url"], OAM_ASSET_URL)
        self.assertEqual(catalog["local_path"], result["local_path"])

        snapshot = json.loads(
            document.snapshot_path(self.workspace).read_text(encoding="utf-8")
        )
        persisted = [layer for layer in snapshot["layers"] if layer["id"] == result["layer_id"]]
        self.assertEqual(
            persisted[0]["metadata"]["geoaiCatalog"]["local_path"], result["local_path"]
        )
        manifest = json.loads((self.workspace.root / "workspace.json").read_text(encoding="utf-8"))
        self.assertIn(result["local_path"], manifest["outputs"])
        self.assertIn("map", self.notifications)

        jobs = {event["job_id"] for event in self.sink.events}
        self.assertEqual(len(jobs), 1)
        done = self.sink.completed()
        self.assertEqual([event["artifact"] for event in done], [result["local_path"]])
        self.assertEqual(done[0]["kind"], "download")
        self.assertEqual(done[0]["unit"], "bytes")
        self.assertEqual(done[0]["parent_id"], "cell-7")

    def test_repeated_add_returns_the_existing_layer(self):
        scene = self.search_one_scene()

        with patch(
            "urllib.request.urlopen",
            side_effect=self.serve({OAM_ASSET_URL: b"COG"}),
        ):
            first = self.call("add_catalog_scene", scene["scene_key"])
            second = self.call("add_catalog_scene", scene["scene_key"])

        self.assertEqual(second["status"], "existing")
        self.assertEqual(second["layer_id"], first["layer_id"])
        self.assertEqual(second["name"], "Flood survey")
        self.assertEqual(second["local_path"], first["local_path"])
        self.assertEqual(len(layers.layers(self.map)), 1)
        self.assertEqual(len(self.sink.job_ids()), 1)

    def test_download_catalog_scene_streams_the_asset_into_data(self):
        scene = self.search_one_scene()

        with patch(
            "urllib.request.urlopen",
            side_effect=self.serve({OAM_ASSET_URL: b"COG"}),
        ):
            relative = self.call("download_catalog_scene", scene["scene_key"], "oam-1.tif")

        self.assertEqual(relative, "data/oam-1.tif")
        self.assertEqual(self.sink.completed()[0]["artifact"], "data/oam-1.tif")
        manifest = json.loads((self.workspace.root / "workspace.json").read_text(encoding="utf-8"))
        self.assertIn("data/oam-1.tif", manifest["outputs"])


if __name__ == "__main__":
    unittest.main()
