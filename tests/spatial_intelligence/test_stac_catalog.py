"""STAC search: catalogs, collections, scenes, signing, and the pack tools."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.contracts.progress import Reporter
from spatial_intelligence.geo.catalogs import stac
from spatial_intelligence.geo.catalogs.scenes import SceneCache
from spatial_intelligence.map import document, layers
from spatial_intelligence.tools import ToolRegistry, ToolRuntime
from spatial_intelligence.tools.packs.catalog import CatalogPack
from spatial_intelligence.tools.runtime import RuntimeEvents
from spatial_intelligence.workspace import Workspace

EARTH_SEARCH_SEARCH_URL = f"{stac.EARTH_SEARCH_API}/search"
EARTH_SEARCH_COLLECTIONS_URL = (
    f"{stac.EARTH_SEARCH_API}/collections?limit={stac.MAX_COLLECTIONS_PER_PAGE}"
)
PC_SEARCH_URL = f"{stac.PLANETARY_COMPUTER_API}/search"
PC_TOKEN_URL = f"{stac.PLANETARY_COMPUTER_SAS_API}/sentinel-2-l2a"
PC_ASSET_URL = "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x/visual.tif"

#: The STAC scene shape, pinned so a provider cannot quietly add or drop a field
#: the download and map tools read. (The Vantor and OpenAerialMap scenes are
#: pinned by test_catalog_pack.py; this is the superset its extra fields join.)
SCENE_KEYS = [
    "asset",
    "asset_url",
    "bands",
    "bbox",
    "catalog",
    "cloud_cover",
    "collection",
    "datetime",
    "gsd",
    "id",
    "provider",
    "render",
    "scene_key",
    "sensor",
    "signed",
    "thumbnail_url",
    "title",
]


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


class RecordingTransport:
    """A ``urlopen`` replacement that serves canned bodies and records calls."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[dict] = []

    def __call__(self, request, timeout=0):
        url = getattr(request, "full_url", request)
        raw = getattr(request, "data", None)
        self.calls.append({"url": url, "json": json.loads(raw.decode("utf-8")) if raw else None})
        body = self.responses.get(url)
        if body is None:
            raise AssertionError(f"unexpected catalog URL: {url}")
        if isinstance(body, Exception):
            raise body
        if isinstance(body, bytes):
            return FakeResponse(body)
        return FakeResponse(json.dumps(body).encode("utf-8"))

    def requests_to(self, url: str) -> list[dict]:
        return [call for call in self.calls if call["url"] == url]


def search_body(
    items: list[dict],
    *,
    matched: int | None = None,
    next_href: str | None = None,
    next_method: str | None = None,
) -> dict:
    body: dict = {"type": "FeatureCollection", "features": items}
    if matched is not None:
        body["context"] = {"limit": len(items), "matched": matched, "returned": len(items)}
    if next_href:
        link: dict = {"rel": "next", "href": next_href}
        if next_method:
            link["method"] = next_method
        body["links"] = [link]
    return body


def stac_item(
    item_id: str = "S2A_10SEG_20260912_0_L2A",
    *,
    cloud: float | None = 0.5,
    datetime: str = "2026-09-12T19:04:23Z",
    collection: str = "sentinel-2-l2a",
    bbox: list[float] | None = None,
    visual: bool = True,
) -> dict:
    """One Earth Search item: a visual COG, two bands, a JPEG thumbnail."""
    assets = {
        "red": {"href": "https://example.com/red.tif", "type": "image/tiff; profile=cloud-optimized"},
        "nir": {"href": "https://example.com/nir.tif", "type": "image/tiff; profile=cloud-optimized"},
        "thumbnail": {"href": "https://example.com/thumb.jpg", "type": "image/jpeg"},
        "granule_metadata": {"href": "https://example.com/granule.xml", "type": "application/xml"},
    }
    if visual:
        assets["visual"] = {
            "href": "https://example.com/visual.tif",
            "type": "image/tiff; profile=cloud-optimized",
            "gsd": 10,
        }
    properties: dict = {
        "datetime": datetime,
        "platform": "sentinel-2a",
        "constellation": "sentinel-2",
    }
    if cloud is not None:
        properties["eo:cloud_cover"] = cloud
    return {
        "id": item_id,
        "collection": collection,
        "bbox": bbox or [85.0, 27.0, 86.0, 28.0],
        "properties": properties,
        "assets": assets,
    }


def pc_item(item_id: str = "S2A_MSIL2A_20260912T185831") -> dict:
    """One Planetary Computer item: unsigned Azure blob assets."""
    return {
        "id": item_id,
        "collection": "sentinel-2-l2a",
        "bbox": [85.0, 27.0, 86.0, 28.0],
        "properties": {"datetime": "2026-09-12T18:58:31Z", "eo:cloud_cover": 1.5},
        "assets": {
            "visual": {
                "href": PC_ASSET_URL,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            },
            "B04": {
                "href": "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/x/B04.tif",
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            },
        },
    }


def dem_item(
    item_id: str = "Copernicus_DSM_COG_30_N37_00_W123_00_DEM",
    *,
    href: str = "s3://copernicus-dem-90m/tile/DEM.tif",
) -> dict:
    """One Earth Search DEM item: an ``s3://`` asset on a bucket with no bands."""
    return {
        "id": item_id,
        "collection": "cop-dem-glo-90",
        "bbox": [85.0, 27.0, 86.0, 28.0],
        "properties": {"datetime": "2021-04-22T00:00:00Z", "gsd": 90, "platform": "tandem-x"},
        "assets": {
            "data": {
                "href": href,
                "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            }
        },
    }


class StacTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.cache = SceneCache(self.workspace)

    def tearDown(self):
        self._tmp.cleanup()


class CatalogRegistryTests(StacTestCase):
    def test_built_in_catalogs_carry_their_api_and_signing_flag(self):
        catalogs = {entry["catalog"]: entry for entry in stac.list_catalogs()}

        self.assertEqual(sorted(catalogs), ["earth-search", "planetary-computer"])
        self.assertEqual(catalogs["earth-search"]["api"], stac.EARTH_SEARCH_API)
        self.assertFalse(catalogs["earth-search"]["signed"])
        self.assertTrue(catalogs["planetary-computer"]["signed"])
        self.assertIn("sentinel-2-l2a", catalogs["earth-search"]["common_collections"])

    def test_a_stac_api_url_resolves_and_is_signed_only_on_planetary_computer(self):
        default = stac.resolve_catalog("")
        own = stac.resolve_catalog("https://example.com/stac/")
        pc = stac.resolve_catalog("https://planetarycomputer.microsoft.com/api/stac/v1")

        self.assertEqual(default.key, stac.DEFAULT_CATALOG)
        self.assertEqual(own.api, "https://example.com/stac")
        self.assertFalse(own.signed)
        self.assertTrue(pc.signed)

    def test_an_unknown_catalog_is_a_tool_input_error(self):
        with self.assertRaises(ToolInputError) as caught:
            stac.resolve_catalog("not-a-catalog")

        self.assertIn("unknown STAC catalog", str(caught.exception))


class CollectionListingTests(StacTestCase):
    def collections_body(self) -> dict:
        return {
            "collections": [
                {"id": "sentinel-2-l2a", "title": "Sentinel-2 Level-2A", "keywords": ["sentinel"]},
                {"id": "landsat-c2-l2", "title": "Landsat Collection 2 Level-2", "keywords": ["landsat"]},
                {
                    "id": "io-lulc-annual-v02",
                    "title": "Annual Land Cover",
                    "description": "Global land cover, 10 m.",
                    "keywords": ["land", "cover"],
                },
            ],
            "links": [],
        }

    def test_a_query_ranks_collections_by_matched_terms(self):
        transport = RecordingTransport({EARTH_SEARCH_COLLECTIONS_URL: self.collections_body()})

        with patch("urllib.request.urlopen", side_effect=transport):
            found = stac.list_collections(stac.EARTH_SEARCH_API, "annual land cover")

        self.assertEqual([entry["id"] for entry in found], ["io-lulc-annual-v02"])
        self.assertEqual(found[0]["title"], "Annual Land Cover")
        self.assertEqual(found[0]["keywords"], ["land", "cover"])

    def test_no_query_lists_collections_by_id_under_the_limit(self):
        transport = RecordingTransport({EARTH_SEARCH_COLLECTIONS_URL: self.collections_body()})

        with patch("urllib.request.urlopen", side_effect=transport):
            found = stac.list_collections(stac.EARTH_SEARCH_API, limit=2)

        self.assertEqual([entry["id"] for entry in found], ["io-lulc-annual-v02", "landsat-c2-l2"])

    def test_listing_follows_a_next_page(self):
        first = self.collections_body()
        first["links"] = [{"rel": "next", "href": "https://example.com/page2"}]
        second = {"collections": [{"id": "naip", "title": "NAIP"}], "links": []}
        transport = RecordingTransport(
            {EARTH_SEARCH_COLLECTIONS_URL: first, "https://example.com/page2": second}
        )

        with patch("urllib.request.urlopen", side_effect=transport):
            found = stac.list_collections(stac.EARTH_SEARCH_API)

        self.assertEqual([entry["id"] for entry in found], ["io-lulc-annual-v02", "landsat-c2-l2", "naip", "sentinel-2-l2a"])


class SceneSearchTests(StacTestCase):
    def search(self, body: dict, **kwargs) -> tuple[dict, RecordingTransport]:
        transport = RecordingTransport({EARTH_SEARCH_SEARCH_URL: body})
        with patch("urllib.request.urlopen", side_effect=transport):
            result = stac.search_scenes(
                stac.resolve_catalog("earth-search"),
                collections=["sentinel-2-l2a"],
                cache=self.cache,
                **kwargs,
            )
        return result, transport

    def test_a_scene_is_normalized_cached_and_reported_with_the_matched_count(self):
        body = search_body([stac_item()], matched=4)

        result, _ = self.search(body, bounds=[85.0, 27.0, 86.0, 28.0], limit=20)
        scene = result["scenes"][0]

        self.assertEqual(result["catalog"], "earth-search")
        self.assertEqual(result["matched"], 4)
        self.assertEqual(sorted(scene), SCENE_KEYS)
        self.assertEqual(scene["id"], "S2A_10SEG_20260912_0_L2A")
        self.assertEqual(scene["provider"], "Earth Search")
        self.assertEqual(scene["catalog"], "earth-search")
        self.assertEqual(scene["collection"], "sentinel-2-l2a")
        self.assertEqual(scene["sensor"], "sentinel-2a")
        self.assertEqual(scene["cloud_cover"], 0.5)
        self.assertEqual(scene["gsd"], 10)
        self.assertEqual(scene["asset"], "visual")
        self.assertEqual(scene["asset_url"], "https://example.com/visual.tif")
        self.assertEqual(scene["thumbnail_url"], "https://example.com/thumb.jpg")
        self.assertEqual(scene["render"], "cog")
        self.assertEqual(sorted(scene["bands"]), ["nir", "red", "visual"])
        self.assertTrue(scene["scene_key"].startswith("stac:S2A_10SEG_20260912_0_L2A:"))
        self.assertFalse(scene["signed"])
        self.assertEqual(self.cache.get(scene["scene_key"])["asset_url"], scene["asset_url"])

    def test_the_search_body_carries_a_full_timestamp_interval_and_no_cloud_filter(self):
        # A ``query`` cloud filter silently matches nothing on a collection
        # without the property, so clouds are filtered here instead.
        _, transport = self.search(
            search_body([stac_item()]),
            bounds=[85.0, 27.0, 86.0, 28.0],
            start="2026-08-26",
            end="2026-09-25",
            cloud_max=20,
        )

        payload = transport.requests_to(EARTH_SEARCH_SEARCH_URL)[0]["json"]
        self.assertEqual(payload["collections"], ["sentinel-2-l2a"])
        self.assertEqual(payload["bbox"], [85.0, 27.0, 86.0, 28.0])
        self.assertEqual(payload["datetime"], "2026-08-26T00:00:00Z/2026-09-25T23:59:59Z")
        self.assertNotIn("query", payload)

    def test_cloud_sort_filters_and_keeps_scenes_with_no_cloud_property(self):
        body = search_body(
            [
                stac_item("cloudy", cloud=60.0, datetime="2026-09-20T00:00:00Z"),
                stac_item("clear", cloud=2.0, datetime="2026-09-10T00:00:00Z"),
                stac_item("unknown", cloud=None, datetime="2026-09-15T00:00:00Z"),
            ]
        )

        unfiltered, _ = self.search(body, sort="cloud")
        filtered, _ = self.search(body, sort="cloud", cloud_max=10)

        # Clearest first; a scene with no cloud property cannot be vouched for,
        # so it follows the ones that can, while cloud_max keeps it as a
        # candidate rather than dropping it.
        self.assertEqual([scene["id"] for scene in unfiltered["scenes"]], ["clear", "cloudy", "unknown"])
        self.assertEqual([scene["id"] for scene in filtered["scenes"]], ["clear", "unknown"])

    def test_date_sort_is_newest_first(self):
        body = search_body(
            [
                stac_item("older", datetime="2026-08-30T00:00:00Z"),
                stac_item("newer", datetime="2026-09-20T00:00:00Z"),
            ]
        )

        result, _ = self.search(body)

        self.assertEqual([scene["id"] for scene in result["scenes"]], ["newer", "older"])

    def test_scenes_without_a_geotiff_asset_and_outside_the_aoi_are_dropped(self):
        body = search_body(
            [
                stac_item("no-visual", visual=False, bbox=[0.0, 0.0, 1.0, 1.0]),
                stac_item("with-visual"),
            ]
        )

        result, _ = self.search(body, bounds=[85.0, 27.0, 86.0, 28.0])

        self.assertEqual([scene["id"] for scene in result["scenes"]], ["with-visual"])
        # An item outside the area is not "unusable": nothing about it was wrong,
        # it simply was not asked for.
        self.assertEqual(result["unusable"], 0)

    def test_an_item_with_no_readable_raster_inside_the_area_is_counted(self):
        # Only a thumbnail and metadata: the shape a point-cloud or vector
        # collection publishes, which builds no scene at all (``visual=False``
        # still leaves the band GeoTIFFs, so it is a different case).
        bare = stac_item("no-raster")
        bare["assets"] = {
            "thumbnail": {"href": "https://example.com/thumb.jpg", "type": "image/jpeg"},
            "granule_metadata": {"href": "https://example.com/granule.xml", "type": "application/xml"},
        }
        body = search_body([bare, stac_item("with-visual")])

        result, _ = self.search(body, bounds=[85.0, 27.0, 86.0, 28.0])

        self.assertEqual([scene["id"] for scene in result["scenes"]], ["with-visual"])
        # The model can tell "nothing published" from "nothing readable here".
        self.assertEqual(result["unusable"], 1)

    def test_an_item_page_is_followed_through_its_next_link(self):
        # Both built-in providers hand back a POST link for a POST search, so that
        # is the branch that matters; the GET form is covered next to it.
        first = search_body(
            [stac_item("page-1")],
            next_href="https://example.com/page2",
            next_method="POST",
        )
        transport = RecordingTransport(
            {
                EARTH_SEARCH_SEARCH_URL: first,
                "https://example.com/page2": search_body([stac_item("page-2")]),
            }
        )

        with patch("urllib.request.urlopen", side_effect=transport):
            result = stac.search_scenes(
                stac.resolve_catalog("earth-search"),
                collections=["sentinel-2-l2a"],
                cache=self.cache,
            )

        self.assertEqual([scene["id"] for scene in result["scenes"]], ["page-1", "page-2"])
        self.assertEqual(len(transport.requests_to("https://example.com/page2")), 1)

    def test_a_get_next_link_is_followed_without_a_body(self):
        first = search_body([stac_item("page-1")], next_href="https://example.com/page2")
        transport = RecordingTransport(
            {
                EARTH_SEARCH_SEARCH_URL: first,
                "https://example.com/page2": search_body([stac_item("page-2")]),
            }
        )

        with patch("urllib.request.urlopen", side_effect=transport):
            result = stac.search_scenes(
                stac.resolve_catalog("earth-search"),
                collections=["sentinel-2-l2a"],
                cache=self.cache,
            )

        self.assertEqual([scene["id"] for scene in result["scenes"]], ["page-1", "page-2"])

    def test_the_same_item_is_not_reported_twice(self):
        body = search_body([stac_item("dup"), stac_item("dup")])

        result, _ = self.search(body)

        self.assertEqual(len(result["scenes"]), 1)

    def test_an_s3_asset_is_read_over_its_public_https_address(self):
        # Earth Search publishes most collections as s3:// URIs; the public ones
        # are readable at the bucket's virtual-host HTTPS address.
        result, _ = self.search(search_body([dem_item()]))
        scene = result["scenes"][0]

        self.assertEqual(scene["collection"], "cop-dem-glo-90")
        self.assertEqual(scene["asset"], "data")
        self.assertEqual(scene["gsd"], 90)
        self.assertEqual(
            scene["asset_url"], "https://copernicus-dem-90m.s3.amazonaws.com/tile/DEM.tif"
        )
        self.assertEqual(scene["bands"], {"data": scene["asset_url"]})

    def test_the_named_display_asset_wins_over_a_translated_one(self):
        item = dem_item()
        item["assets"]["red"] = {
            "href": "https://example.com/red.tif",
            "type": "image/tiff; profile=cloud-optimized",
        }

        result, _ = self.search(search_body([item]))

        # "data" is a display name (it is what a DEM publishes), so it is chosen
        # ahead of any other readable band, translated or not.
        self.assertEqual(result["scenes"][0]["asset"], "data")

    def test_a_direct_https_asset_breaks_a_tie_between_unnamed_bands(self):
        item = dem_item()
        item["assets"] = {
            "alpha": {
                "href": "s3://bucket/alpha.tif",
                "type": "image/tiff; profile=cloud-optimized",
            },
            "beta": {
                "href": "https://example.com/beta.tif",
                "type": "image/tiff; profile=cloud-optimized",
            },
        }

        result, _ = self.search(search_body([item]))
        scene = result["scenes"][0]

        self.assertEqual(scene["asset"], "beta")
        self.assertEqual(list(scene["bands"]), ["beta", "alpha"])

    def test_an_asset_that_cannot_be_read_anonymously_is_dropped(self):
        body = search_body([dem_item(href="gs://bucket/tile/DEM.tif")])

        result, _ = self.search(body)

        self.assertEqual(result["scenes"], [])

    def test_invalid_arguments_are_rejected(self):
        cases = [
            {"collections": []},
            {"collections": ["sentinel-2-l2a"], "bounds": [86.0, 27.0, 85.0, 28.0]},
            {"collections": ["sentinel-2-l2a"], "bounds": [85.0, 27.0, 86.0]},
            {"collections": ["sentinel-2-l2a"], "start": "26-08-2026"},
            {"collections": ["sentinel-2-l2a"], "start": "2026-09-25", "end": "2026-08-26"},
            {"collections": ["sentinel-2-l2a"], "cloud_max": 150},
            {"collections": ["sentinel-2-l2a"], "sort": "cloudiest"},
        ]

        for kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(ToolInputError):
                    stac.search_scenes(
                        stac.resolve_catalog("earth-search"), cache=self.cache, **kwargs
                    )


class PlanetaryComputerSigningTests(StacTestCase):
    def test_assets_are_signed_once_per_collection(self):
        transport = RecordingTransport(
            {
                PC_SEARCH_URL: search_body([pc_item("item-1"), pc_item("item-2")]),
                PC_TOKEN_URL: {"token": "se=2026&sig=abc", "msft:expiry": "2026-09-25T21:00:00Z"},
            }
        )
        sas = stac.SasTokenCache()

        with patch("urllib.request.urlopen", side_effect=transport):
            result = stac.search_scenes(
                stac.resolve_catalog("planetary-computer"),
                collections=["sentinel-2-l2a"],
                cache=self.cache,
                sas=sas,
            )

        self.assertEqual(len(transport.requests_to(PC_TOKEN_URL)), 1)
        scene = result["scenes"][0]
        self.assertEqual(scene["provider"], "Planetary Computer")
        self.assertEqual(scene["asset_url"], f"{PC_ASSET_URL}?se=2026&sig=abc")
        self.assertTrue(scene["bands"]["B04"].endswith("?se=2026&sig=abc"))
        self.assertTrue(scene["bands"]["visual"].endswith("?se=2026&sig=abc"))

    def test_the_cached_scene_keeps_unsigned_hrefs_and_signing_happens_on_use(self):
        transport = RecordingTransport(
            {
                PC_SEARCH_URL: search_body([pc_item("item-1")]),
                PC_TOKEN_URL: {"token": "se=2026&sig=abc", "msft:expiry": "2026-09-25T21:00:00Z"},
            }
        )
        sas = stac.SasTokenCache()

        with patch("urllib.request.urlopen", side_effect=transport):
            result = stac.search_scenes(
                stac.resolve_catalog("planetary-computer"),
                collections=["sentinel-2-l2a"],
                cache=self.cache,
                sas=sas,
            )
            # Presented signed, …but the cached scene, which is what reaches disk
            # and what the scene key is derived from, holds the catalog's own href.
            cached = self.cache.get(result["scenes"][0]["scene_key"])
            self.assertEqual(cached["asset_url"], PC_ASSET_URL)
            self.assertEqual(cached["bands"]["B04"], f"{PC_ASSET_URL.rsplit('/', 1)[0]}/B04.tif")
            self.assertTrue(cached["signed"])
            self.assertNotIn("sig=", json.dumps(cached))
            # Signing at use is what survives the token's expiry.
            self.assertEqual(
                stac.asset_href(cached, sas=sas), f"{PC_ASSET_URL}?se=2026&sig=abc"
            )

        self.assertEqual(len(transport.requests_to(PC_TOKEN_URL)), 1)

    def test_a_collection_without_a_token_is_a_tool_input_error(self):
        transport = RecordingTransport(
            {PC_SEARCH_URL: search_body([pc_item()]), PC_TOKEN_URL: {}}
        )
        with patch("urllib.request.urlopen", side_effect=transport):
            with self.assertRaises(ToolInputError) as caught:
                stac.search_scenes(
                    stac.resolve_catalog("planetary-computer"),
                    collections=["sentinel-2-l2a"],
                    cache=self.cache,
                    sas=stac.SasTokenCache(),
                )

        self.assertIn("sentinel-2-l2a", str(caught.exception))

    def test_a_transport_failure_while_signing_is_reported_with_the_collection(self):
        transport = RecordingTransport(
            {PC_SEARCH_URL: search_body([pc_item()]), PC_TOKEN_URL: OSError("connection reset")}
        )

        with patch("urllib.request.urlopen", side_effect=transport):
            with self.assertRaises(ToolInputError) as caught:
                stac.search_scenes(
                    stac.resolve_catalog("planetary-computer"),
                    collections=["sentinel-2-l2a"],
                    cache=self.cache,
                    sas=stac.SasTokenCache(),
                )

        self.assertIn("could not get a signing token", str(caught.exception))


class CatalogPackStacTests(StacTestCase):
    def setUp(self):
        super().setUp()
        self.map = document.create_map(self.workspace)
        self.notifications: list[str] = []
        runtime = ToolRuntime(
            workspace=self.workspace,
            map=self.map,
            reporter=Reporter(_Sink(), parent_id="cell-3"),
            events=RuntimeEvents(
                files_changed=lambda: self.notifications.append("files"),
                map_changed=lambda: self.notifications.append("map"),
            ),
        )
        self.registry = ToolRegistry()
        self.registry.add_pack(CatalogPack, runtime)

    def call(self, name: str, *args, **kwargs):
        return self.registry.get(name).callable(*args, **kwargs)

    def test_the_stac_search_tools_are_read_only_and_network_like_the_others(self):
        # The registry's name list is pinned by test_catalog_pack.py; here the
        # point is what the STAC tools may touch.
        self.assertEqual(
            self.registry.get("search_stac_scenes").effects,
            frozenset({Effect.READ, Effect.NETWORK}),
        )
        self.assertEqual(self.registry.get("list_stac_catalogs").effects, frozenset({Effect.READ}))
        self.assertTrue(self.registry.replay_safe("search_stac_scenes"))

    def test_a_searched_scene_downloads_into_data_and_becomes_a_layer(self):
        transport = RecordingTransport(
            {EARTH_SEARCH_SEARCH_URL: search_body([stac_item()]), "https://example.com/visual.tif": b"COG"}
        )

        with patch("urllib.request.urlopen", side_effect=transport):
            found = self.call(
                "search_stac_scenes",
                ["sentinel-2-l2a"],
                "earth-search",
                [85.0, 27.0, 86.0, 28.0],
                "2026-08-26",
                "2026-09-25",
                20.0,
                "cloud",
            )
            added = self.call(
                "add_catalog_scene", found["scenes"][0]["scene_key"], "S2 over Nepal"
            )

        self.assertEqual(added["status"], "added")
        # The filename carries the provider and item id, so a workspace holding
        # scenes from several catalogs stays readable.
        self.assertTrue(
            added["local_path"].startswith("data/earth-search-S2A_10SEG_20260912_0_L2A-")
        )
        self.assertTrue(added["local_path"].endswith(".tif"))
        self.assertEqual(added["name"], "S2 over Nepal")
        self.assertEqual(len(layers.layers(self.map)), 1)

        stored = layers.find_layer(self.map, added["layer_id"])["metadata"]["geoaiCatalog"]
        self.assertEqual(stored["catalog"], "earth-search")
        self.assertEqual(stored["collection"], "sentinel-2-l2a")
        self.assertEqual(stored["asset"], "visual")
        self.assertEqual(stored["local_path"], added["local_path"])
        self.assertEqual(
            (self.workspace.root / added["local_path"]).read_bytes(), b"COG"
        )
        self.assertIn("map", self.notifications)

    def test_the_catalog_and_collection_tools_answer_without_a_map(self):
        transport = RecordingTransport(
            {
                EARTH_SEARCH_COLLECTIONS_URL: {
                    "collections": [{"id": "sentinel-2-l2a", "title": "Sentinel-2"}],
                    "links": [],
                }
            }
        )

        catalogs = self.call("list_stac_catalogs")
        with patch("urllib.request.urlopen", side_effect=transport):
            collections = self.call("search_stac_collections", "sentinel", "earth-search")

        self.assertEqual([entry["catalog"] for entry in catalogs], ["earth-search", "planetary-computer"])
        self.assertEqual([entry["id"] for entry in collections], ["sentinel-2-l2a"])


class _Sink:
    """A progress sink that keeps nothing; these tests assert on the map."""

    def emit(self, event):
        return None


if __name__ == "__main__":
    unittest.main()
