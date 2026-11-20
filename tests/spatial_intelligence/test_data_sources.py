"""Data-source contract, registry, adapters, and generic tools."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_intelligence.contracts.services import DATA_SOURCES_SERVICE
from spatial_intelligence.data_sources import (
    Asset,
    DataQuery,
    DataSourceRegistry,
    Dataset,
    OpenAerialMapSource,
    STACSource,
    USGSSource,
)
from spatial_intelligence.data_sources.providers.arcgis import ArcGISSource
from spatial_intelligence.data_sources.providers.gibs import GIBSSource, reset_cache
from spatial_intelligence.map import document, layers
from spatial_intelligence.tools import ToolRegistry, ToolRuntime
from spatial_intelligence.tools.packs.data import DataPack
from spatial_intelligence.tools.runtime import RuntimeEvents
from spatial_intelligence.workspace import Workspace


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


def open_url(_request, timeout=0):
    return FakeResponse(b"COG")


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()

    def tearDown(self):
        self._tmp.cleanup()

    def runtime(self, registry: DataSourceRegistry) -> ToolRuntime:
        return ToolRuntime(
            workspace=self.workspace,
            services={DATA_SOURCES_SERVICE: registry},
            events=RuntimeEvents(),
        )

    def registry(self, *sources) -> DataSourceRegistry:
        return DataSourceRegistry(list(sources))


class RegistryTests(WorkspaceCase):
    def test_a_disabled_source_is_hidden_and_rejected(self):
        registry = self.registry(OpenAerialMapSource(), STACSource())
        registry.configure({"disabled_sources": ["stac"]})

        self.assertEqual([s.descriptor.id for s in registry.enabled()], ["openaerialmap"])
        with self.assertRaisesRegex(Exception, "disabled"):
            registry.get("stac", enabled_only=True)

    def test_payload_marks_enabled_and_available(self):
        registry = self.registry(OpenAerialMapSource())
        row = registry.payload()[0]

        self.assertTrue(row["enabled"])
        self.assertTrue(row["available"])
        self.assertIn("search", row["capabilities"])


class STACAdapterTests(WorkspaceCase):
    def test_empty_filter_lists_collections_and_text_narrows_them(self):
        collections = {
            "collections": [
                {"id": "sentinel-2-l2a", "title": "Sentinel-2 L2A"},
                {"id": "landsat-c2-l2", "title": "Landsat Collection 2"},
            ],
            "links": [],
        }
        with patch(
            "spatial_intelligence.geo.catalogs.stac.fetch_json", return_value=collections
        ):
            page = STACSource().search(DataQuery(text="sentinel"))

        self.assertEqual([row.id.split(":")[-1] for row in page.datasets], ["sentinel-2-l2a"])
        self.assertEqual(page.datasets[0].kind, "collection")

    def test_scene_search_uses_optional_text_and_normalizes_assets(self):
        item = {
            "id": "item-1",
            "bbox": [85.0, 27.0, 86.0, 28.0],
            "collection": "sentinel-2-l2a",
            "properties": {
                "datetime": "2026-01-01T00:00:00Z",
                "eo:cloud_cover": 5,
                "platform": "sentinel-2a",
            },
            "assets": {
                "visual": {"href": "https://example.com/visual.tif", "type": "image/tiff"},
                "B08": {"href": "https://example.com/B08.tif", "type": "image/tiff"},
            },
        }
        with patch(
            "spatial_intelligence.geo.catalogs.stac.post_json",
            return_value={"features": [item], "links": [], "context": {"matched": 1}},
        ):
            page = STACSource().search(
                DataQuery(text="", filters={"collections": ["sentinel-2-l2a"]})
            )

        dataset = page.datasets[0]
        self.assertEqual(dataset.id, "earth-search:item-1")
        self.assertEqual(dataset.kind, "raster")
        self.assertEqual(dataset.bbox, (85.0, 27.0, 86.0, 28.0))
        self.assertEqual(dataset.assets[0].id, "visual")
        self.assertEqual(len(dataset.assets), 2)
        self.assertEqual(page.total, 1)


class USGSAdapterTests(WorkspaceCase):
    def test_search_normalizes_a_tnm_product(self):
        body = {
            "total": 1,
            "items": [
                {
                    "title": "USGS 1 Meter 15 x78y326 Test",
                    "sourceId": "dem-1",
                    "downloadURL": "https://example.com/dem.tif",
                    "format": "GeoTIFF",
                    "sizeInBytes": 123,
                    "publicationDate": "2024-01-01",
                    "boundingBox": {
                        "minX": -90.0,
                        "minY": 29.0,
                        "maxX": -89.9,
                        "maxY": 29.1,
                    },
                }
            ],
        }
        with patch(
            "spatial_intelligence.data_sources.providers.usgs.fetch_json",
            return_value=body,
        ):
            page = USGSSource().search(DataQuery(bbox=(-90.0, 29.0, -89.0, 30.0)))

        dataset = page.datasets[0]
        self.assertEqual(dataset.id, "dem-1")
        self.assertEqual(dataset.kind, "raster")
        self.assertEqual(dataset.assets[0].href, "https://example.com/dem.tif")
        self.assertEqual(page.total, 1)


class GIBSAdapterTests(WorkspaceCase):
    XML = """<Capabilities xmlns="http://www.opengis.net/wmts/1.0"
        xmlns:ows="http://www.opengis.net/ows/1.1">
      <Contents>
        <Layer>
          <ows:Title>Test True Color</ows:Title>
          <ows:Identifier>TEST_TRUE_COLOR</ows:Identifier>
          <ows:WGS84BoundingBox>
            <ows:LowerCorner>-180 -85</ows:LowerCorner>
            <ows:UpperCorner>180 85</ows:UpperCorner>
          </ows:WGS84BoundingBox>
          <TileMatrixSetLink><TileMatrixSet>GoogleMapsCompatible_Level9</TileMatrixSet></TileMatrixSetLink>
          <ResourceURL resourceType="tile" format="image/jpeg"
            template="https://gibs.example/TEST/{Time}/{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.jpg"/>
        </Layer>
      </Contents>
    </Capabilities>"""

    def setUp(self):
        super().setUp()
        reset_cache()

    def tearDown(self):
        reset_cache()
        super().tearDown()

    def test_layer_search_and_date_templating(self):
        with patch(
            "spatial_intelligence.data_sources.providers.gibs.fetch_text",
            return_value=self.XML,
        ):
            source = GIBSSource()
            page = source.search(DataQuery(start="2026-01-02"))
            spec = source.materialize(page.datasets[0])[0]

        self.assertEqual(page.datasets[0].id, "TEST_TRUE_COLOR")
        self.assertEqual(page.datasets[0].kind, "tiles")
        self.assertEqual(spec.kind, "wmts")
        self.assertIn("/2026-01-02/", spec.uri)
        self.assertIn("{z}/{y}/{x}", spec.uri)


class ArcGISAdapterTests(WorkspaceCase):
    def test_search_resolves_a_feature_layer(self):
        item = {
            "id": "item-1",
            "type": "Feature Service",
            "title": "Test Parcels",
            "url": "https://services.example/FeatureServer",
            "extent": [[-90.0, 29.0], [-89.0, 30.0]],
            "licenseInfo": "Public domain",
        }
        service = {"layers": [{"id": 0, "name": "Parcels"}]}

        def fake(url, **kwargs):
            if "sharing/rest/search" in url:
                return {"total": 1, "results": [item]}
            if url.endswith("?f=json"):
                return service
            raise AssertionError(f"unexpected ArcGIS URL: {url}")

        with patch(
            "spatial_intelligence.data_sources.providers.arcgis.fetch_json",
            side_effect=fake,
        ):
            page = ArcGISSource().search(DataQuery(text="parcels"))
            spec = ArcGISSource().materialize(page.datasets[0])[0]

        dataset = page.datasets[0]
        self.assertEqual(dataset.id, "item-1:0")
        self.assertEqual(dataset.kind, "vector")
        self.assertIn("/0/query", dataset.assets[0].href)
        self.assertEqual(spec.kind, "vector")
        self.assertIn("arcgis_feature", spec.options)

    def test_search_terms_are_optional(self):
        def fake(url, **kwargs):
            return {"total": 0, "results": []}

        with patch(
            "spatial_intelligence.data_sources.providers.arcgis.fetch_json",
            side_effect=fake,
        ):
            page = ArcGISSource().search(DataQuery())

        self.assertEqual(page.datasets, ())


class DataPackTests(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.map = document.create_map(self.workspace)

    def pack(self, *sources):
        registry = self.registry(*sources)
        runtime = self.runtime(registry)
        runtime.map = self.map
        tools = ToolRegistry()
        tools.add_pack(DataPack, runtime)
        return tools

    def test_list_and_describe_use_the_registry_and_optional_text(self):
        tools = self.pack(OpenAerialMapSource(), STACSource())

        listed = tools.get("list_data_sources").callable()
        filtered = tools.get("list_data_sources").callable(query="aerial")
        described = tools.get("describe_data_source").callable("openaerialmap")

        self.assertEqual(len(listed), 2)
        self.assertEqual([row["id"] for row in filtered], ["openaerialmap"])
        self.assertEqual(described["id"], "openaerialmap")
        self.assertIn("search", described["capabilities"])

    def test_search_and_add_open_aerial_map(self):
        meta = {
            "meta": {"found": "1"},
            "results": [
                {
                    "_id": "oam-1",
                    "uuid": "https://example.com/aerial.tif",
                    "title": "Flood survey",
                    "provider": "Example",
                    "bbox": [85.0, 27.0, 86.0, 28.0],
                    "acquisition_end": "2026-01-01T00:00:00Z",
                    "properties": {"thumbnail": "https://example.com/thumb.png"},
                }
            ],
        }
        tools = self.pack(OpenAerialMapSource())
        with patch(
            "spatial_intelligence.geo.catalogs.openaerialmap.fetch_json",
            return_value=meta,
        ):
            page = tools.get("search_data").callable(
                "openaerialmap", bbox=[85.0, 27.0, 86.0, 28.0]
            )

        dataset = page["datasets"][0]
        self.assertEqual(dataset["id"], "oam-1")
        self.assertEqual(dataset["key"], "openaerialmap:oam-1")

        with patch("urllib.request.urlopen", side_effect=open_url):
            added = tools.get("add_data").callable("openaerialmap", "oam-1")

        self.assertEqual(added["status"], "added")
        self.assertEqual(len(added["layers"]), 1)
        layer = layers.find_layer(self.map, added["layers"][0]["layer_id"])
        provenance = layer["metadata"]["geoaiDataSource"]
        self.assertEqual(provenance["source_id"], "openaerialmap")
        self.assertEqual(provenance["dataset_id"], "oam-1")
        self.assertTrue(provenance["local_path"].startswith("data/"))
        self.assertTrue((self.workspace.root / provenance["local_path"]).exists())

    def test_arcgis_feature_download_pages_before_mapping(self):
        from spatial_intelligence.data_sources.cache import DatasetCache

        dataset = Dataset(
            source_id="arcgis",
            id="item-1:0",
            title="Test parcels",
            kind="vector",
            bbox=(-90.0, 29.0, -89.0, 30.0),
            properties={
                "service_type": "Feature Service",
                "service_url": "https://services.example/FeatureServer",
                "layer_id": 0,
                "query_url": "https://services.example/FeatureServer/0/query?f=geojson",
            },
            assets=(
                Asset(
                    id="data",
                    href="https://services.example/FeatureServer/0/query?f=geojson",
                    media_type="application/geo+json",
                ),
            ),
        )
        DatasetCache(self.workspace).remember(dataset)
        tools = self.pack(ArcGISSource())
        pages = [
            {
                "features": [{"type": "Feature", "properties": {"id": 1}, "geometry": None}],
                "exceededTransferLimit": True,
            },
            {
                "features": [{"type": "Feature", "properties": {"id": 2}, "geometry": None}],
                "exceededTransferLimit": False,
            },
        ]
        with patch(
            "spatial_intelligence.data_sources.service.fetch_json",
            side_effect=pages,
        ):
            added = tools.get("add_data").callable("arcgis", "item-1:0")

        provenance = layers.find_layer(
            self.map, added["layers"][0]["layer_id"]
        )["metadata"]["geoaiDataSource"]
        written = json.loads(
            (self.workspace.root / provenance["local_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual([feature["properties"]["id"] for feature in written["features"]], [1, 2])

    def test_disabled_source_is_not_listed_and_cannot_search(self):
        registry = self.registry(OpenAerialMapSource())
        registry.configure({"disabled_sources": ["openaerialmap"]})
        runtime = self.runtime(registry)
        runtime.map = self.map
        tools = ToolRegistry()
        tools.add_pack(DataPack, runtime)

        self.assertEqual(tools.get("list_data_sources").callable(), [])
        with self.assertRaisesRegex(Exception, "disabled"):
            tools.get("search_data").callable(
                "openaerialmap", bbox=[85.0, 27.0, 86.0, 28.0]
            )

    def test_dataset_cache_survives_a_pack_rebuild(self):
        from spatial_intelligence.data_sources.cache import DatasetCache

        dataset = Dataset(
            source_id="openaerialmap",
            id="cached-1",
            title="Cached",
            kind="raster",
            bbox=(85.0, 27.0, 86.0, 28.0),
            assets=(),
        )
        DatasetCache(self.workspace).remember(dataset)
        restored = DatasetCache(self.workspace).get("openaerialmap:cached-1")

        self.assertIsNotNone(restored)
        self.assertEqual(restored.title, "Cached")


if __name__ == "__main__":
    unittest.main()
