"""Drone import detection and map insertion.

The detector is exercised against real file headers: a JPEG with EXIF GPS, a
TIFF with only DJI XMP (the format Orthority's ODM test images use), a
georeferenced GeoTIFF, and a plain image that must be skipped. The map test uses
a real headless ``geolibre.Map`` so the persisted project is the observable
contract.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from starlette.testclient import TestClient

from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.geo import drone, orthorectify
from spatial_intelligence.map import document
from spatial_intelligence.server import deps
from spatial_intelligence.server.app import create_app
from spatial_intelligence.workspace import Workspace

SAMPLE_CAPTURE = Path(__file__).resolve().parents[2] / "examples" / "data" / "dji-orthority"


def _dms(value: float) -> tuple[int, int, float]:
    """Split decimal degrees into the degrees/minutes/seconds EXIF uses."""
    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    return degrees, minutes, (minutes_float - minutes) * 60


def _jpeg(path: Path, *, latitude=24.680278, longitude=120.951702) -> None:
    """Write a small JPEG with a DJI make/model and an EXIF GPS IFD."""
    exif = Image.Exif()
    exif[271] = "DJI"
    exif[272] = "FC6310R"
    exif[36867] = "2026:05:12 10:00:00"
    exif[34853] = {
        1: "N",
        2: _dms(latitude),
        3: "E",
        4: _dms(longitude),
        5: 0,
        6: 100.0,
        17: 92.5,
    }
    Image.new("RGB", (12, 12), (20, 40, 60)).save(path, exif=exif)


def _xmp_tiff(path: Path, *, latitude=24.680278, longitude=120.951702) -> None:
    """Write a TIFF whose GPS and camera are only in the DJI XMP packet."""
    xmp = (
        '<?xpacket begin=""?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description '
        'xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
        'xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
        f'drone-dji:GpsLatitude="{latitude}" '
        f'drone-dji:GpsLongtitude="{longitude}" '
        'drone-dji:GimbalYawDegree="+92.90" '
        'drone-dji:AbsoluteAltitude="+186.57" '
        'tiff:Make="DJI" tiff:Model="FC6310R"/>'
        "</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    )
    Image.new("RGB", (12, 12), (30, 50, 70)).save(
        path, tiffinfo={700: xmp.encode("utf-8")}
    )


def _raster(path: Path) -> None:
    """Write a small UTM GeoTIFF."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=6,
        count=3,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(500_000, 4_000_000, 1, 1),
    ) as dataset:
        dataset.write(numpy.full((3, 6, 8), 120, dtype="uint8"))


class DroneDetectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        flight = self.workspace.data / "flight"
        flight.mkdir(parents=True)
        _jpeg(flight / "IMG_0001.JPG")
        _xmp_tiff(flight / "IMG_0002.TIF")
        _raster(self.workspace.data / "ortho.tif")
        Image.new("RGB", (8, 8), (1, 2, 3)).save(flight / "plain.png")

    def test_detects_photos_and_georeferenced_rasters_and_skips_plain_images(self):
        result = drone.detect_files(self.workspace, ["data"])

        self.assertEqual(result["photos"], 2)
        self.assertEqual(result["rasters"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(len(result["bounds"]), 4)

        photos = [item for item in result["files"] if item["kind"] == "photo"]
        self.assertEqual({item["path"] for item in photos}, {
            "data/flight/IMG_0001.JPG",
            "data/flight/IMG_0002.TIF",
        })
        self.assertTrue(all(item["drone"] for item in photos))
        self.assertTrue(all(item["camera"] == "DJI FC6310R" for item in photos))
        self.assertAlmostEqual(photos[0]["lat"], 24.680278, places=5)
        self.assertAlmostEqual(photos[0]["lon"], 120.951702, places=5)
        self.assertEqual(photos[0]["altitude"], 100.0)

        raster = next(item for item in result["files"] if item["kind"] == "raster")
        self.assertEqual(raster["path"], "data/ortho.tif")
        self.assertEqual(raster["crs"], "EPSG:32633")
        self.assertEqual(raster["bands"], 3)
        self.assertEqual(raster["role"], "ortho")
        self.assertEqual(len(raster["bounds"]), 4)
        self.assertFalse(result["orthorectify"]["available"])
        self.assertIn("DEM", result["orthorectify"]["reason"])

    def test_a_selection_without_supported_files_is_an_input_error(self):
        plain = self.workspace.data / "notes.txt"
        plain.write_text("not imagery", encoding="utf-8")

        with self.assertRaises(ToolInputError):
            drone.detect_files(self.workspace, ["data/notes.txt"])


class DroneImportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        flight = self.workspace.data / "flight"
        flight.mkdir(parents=True)
        _jpeg(flight / "IMG_0001.JPG")
        _xmp_tiff(flight / "IMG_0002.TIF")
        _raster(self.workspace.data / "ortho.tif")
        self.map = document.create_map(self.workspace)

    def test_import_adds_a_raster_and_one_photo_layer_per_folder(self):
        result = drone.import_files(
            self.workspace,
            self.map,
            lambda rel: f"http://127.0.0.1:9999/api/files/{rel}",
            ["data"],
        )

        self.assertEqual(result["photos"], 2)
        self.assertEqual(result["rasters"], 1)
        self.assertEqual(len(result["layers"]), 2)

        project = self.map.to_project()
        layers = project["layers"]
        self.assertEqual({layer["type"] for layer in layers}, {"cog", "geojson"})

        raster = next(layer for layer in layers if layer["type"] == "cog")
        self.assertIn("results/ortho_cog.tif", raster["source"]["url"])

        photo_layer = next(layer for layer in layers if layer["type"] == "geojson")
        self.assertEqual(photo_layer["name"], "flight photos")
        self.assertEqual(len(photo_layer["geojson"]["features"]), 2)
        self.assertTrue(
            (self.workspace.results / "drone" / "flight-photos.geojson").is_file()
        )

    def test_import_is_persisted_to_the_workspace_snapshot(self):
        drone.import_files(
            self.workspace,
            self.map,
            lambda rel: f"http://127.0.0.1:9999/api/files/{rel}",
            ["data"],
        )
        self.map.save_project(str(document.snapshot_path(self.workspace)))

        snapshot = json.loads(document.snapshot_path(self.workspace).read_text("utf-8"))
        self.assertEqual(len(snapshot["layers"]), 2)


@unittest.skipUnless(
    SAMPLE_CAPTURE.is_dir() and orthorectify.available(),
    "requires Orthority and the dji-orthority sample capture",
)
class DroneOrthorectifyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        shutil.copytree(SAMPLE_CAPTURE, self.workspace.data / "dji-orthority")

    def test_prepare_and_import_adds_a_merged_orthomosaic(self):
        detected = drone.detect_files(self.workspace, ["data/dji-orthority"])
        status = detected["orthorectify"]
        self.assertTrue(status["available"])
        self.assertEqual(status["photos"], 4)
        self.assertEqual(status["dem"], "data/dji-orthority/odm_dem/dsm.tif")

        prepared = orthorectify.prepare_orthomosaic(
            self.workspace, ["data/dji-orthority"]
        )
        self.assertEqual(prepared["frames"], 4)
        self.assertEqual(prepared["crs"], "EPSG:32651")
        output = self.workspace.resolve(prepared["path"], must_exist=True)
        with rasterio.open(output) as dataset:
            self.assertEqual(dataset.count, 3)
            self.assertEqual(dataset.crs.to_string(), "EPSG:32651")
            self.assertTrue(dataset.overviews(1))
            self.assertGreater(dataset.width, 0)

        map_obj = document.create_map(self.workspace)
        result = drone.import_files(
            self.workspace,
            map_obj,
            lambda rel: f"http://127.0.0.1:9999/api/files/{rel}",
            ["data/dji-orthority"],
            orthorectify=True,
            prepared=prepared,
        )
        self.assertEqual(result["orthorectified"]["frames"], 4)
        self.assertEqual(result["photos"], 4)
        self.assertEqual(result["rasters"], 2)  # DEM + generated orthomosaic
        layers = map_obj.to_project()["layers"]
        names = {layer["name"] for layer in layers}
        self.assertIn("orthomosaic", names)
        self.assertIn("dji-orthority photos", names)


class DroneRouteTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._previous_home = os.environ.get("GEOAI_HOME")
        os.environ["GEOAI_HOME"] = self._tmp.name
        deps.close_app_state()
        self.client = TestClient(create_app())

    def tearDown(self):
        deps.close_app_state()
        if self._previous_home is None:
            os.environ.pop("GEOAI_HOME", None)
        else:
            os.environ["GEOAI_HOME"] = self._previous_home
        self._tmp.cleanup()

    def test_detect_then_import_route_updates_the_live_map(self):
        response = self.client.post("/api/workspace/new", json={"name": "drone"})
        self.assertEqual(response.status_code, 200, response.text)
        workspace = deps.app_state().workspaces.workspace
        flight = workspace.data / "flight"
        flight.mkdir(parents=True)
        _jpeg(flight / "IMG_0001.JPG")
        _raster(workspace.data / "ortho.tif")

        detected = self.client.post("/api/drone/detect", json={"paths": ["data"]})
        self.assertEqual(detected.status_code, 200, detected.text)
        body = detected.json()
        self.assertEqual(body["photos"], 1)
        self.assertEqual(body["rasters"], 1)

        imported = self.client.post("/api/drone/import", json={"paths": ["data"]})
        self.assertEqual(imported.status_code, 200, imported.text)
        self.assertEqual(len(imported.json()["layers"]), 2)

        snapshot = self.client.get("/api/state").json()
        self.assertEqual(len(snapshot["map_project"]["layers"]), 2)

    @unittest.skipUnless(
        SAMPLE_CAPTURE.is_dir() and orthorectify.available(),
        "requires Orthority and the dji-orthority sample capture",
    )
    def test_import_route_can_build_an_orthomosaic(self):
        response = self.client.post("/api/workspace/new", json={"name": "drone-ortho"})
        self.assertEqual(response.status_code, 200, response.text)
        workspace = deps.app_state().workspaces.workspace
        shutil.copytree(SAMPLE_CAPTURE, workspace.data / "dji-orthority")

        detected = self.client.post(
            "/api/drone/detect", json={"paths": ["data/dji-orthority"]}
        )
        self.assertEqual(detected.status_code, 200, detected.text)
        self.assertTrue(detected.json()["orthorectify"]["available"])

        imported = self.client.post(
            "/api/drone/import",
            json={"paths": ["data/dji-orthority"], "orthorectify": True},
        )
        self.assertEqual(imported.status_code, 200, imported.text)
        body = imported.json()
        self.assertEqual(body["orthorectified"]["frames"], 4)
        self.assertEqual(body["rasters"], 2)

        snapshot = self.client.get("/api/state").json()
        names = {layer["name"] for layer in snapshot["map_project"]["layers"]}
        self.assertIn("orthomosaic", names)


if __name__ == "__main__":
    unittest.main()
