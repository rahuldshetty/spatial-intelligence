"""Detect drone imagery in a workspace and add it to the live map.

Detection reads headers only: a raster's CRS/bounds through rasterio, a photo's
EXIF/XMP through Pillow. No pixels are decoded before the user confirms.
"""

from __future__ import annotations

import json
import math
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import rasterio
from PIL import Image, UnidentifiedImageError
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform_bounds

from ..contracts.errors import ToolInputError
from ..map import document as map_document
from ..map import layers as layerops
from ..workspace import Workspace
from . import orthorectify as ortho_service
from . import raster

#: Photo containers Pillow can read a header from. TIFF is included because
#: some drone cameras write TIFF frames with EXIF/XMP instead of JPEG.
PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"})

#: Raster containers a georeferenced orthomosaic/DSM commonly arrives in.
RASTER_SUFFIXES = frozenset({".tif", ".tiff"})

#: Camera makes that identify a capture as drone imagery. The XMP namespace is
#: a stronger signal, so a file can still be marked as drone without a match.
DRONE_MAKES = (
    "dji",
    "autel",
    "parrot",
    "sensefly",
    "wingtra",
    "yuneec",
    "skydio",
    "3dr",
    "delair",
    "quantum",
)

#: XMP tag where TIFF stores its packet. JPEG has no equivalent EXIF tag; its
#: XMP lives in an APP1 segment, which Pillow exposes through ``info["xmp"]``.
XMP_TAG = 700

#: Cap the files inspected so a mis-selected workspace root cannot turn
#: detection into an unbounded scan. The walk stops at the cap.
MAX_SCAN_FILES = 5000

#: XMP packets are a few kilobytes; anything larger is malformed or hostile and
#: is not handed to the XML parser.
MAX_XMP_BYTES = 1_048_576

#: EXIF IFD and tag ids used here. Named for readability at the call site.
GPS_IFD = 34853
EXIF_IFD = 34665
EXIF_MAKE = 271
EXIF_MODEL = 272
EXIF_DATETIME = 36867
GPS_LATITUDE = 2
GPS_LATITUDE_REF = 1
GPS_LONGITUDE = 4
GPS_LONGITUDE_REF = 3
GPS_ALTITUDE = 6
GPS_ALTITUDE_REF = 5
GPS_DIRECTION = 17

#: Fields read from both EXIF and XMP. XMP matches are by local name, so vendor
#: prefixes (``drone-dji:``, ``tiff:``, ``exif:``) do not matter.
_XMP_KEYS = {
    "GpsLatitude": "latitude",
    "GpsLongitude": "longitude",
    "GpsLongtitude": "longitude",  # DJI's spelling in older XMP packets
    "GPSLatitude": "latitude",
    "GPSLongitude": "longitude",
    "GPSLatitudeRef": "latitude_ref",
    "GPSLongitudeRef": "longitude_ref",
    "GimbalYawDegree": "direction",
    "FlightYawDegree": "direction",
    "AbsoluteAltitude": "altitude",
    "RelativeAltitude": "relative_altitude",
    "Make": "make",
    "Model": "model",
    "CreateDate": "timestamp",
    "DateTimeOriginal": "timestamp",
}

#: First number in a metadata string, tolerating DJI's leading ``+`` and degree
#: suffixes that some exporters append.
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def detect_files(workspace: Workspace, paths: Iterable[str]) -> dict[str, Any]:
    """Return the supported drone files in ``paths``.

    ``paths`` may name files or directories. Directories are walked recursively;
    each file is classified by its header only. The result is intentionally
    shaped for the import dialog: one row per file plus counts and a union
    bounding box.
    """
    sources, truncated = _candidate_paths(workspace, paths)
    files: list[dict[str, Any]] = []
    skipped = 0
    for source in sources:
        record = _classify(source)
        if record is None:
            skipped += 1
            continue
        record["path"] = workspace.relative(source)
        record["size"] = source.stat().st_size
        files.append(record)

    files.sort(key=lambda item: item["path"])
    photos = sum(1 for item in files if item["kind"] == "photo")
    rasters = len(files) - photos
    return {
        "files": files,
        "photos": photos,
        "rasters": rasters,
        "skipped": skipped,
        "truncated": truncated,
        "bounds": _union_bounds(files),
        "orthorectify": _orthorectify_status(files),
    }


def import_files(
    workspace: Workspace,
    map_obj: Any,
    file_url: Callable[[str], str],
    paths: Iterable[str],
    *,
    orthorectify: bool = False,
    prepared: dict[str, Any] | None = None,
    detected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add detected rasters and photos to the live map.

    ``detected``/``prepared`` reuse work already done outside the map lock.
    A source that is already on the map is updated or reused instead of being
    added twice, and a failure removes the layers this call added.
    """
    detected = detected if detected is not None else detect_files(workspace, paths)
    files = list(detected["files"])
    if orthorectify:
        if prepared is None:
            prepared = ortho_service.prepare_orthomosaic(workspace, paths)
        generated = detect_files(workspace, [prepared["path"]])["files"]
        if not generated:
            raise ToolInputError("the generated orthomosaic is not a readable raster")
        files.extend(generated)

    layers: list[dict[str, Any]] = []
    added_ids: list[str] = []
    rasters = [item for item in files if item["kind"] == "raster"]
    photos = [item for item in files if item["kind"] == "photo"]

    try:
        for record in rasters:
            source = workspace.resolve(record["path"], must_exist=True)
            loadable, _ = raster.to_cog_if_needed(workspace, record["path"])
            relative = workspace.relative(loadable)
            existing = _layer_for_source(map_obj, relative)
            if existing is not None:
                layers.append(_layer_summary(existing, "raster", relative))
                continue
            name = _unique_name(map_obj, _raster_layer_name(record), relative)
            layer_id = layerops.add_raster(
                workspace,
                map_obj,
                relative,
                name,
                bands=_rgb_bands(source),
                file_url=file_url,
            )
            added_ids.append(layer_id)
            layers.append(
                {"id": layer_id, "name": name, "kind": "raster", "path": relative}
            )

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in photos:
            groups[str(PurePosixPath(record["path"]).parent)].append(record)
        for parent, records in sorted(groups.items()):
            collection = _photo_geojson(records)
            relative = _write_photo_geojson(workspace, parent, collection)
            existing = _layer_for_source(map_obj, relative)
            if existing is not None:
                existing["geojson"] = collection
                map_document.persist_map(map_obj, workspace)
                layers.append(_layer_summary(existing, "photos", relative))
                continue
            name = _unique_name(map_obj, _photo_layer_name(parent), relative)
            layer_id = layerops.add_geojson(
                workspace,
                map_obj,
                relative,
                name,
                style={
                    "fillColor": "#f59e0b",
                    "strokeColor": "#b45309",
                    "circleRadius": 6,
                    "fillOpacity": 0.85,
                },
                source=relative,
            )
            added_ids.append(layer_id)
            layers.append(
                {"id": layer_id, "name": name, "kind": "photos", "path": relative}
            )
    except Exception:
        for layer_id in reversed(added_ids):
            try:
                layerops.remove_layer(workspace, map_obj, layer_id)
            except Exception:
                pass
        if added_ids:
            map_document.persist_map(map_obj, workspace)
        raise

    bounds = detected["bounds"]
    if prepared and prepared.get("bounds"):
        bounds = _combine_bounds(bounds, prepared["bounds"])
    if bounds:
        layerops.fit_bounds(workspace, map_obj, bounds)

    return {
        "layers": layers,
        "added": len(added_ids),
        "reused": len(layers) - len(added_ids),
        "photos": detected["photos"],
        "rasters": len(rasters),
        "bounds": bounds,
        "orthorectified": prepared if orthorectify else None,
    }


def _orthorectify_status(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe whether this selection can be turned into an orthomosaic."""
    photos = [item for item in files if item["kind"] == "photo"]
    dems = [
        item
        for item in files
        if item["kind"] == "raster" and item.get("role") == "dem"
    ]
    dem = _pick_dem(photos, dems)
    shown = dem or (dems[0] if dems else None)
    status: dict[str, Any] = {
        "available": False,
        "photos": len(photos),
        "photo_paths": [],
        "dem": shown["path"] if shown else None,
        "reason": "",
    }
    if not photos:
        status["reason"] = "no geotagged photos to orthorectify"
    elif not ortho_service.available():
        status["reason"] = ortho_service.install_hint()
    elif not dems:
        status["reason"] = "no DEM/DSM in the selection"
    elif dem is None:
        status["reason"] = "the DEM does not overlap the photo locations"
    elif len(photos) > ortho_service.MAX_PHOTOS:
        status["reason"] = (
            f"{len(photos)} photos is over the {ortho_service.MAX_PHOTOS}-photo "
            "limit for in-app orthorectification; use ODM/WebODM"
        )
    else:
        status["available"] = True
        status["photo_paths"] = [item["path"] for item in photos]
    return status


def _pick_dem(
    photos: list[dict[str, Any]], dems: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Choose the DEM that contains (or overlaps most of) the photo points."""
    if not dems:
        return None
    photo_box = _union_bounds(photos)
    if photo_box is None:
        return dems[0]
    for dem in dems:
        if _contains(dem.get("bounds"), photo_box):
            return dem
    overlapping = [dem for dem in dems if _overlap_area(dem.get("bounds"), photo_box)]
    if not overlapping:
        return None
    return max(overlapping, key=lambda dem: _overlap_area(dem.get("bounds"), photo_box))


def _contains(outer: list[float] | None, inner: list[float] | None) -> bool:
    if not outer or not inner:
        return False
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _overlap_area(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width * height if width > 0 and height > 0 else 0.0


def _candidate_paths(
    workspace: Workspace, paths: Iterable[str]
) -> tuple[list[Path], bool]:
    """Resolve the selection to unique supported files, walking directories.

    Returns the files and whether :data:`MAX_SCAN_FILES` cut the walk short.
    """
    selected: list[Path] = []
    seen: set[Path] = set()
    truncated = False
    scanned = 0
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = workspace.resolve(raw.strip(), must_exist=True)
        if path.is_dir():
            targets = (item for item in path.rglob("*") if item.is_file())
        else:
            targets = iter([path])
        for target in targets:
            scanned += 1
            if scanned > MAX_SCAN_FILES:
                truncated = True
                break
            suffix = target.suffix.lower()
            if suffix not in PHOTO_SUFFIXES and suffix not in RASTER_SUFFIXES:
                continue
            resolved = target.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            selected.append(target)
        if truncated:
            break
    if not selected:
        if truncated:
            raise ToolInputError(
                f"scanned {MAX_SCAN_FILES} files without finding supported drone "
                "imagery; select a narrower folder"
            )
        raise ToolInputError(
            "no supported drone files in the selection (need GeoTIFF/COG or geotagged JPEG/TIFF)"
        )
    return selected, truncated


def _classify(source: Path) -> dict[str, Any] | None:
    """Classify one file as a georeferenced raster or a geotagged photo."""
    if source.suffix.lower() in RASTER_SUFFIXES:
        raster = _raster_record(source)
        if raster is not None:
            return raster
    if source.suffix.lower() in PHOTO_SUFFIXES:
        return _photo_record(source)
    return None


def _raster_record(source: Path) -> dict[str, Any] | None:
    """Return a raster record when the file has a CRS and a usable extent."""
    try:
        with warnings.catch_warnings():
            # A raw TIFF is expected to have no georeferencing; the photo reader
            # handles it next, so the warning is noise rather than a problem.
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            dataset = rasterio.open(source)
        with dataset:
            if dataset.crs is None or dataset.width <= 0 or dataset.height <= 0:
                return None
            if getattr(dataset.transform, "is_identity", False):
                return None
            bounds = dataset.bounds
            if not all(math.isfinite(value) for value in bounds):
                return None
            if bounds.left >= bounds.right or bounds.bottom >= bounds.top:
                return None
            west, south, east, north = transform_bounds(
                dataset.crs, "EPSG:4326", *bounds, densify_pts=21
            )
            crs = dataset.crs.to_string()
            bands = dataset.count
            width = dataset.width
            height = dataset.height
    except Exception:
        # A raw TIFF is not a raster failure; the photo reader gets it next.
        return None

    if not (-180 <= west <= 180 and -90 <= south <= 90):
        return None
    if not (-180 <= east <= 180 and -90 <= north <= 90):
        return None

    name = source.stem.lower()
    role = "dem" if bands == 1 or any(
        hint in name
        for hint in ("dsm", "dtm", "dem", "elevation", "surface", "terrain", "height")
    ) else "ortho"
    detail = f"GeoTIFF · {crs} · {width}×{height} px · {bands} band(s)"
    if role == "dem":
        detail += " · DEM"
    return {
        "kind": "raster",
        "label": source.stem.replace("_", " ").strip() or "Georeferenced raster",
        "detail": detail,
        "bounds": [west, south, east, north],
        "crs": crs,
        "bands": bands,
        "role": role,
    }


def _photo_record(source: Path) -> dict[str, Any] | None:
    """Return a photo record when EXIF/XMP gives a usable GPS position."""
    try:
        with Image.open(source) as image:
            exif = image.getexif()
            gps = _exif_gps(exif)
            xmp_text = _xmp_text(exif, image)
            xmp = _xmp_fields(xmp_text)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return None

    if gps:
        latitude, longitude = gps["latitude"], gps["longitude"]
    else:
        latitude = _xmp_coord(xmp.get("latitude"), xmp.get("latitude_ref"))
        longitude = _xmp_coord(xmp.get("longitude"), xmp.get("longitude_ref"))
    if latitude is None or longitude is None:
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None

    altitude = (gps or {}).get("altitude")
    altitude_kind = "gps" if altitude is not None else None
    if altitude is None:
        altitude = _number(xmp.get("altitude"))
        altitude_kind = "absolute" if altitude is not None else None
    if altitude is None:
        altitude = _number(xmp.get("relative_altitude"))
        altitude_kind = "relative" if altitude is not None else None
    direction = (gps or {}).get("direction")
    if direction is None:
        direction = _number(xmp.get("direction"))
    timestamp = _exif_datetime_original(exif) or xmp.get("timestamp", "")

    make = _text(exif.get(EXIF_MAKE)) or xmp.get("make", "")
    model = _text(exif.get(EXIF_MODEL)) or xmp.get("model", "")
    camera = " ".join(part for part in (make, model) if part).strip()
    drone = _looks_like_drone(camera, xmp_text)
    kind_label = "Drone photo" if drone else "Geotagged photo"
    detail = f"{kind_label} · {camera or 'unknown camera'}"
    detail += f" · {latitude:.6f}, {longitude:.6f}"
    if altitude is not None:
        suffix = " relative" if altitude_kind == "relative" else ""
        detail += f" · {altitude:.1f} m{suffix}"

    record: dict[str, Any] = {
        "kind": "photo",
        "label": camera or "Photo",
        "detail": detail,
        "bounds": [longitude, latitude, longitude, latitude],
        "lon": longitude,
        "lat": latitude,
        "camera": camera,
        "timestamp": timestamp,
        "drone": drone,
    }
    if altitude is not None:
        record["altitude"] = altitude
        record["altitude_kind"] = altitude_kind or "gps"
    if direction is not None:
        record["direction"] = direction
    return record


def _exif_gps(exif: Any) -> dict[str, float] | None:
    """Read the EXIF GPS IFD into decimal degrees and metres."""
    try:
        gps = exif.get_ifd(GPS_IFD) or {}
    except Exception:
        return None
    if not gps:
        return None
    latitude = _dms(gps.get(GPS_LATITUDE))
    longitude = _dms(gps.get(GPS_LONGITUDE))
    if latitude is None or longitude is None:
        return None
    if _text(gps.get(GPS_LATITUDE_REF)).upper().startswith("S"):
        latitude = -latitude
    if _text(gps.get(GPS_LONGITUDE_REF)).upper().startswith("W"):
        longitude = -longitude

    result = {"latitude": latitude, "longitude": longitude}
    altitude = _number(gps.get(GPS_ALTITUDE))
    if altitude is not None:
        # GPSAltitudeRef is 0 above sea level and 1 below it.
        if _text(gps.get(GPS_ALTITUDE_REF)).strip("\x00") in ("\x01", "1"):
            altitude = -altitude
        result["altitude"] = altitude
    direction = _number(gps.get(GPS_DIRECTION))
    if direction is not None:
        result["direction"] = direction
    return result


def _exif_datetime_original(exif: Any) -> str:
    """Read DateTimeOriginal from the top level or the Exif sub-IFD."""
    value = _text(exif.get(EXIF_DATETIME))
    if value:
        return value
    try:
        sub = exif.get_ifd(EXIF_IFD) or {}
    except Exception:
        return ""
    return _text(sub.get(EXIF_DATETIME))


def _xmp_text(exif: Any, image: Any) -> str:
    """Return the XMP packet from a TIFF tag or a JPEG APP1 segment."""
    raw = exif.get(XMP_TAG)
    if not raw:
        raw = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp")
    if not raw:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return str(raw)


def _xmp_fields(packet: str) -> dict[str, str]:
    """Read the XMP fields this module uses, matched by local name.

    Vendor prefixes (``drone-dji:``, ``exif:``, ``tiff:``) do not matter, and a
    malformed packet is ignored instead of failing the import.
    """
    if not packet:
        return {}
    packet = packet[:MAX_XMP_BYTES]
    start = packet.find("<x:xmpmeta")
    end = packet.find("</x:xmpmeta>")
    if start >= 0 and end > start:
        packet = packet[start : end + len("</x:xmpmeta>")]
    else:
        start = packet.find("<rdf:RDF")
        end = packet.find("</rdf:RDF>")
        if start >= 0 and end > start:
            packet = packet[start : end + len("</rdf:RDF>")]

    fields: dict[str, str] = {}
    try:
        root = ET.fromstring(packet)
    except ET.ParseError:
        # Some cameras write a truncated/odd packet; pull the few known keys.
        for key, target in _XMP_KEYS.items():
            match = re.search(rf'{re.escape(key)}="([^"]*)"', packet)
            if match and target not in fields:
                fields[target] = _unescape(match.group(1)).strip()
        return fields

    for element in root.iter():
        for key, value in element.attrib.items():
            local = key.rsplit("}", 1)[-1]
            target = _XMP_KEYS.get(local)
            if target and target not in fields:
                fields[target] = _unescape(value).strip()
        if element.text:
            text = element.text.strip()
            if text and element.tag.rsplit("}", 1)[-1] in _XMP_KEYS:
                target = _XMP_KEYS[element.tag.rsplit("}", 1)[-1]]
                fields.setdefault(target, text)
    return fields


def _looks_like_drone(camera: str, xmp_text: str) -> bool:
    lowered = camera.lower()
    if any(make in lowered for make in DRONE_MAKES):
        return True
    packet = xmp_text.lower()
    return "drone-dji" in packet or "gimbal" in packet


def _photo_geojson(records: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    for record in records:
        properties = {
            "name": PurePosixPath(record["path"]).name,
            "path": record["path"],
            "camera": record.get("camera", ""),
            "timestamp": record.get("timestamp", ""),
        }
        for key in ("altitude", "direction"):
            value = record.get(key)
            if value is not None:
                properties[key] = round(float(value), 3)
        if record.get("altitude_kind"):
            properties["altitude_kind"] = record["altitude_kind"]
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [record["lon"], record["lat"]],
                },
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _write_photo_geojson(
    workspace: Workspace, parent: str, collection: dict[str, Any]
) -> str:
    """Write the point layer under ``results/`` and return its relative path."""
    target = workspace.resolve(
        f"results/drone/{_slug(parent)}-photos.geojson", write=True
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    relative = workspace.relative(target)
    workspace.record_output(relative)
    return relative


def _slug(value: str) -> str:
    """A filesystem-safe name for a parent path (so same-named folders differ)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "drone"


def _layer_for_source(map_obj: Any, relative: str) -> dict[str, Any] | None:
    for layer in layerops.layers(map_obj):
        metadata = layer.get("metadata") or {}
        if metadata.get(layerops.LOCAL_SOURCE_KEY) == relative:
            return layer
    return None


def _layer_summary(layer: dict[str, Any], kind: str, path: str) -> dict[str, Any]:
    return {
        "id": str(layer.get("id")),
        "name": layer.get("name"),
        "kind": kind,
        "path": path,
        "reused": True,
    }


def _unique_name(map_obj: Any, base: str, source: str) -> str:
    taken = {
        str(layer.get("name"))
        for layer in layerops.layers(map_obj)
        if (layer.get("metadata") or {}).get(layerops.LOCAL_SOURCE_KEY) != source
    }
    if base not in taken:
        return base
    index = 2
    while f"{base} ({index})" in taken:
        index += 1
    return f"{base} ({index})"


def _raster_layer_name(record: dict[str, Any]) -> str:
    name = PurePosixPath(record["path"]).stem.replace("_", " ").strip()
    return name or "Drone raster"


def _photo_layer_name(parent: str) -> str:
    folder = PurePosixPath(parent).name
    return f"{folder} photos" if folder else "Drone photos"


def _rgb_bands(source: Path) -> list[int] | None:
    """Select the first three bands of a colour raster for display.

    A three-band ortho left at ``bands=None`` renders as grayscale, so the
    explicit selection keeps imagery in colour. A fourth band is ignored,
    which drops an alpha channel.
    """
    try:
        with rasterio.open(source) as dataset:
            if dataset.count < 3:
                return None
            if dataset.count <= 4:
                return [1, 2, 3]
            color = [item.name for item in dataset.colorinterp[:3]]
            if color == ["red", "green", "blue"]:
                return [1, 2, 3]
    except Exception:
        pass
    return None


def _union_bounds(files: list[dict[str, Any]]) -> list[float] | None:
    return _combine_bounds(*(item["bounds"] for item in files if item.get("bounds")))


def _combine_bounds(*boxes: list[float] | None) -> list[float] | None:
    """Union of ``[west, south, east, north]`` boxes, ignoring empty ones."""
    present = [box for box in boxes if box]
    if not present:
        return None
    return [
        min(box[0] for box in present),
        min(box[1] for box in present),
        max(box[2] for box in present),
        max(box[3] for box in present),
    ]


def _dms(value: Any) -> float | None:
    """Convert an EXIF degrees/minutes/seconds tuple to decimal degrees."""
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    parts = [_number(item) for item in value]
    if parts[0] is None or parts[1] is None:
        return None
    seconds = parts[2] if len(parts) > 2 and parts[2] is not None else 0.0
    return parts[0] + parts[1] / 60.0 + seconds / 3600.0


def _xmp_coord(value: Any, ref: Any) -> float | None:
    """Parse an XMP coordinate.

    Handles decimal (DJI), ``24,40.8167N``, and degrees/minutes/seconds forms;
    a hemisphere in the value or in ``GPSLatitudeRef``/``GPSLongitudeRef`` sets
    the sign.
    """
    raw = _text(value)
    if not raw:
        return None
    hemisphere = ""
    match = re.search(r"([NSEW])\s*$", raw, re.IGNORECASE)
    if match:
        hemisphere = match.group(1).upper()
        raw = raw[: match.start()]
    else:
        match = re.match(r"\s*([NSEW])", raw, re.IGNORECASE)
        if match:
            hemisphere = match.group(1).upper()
            raw = raw[match.end() :]
    numbers = [float(part) for part in _NUMBER.findall(raw)]
    if not numbers:
        return None
    if len(numbers) == 1:
        result = numbers[0]
    else:
        sign = -1.0 if numbers[0] < 0 else 1.0
        minutes = numbers[1] if len(numbers) > 1 else 0.0
        seconds = numbers[2] if len(numbers) > 2 else 0.0
        result = sign * (abs(numbers[0]) + minutes / 60.0 + seconds / 3600.0)
    ref_text = _text(ref).upper()
    if hemisphere in ("S", "W") or ref_text.startswith(("S", "W")):
        return -abs(result)
    if hemisphere in ("N", "E") or ref_text.startswith(("N", "E")):
        return abs(result)
    return result


def _number(value: Any) -> float | None:
    """Parse a number from an EXIF rational, int/float, or metadata string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator:
        try:
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    match = _NUMBER.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _text(value: Any) -> str:
    """Decode an EXIF string/bytes value and trim NUL padding."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip("\x00 ")
    return str(value).strip("\x00 ")


def _unescape(value: str) -> str:
    return (
        value.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )


__all__ = ["detect_files", "import_files"]
