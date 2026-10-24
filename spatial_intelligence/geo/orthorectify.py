"""Build an orthomosaic from raw drone frames with Orthority.

Orthority is a default dependency (AGPL-3.0-or-later). Nothing here imports it
until the user actually asks for an orthomosaic, so detection and the normal
import path stay cheap.

Orthority orthorectifies one frame at a time. This module runs it over all
selected frames against the supplied DEM and then merges the per-frame orthos
into one COG. The merge is a simple first-valid-pixel mosaic; it does not
bundle-adjust camera poses or blend seams, so accuracy is bounded by the
capture metadata and DEM.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.merge import merge
from rasterio.warp import transform_bounds

from ..contracts.errors import ToolInputError
from ..workspace import Workspace

#: Cap a single synchronous import. Larger surveys belong in ODM/WebODM, which
#: does bundle adjustment and is not limited by one HTTP request.
MAX_PHOTOS = 200

#: Names that mark a single-band raster as an elevation model.
_DEM_HINTS = ("dsm", "dtm", "dem", "elevation", "surface", "terrain", "height")


def available() -> bool:
    """Whether Orthority is installed in this environment."""
    return importlib.util.find_spec("orthority") is not None


def install_hint() -> str:
    """One actionable sentence for the UI when Orthority is missing."""
    return (
        "Orthority is missing from this install; reinstall spatial-intelligence "
        "so its default orthorectification dependency is present"
    )


def prepare_orthomosaic(workspace: Workspace, paths: Iterable[str]) -> dict[str, Any]:
    """Detect a photo/DEM selection and build its orthomosaic.

    This is the slow, map-free half of the import. ``app_state`` calls it before
    taking the session/map locks, so a long Orthority run does not block state
    polling or the rest of the UI.
    """
    from . import drone

    detected = drone.detect_files(workspace, paths)
    status = detected["orthorectify"]
    if not status.get("available"):
        raise ToolInputError(status.get("reason") or "this selection cannot be orthorectified")
    photos = [
        workspace.resolve(item, must_exist=True) for item in status.get("photo_paths", [])
    ]
    dem = workspace.resolve(status["dem"], must_exist=True)
    return orthorectify(workspace, photos, dem)


def orthorectify(
    workspace: Workspace,
    photos: Iterable[Path],
    dem: Path,
) -> dict[str, Any]:
    """Orthorectify ``photos`` against ``dem`` and merge them into one COG."""
    if not available():
        raise ToolInputError(install_hint())

    photo_files = [Path(path) for path in photos]
    if not photo_files:
        raise ToolInputError("no photos to orthorectify")
    if len(photo_files) > MAX_PHOTOS:
        raise ToolInputError(
            f"orthorectification is limited to {MAX_PHOTOS} photos per import; "
            "use ODM/WebODM for a larger survey"
        )
    if not dem.is_file():
        raise ToolInputError(f"DEM does not exist: {dem}")

    try:
        from orthority.errors import OrthorityError, OrthorityWarning
        from orthority.factory import FrameCameras
        from orthority.ortho import Ortho
    except ImportError as exc:  # pragma: no cover - guarded by available()
        raise ToolInputError(install_hint()) from exc

    with rasterio.open(dem) as dataset:
        dem_crs = dataset.crs
        dem_band = 1
    if dem_crs is None:
        raise ToolInputError("the DEM has no CRS, so it cannot anchor an orthomosaic")

    dataset_name = _dataset_name(photo_files)
    target = workspace.resolve(
        f"results/drone/{dataset_name}/orthomosaic.tif", write=True
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = target.parent / "_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            warnings.simplefilter("ignore", OrthorityWarning)
            # Orthority 0.7 uses the deprecated Affine ``*`` operator internally.
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            try:
                cameras = FrameCameras.from_images(
                    [str(path) for path in photo_files],
                    io_kwargs={"crs": dem_crs, "progress": False},
                    cam_kwargs={"distort": True, "alpha": 1},
                )
            except (OrthorityError, FileNotFoundError, ValueError) as exc:
                raise ToolInputError(
                    "Orthority could not read camera parameters from the photos: "
                    f"{exc}. The frames need EXIF/XMP GPS, orientation, and camera "
                    "intrinsics (for DJI, DewarpData or focal-length tags)."
                ) from exc

            out_crs = cameras.crs or dem_crs
            frame_files: list[Path] = []
            for index, photo in enumerate(photo_files, start=1):
                try:
                    camera = cameras.get(str(photo))
                    with rasterio.open(photo) as source, rasterio.open(dem) as dem_dataset:
                        ortho = Ortho(
                            source,
                            dem_dataset,
                            camera,
                            crs=out_crs,
                            dem_band=dem_band,
                        )
                        frame = frames_dir / f"{index:04d}-{photo.stem}_ORTHO.tif"
                        ortho.process(
                            str(frame),
                            driver="cog",
                            overwrite=True,
                            progress=False,
                        )
                        frame_files.append(frame)
                except (OrthorityError, OSError, ValueError) as exc:
                    raise ToolInputError(
                        f"could not orthorectify {photo.name}: {exc}"
                    ) from exc

            _merge_frames(frame_files, target)

        with rasterio.open(target) as dataset:
            bounds = dataset.bounds
            crs = dataset.crs
            resolution = [abs(dataset.transform.a), abs(dataset.transform.e)]
            west, south, east, north = transform_bounds(
                crs, "EPSG:4326", *bounds, densify_pts=21
            )
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    relative = workspace.relative(target)
    workspace.record_output(relative)
    return {
        "path": relative,
        "frames": len(photo_files),
        "dem": workspace.relative(dem),
        "crs": crs.to_string() if crs is not None else "",
        "bounds": [west, south, east, north],
        "resolution": resolution,
    }


def _merge_frames(frames: list[Path], target: Path) -> None:
    """Merge per-frame orthos into one masked COG."""
    if not frames:
        raise ToolInputError("Orthority produced no ortho images")
    sources = [rasterio.open(frame) for frame in frames]
    try:
        array, transform = merge(sources, masked=True)
        if not np.ma.isMaskedArray(array):
            array = np.ma.masked_array(array, mask=np.zeros(array.shape, dtype=bool))
        profile = sources[0].profile.copy()
        profile.update(
            width=array.shape[2],
            height=array.shape[1],
            transform=transform,
            count=array.shape[0],
            dtype=array.dtype,
            driver="COG",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
            BIGTIFF="YES",
        )
        with rasterio.open(target, "w", **profile) as dataset:
            dataset.write(array.filled(0))
            mask = np.ma.getmaskarray(array)
            if mask.any():
                dataset.write_mask(~mask.any(axis=0))
    finally:
        for source in sources:
            source.close()


def _dataset_name(photos: list[Path]) -> str:
    """A stable output folder name from the capture folder, else ``drone``."""
    parents = {photo.parent for photo in photos}
    folder = str(sorted(parents)[0].name) if len(parents) == 1 else "drone"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", folder).strip("-") or "drone"
    return slug


def is_dem(record: dict[str, Any]) -> bool:
    """Whether a detected raster record looks like an elevation model."""
    if record.get("kind") != "raster":
        return False
    if record.get("role") == "dem":
        return True
    name = PurePosixPath(str(record.get("path", ""))).stem.lower()
    if any(hint in name for hint in _DEM_HINTS):
        return True
    return int(record.get("bands") or 0) == 1


__all__ = [
    "MAX_PHOTOS",
    "available",
    "install_hint",
    "is_dem",
    "orthorectify",
    "prepare_orthomosaic",
]
