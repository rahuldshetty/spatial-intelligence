"""Build an orthomosaic from raw drone frames with Orthority.

Orthority (AGPL-3.0-or-later) is imported only when a mosaic is requested.
Each frame is orthorectified against the DEM and the results are merged into one
COG with a first-valid-pixel mosaic; there is no bundle adjustment or seam
blending, so accuracy is bounded by the capture metadata and DEM.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

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

#: Refuse to merge more than this much raster data: ``merge`` loads every frame
#: at once, and writing the filled array roughly doubles the peak.
MAX_MERGE_BYTES = 768 * 1024 * 1024

Progress = Callable[[int, int, str], None]
Cancelled = Callable[[], bool]


def available() -> bool:
    return importlib.util.find_spec("orthority") is not None


def install_hint() -> str:
    return (
        "Orthority is missing from this install; reinstall spatial-intelligence "
        "so its default orthorectification dependency is present"
    )


def _check_cancelled(cancelled: Cancelled | None) -> None:
    if cancelled is not None and cancelled():
        raise ToolInputError("orthorectification cancelled")


def prepare_orthomosaic(
    workspace: Workspace,
    paths: Iterable[str],
    *,
    detected: dict[str, Any] | None = None,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
) -> dict[str, Any]:
    """Detect the selection (unless given) and build its orthomosaic."""
    from . import drone

    detected = (
        detected if detected is not None else drone.detect_files(workspace, paths)
    )
    status = detected["orthorectify"]
    if not status.get("available"):
        raise ToolInputError(
            status.get("reason") or "this selection cannot be orthorectified"
        )
    photos = [
        workspace.resolve(item, must_exist=True)
        for item in status.get("photo_paths", [])
    ]
    dem = workspace.resolve(status["dem"], must_exist=True)
    return orthorectify(
        workspace, photos, dem, progress=progress, cancelled=cancelled
    )


def orthorectify(
    workspace: Workspace,
    photos: Iterable[Path],
    dem: Path,
    *,
    progress: Progress | None = None,
    cancelled: Cancelled | None = None,
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
    _check_cancelled(cancelled)

    try:
        from orthority.errors import OrthorityError, OrthorityWarning
        from orthority.factory import FrameCameras
        from orthority.ortho import Ortho
    except ImportError as exc:  # pragma: no cover - guarded by available()
        raise ToolInputError(install_hint()) from exc

    dataset_name = _dataset_name(workspace, photo_files)
    target = workspace.resolve(
        f"results/drone/{dataset_name}/orthomosaic.tif", write=True
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = target.parent / "_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    total = len(photo_files)
    frame_files: list[Path] = []
    try:
        with rasterio.open(dem) as dem_dataset:
            if dem_dataset.crs is None:
                raise ToolInputError(
                    "the DEM has no CRS, so it cannot anchor an orthomosaic"
                )
            dem_crs = dem_dataset.crs
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
                for index, photo in enumerate(photo_files, start=1):
                    _check_cancelled(cancelled)
                    try:
                        camera = cameras.get(str(photo))
                        with rasterio.open(photo) as source:
                            ortho = Ortho(
                                source,
                                dem_dataset,
                                camera,
                                crs=out_crs,
                                dem_band=1,
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
                    if progress is not None:
                        progress(
                            index, total, f"frame {index}/{total}: {photo.name}"
                        )

                _check_cancelled(cancelled)
                _merge_frames(frame_files, target)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    with rasterio.open(target) as dataset:
        bounds = dataset.bounds
        crs = dataset.crs
        resolution = [abs(dataset.transform.a), abs(dataset.transform.e)]
        west, south, east, north = transform_bounds(
            crs, "EPSG:4326", *bounds, densify_pts=21
        )

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
    sources: list[rasterio.DatasetReader] = []
    total_bytes = 0
    try:
        for frame in frames:
            source = rasterio.open(frame)
            sources.append(source)
            item_size = np.dtype(source.dtypes[0]).itemsize
            total_bytes += source.width * source.height * source.count * item_size
        if total_bytes > MAX_MERGE_BYTES:
            raise ToolInputError(
                f"the orthomosaic would need {total_bytes // (1024 * 1024)} MiB "
                f"to merge (limit {MAX_MERGE_BYTES // (1024 * 1024)} MiB); "
                "use ODM/WebODM for a survey this large"
            )
        array, transform = merge(sources, masked=True)
        if not np.ma.isMaskedArray(array):
            array = np.ma.masked_array(array, mask=np.zeros(array.shape, dtype=bool))
        profile = sources[0].profile.copy()
        fill = profile.get("nodata")
        if fill is None:
            fill = 0
        profile.pop("tiled", None)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            width=array.shape[2],
            height=array.shape[1],
            transform=transform,
            count=array.shape[0],
            dtype=array.dtype,
            driver="COG",
            BLOCKSIZE=512,
            compress="deflate",
            BIGTIFF="YES",
        )
        with rasterio.open(target, "w", **profile) as dataset:
            dataset.write(array.filled(fill))
            mask = np.ma.getmaskarray(array)
            if mask.any():
                dataset.write_mask(~mask.any(axis=0))
    finally:
        for source in sources:
            source.close()


def _dataset_name(workspace: Workspace, photos: list[Path]) -> str:
    """A stable output folder name, unique across unrelated selections."""
    parents = {photo.parent for photo in photos}
    if len(parents) == 1:
        folder = workspace.relative(next(iter(parents)))
    else:
        joined = "\n".join(sorted(str(parent) for parent in parents))
        folder = "drone-" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", folder).strip("-") or "drone"


__all__ = [
    "MAX_MERGE_BYTES",
    "MAX_PHOTOS",
    "available",
    "install_hint",
    "orthorectify",
    "prepare_orthomosaic",
]
