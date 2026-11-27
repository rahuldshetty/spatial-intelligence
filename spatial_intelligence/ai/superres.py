"""Satlas ESRGAN super-resolution, tiled through ONNX Runtime.

The model upscales 4x and is fully convolutional, so a raster is read in cores
with a halo, run one tile at a time, and written at four times the resolution.
The halo is context only: its output is cropped, which is what keeps tile
boundaries from becoming seams.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform

from ..contracts.errors import ToolInputError
from ..contracts.progress import Job
from ..workspace import Workspace
from .manager import ModelSession
from .tiles import open_source, read_window

SCALE = 4
DEFAULT_TILE = 128
DEFAULT_HALO = 16
MAX_TILES = 64
DEFAULT_OUT = "results/superres.tif"

#: Sentinel-2 reflectance is scaled to this; TCI is ordinary 8-bit RGB.
REFLECTANCE_SCALE = 8160.0
BYTE_SCALE = 255.0


@dataclass(slots=True)
class Timings:
    """Seconds spent reading, running the graph, and writing."""

    read: float = 0.0
    infer: float = 0.0
    write: float = 0.0
    tiles: int = 0
    skipped_tiles: int = 0

    def as_dict(self) -> dict:
        return {
            "tiles": self.tiles,
            "skipped_tiles": self.skipped_tiles,
            "seconds": {
                "read": round(self.read, 2),
                "infer": round(self.infer, 2),
                "write": round(self.write, 2),
            },
        }


def plan_cores(width: int, height: int, tile: int) -> list[tuple[int, int, int, int]]:
    """Return non-overlapping ``(x, y, width, height)`` cores covering a raster."""
    tile = max(32, int(tile))
    return [
        (x, y, min(tile, width - x), min(tile, height - y))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def normalize(array: np.ndarray) -> np.ndarray:
    """Scale one band to 0-1 the way the model was trained.

    TCI and other 8-bit products are divided by 255; integer Sentinel-2 bands
    are divided by 8160 and clipped, which is the reflectance convention the
    training chips were built with. Float input is assumed to be 0-1 already.
    """
    values = array.astype(np.float32)
    if array.dtype == np.uint8:
        return values / BYTE_SCALE
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(values / REFLECTANCE_SCALE, 0.0, 1.0)
    return np.clip(values, 0.0, 1.0)


def _pad(array: np.ndarray, pads: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Pad a tile, reflecting edges unless the raster is smaller than the halo."""
    try:
        return np.pad(array, pads, mode="reflect")
    except ValueError:
        return np.pad(array, pads, mode="edge")


def read_stack(
    datasets: Sequence[Any],
    window: Window,
    bands: Sequence[int],
    halo: int,
) -> tuple[np.ndarray, int, int]:
    """Read one co-registered window from every date, plus a halo.

    Returns ``(channels, height + 2*halo, width + 2*halo)``, date-major, and the
    offsets at which the core begins after edge padding. The region always starts
    ``halo`` pixels before the core, so the core itself starts at ``(halo, halo)``.
    """
    reference = datasets[0]
    left = max(0, window.col_off - halo)
    top = max(0, window.row_off - halo)
    right = min(reference.width, window.col_off + window.width + halo)
    bottom = min(reference.height, window.row_off + window.height + halo)
    inner = Window(left, top, right - left, bottom - top)
    parts = [normalize(dataset.read(indexes=list(bands), window=inner)) for dataset in datasets]
    stacked = np.concatenate(parts, axis=0)
    pad_top = top - (window.row_off - halo)
    pad_left = left - (window.col_off - halo)
    pad_bottom = (window.row_off + window.height + halo) - bottom
    pad_right = (window.col_off + window.width + halo) - right
    pads = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
    if any(before or after for before, after in pads):
        stacked = _pad(stacked, pads)
    return stacked, halo, halo


def _same_grid(reference: Any, other: Any, path: str) -> None:
    """Refuse a date that is not on the reference raster's pixel grid."""
    if other.crs != reference.crs:
        raise ToolInputError(f"{path} is {other.crs}, not {reference.crs}")
    if (other.width, other.height) != (reference.width, reference.height):
        raise ToolInputError(
            f"{path} is {other.width}x{other.height}, not "
            f"{reference.width}x{reference.height}; align the dates to one grid first"
        )
    if not other.transform.almost_equals(reference.transform):
        raise ToolInputError(f"{path} is not on the reference raster's pixel grid")


def _inputs(session: ModelSession) -> str:
    """Return the graph's input name."""
    return session.encoder.get_inputs()[0].name


def _check_model(session: ModelSession, dates: Sequence[str], bands: Sequence[int] | None) -> list[int]:
    """Validate the dates and bands against the catalog entry, and return indexes."""
    spec = session.spec
    if spec.scale != SCALE:
        raise ToolInputError(f"{spec.id} upscales {spec.scale}x, which this pipeline cannot run")
    if len(dates) + 1 != spec.dates:
        raise ToolInputError(
            f"{spec.id} takes {spec.dates} date(s); pass {spec.dates - 1} extra path(s) "
            "in dates, or choose a single-date model"
        )
    expected = len(spec.input_bands)
    indexes = list(bands) if bands else list(range(1, expected + 1))
    if len(indexes) != expected:
        raise ToolInputError(
            f"{spec.id} expects {expected} bands per date in this order: "
            f"{', '.join(spec.input_bands)}; got {len(indexes)}"
        )
    if any(index < 1 for index in indexes):
        raise ToolInputError("bands are 1-based")
    return indexes


def super_resolve_raster(
    workspace: Workspace,
    path: str,
    session: ModelSession,
    *,
    dates: Sequence[str] = (),
    bands: Sequence[int] | None = None,
    bounds: Sequence[float] | None = None,
    tile_size: int = DEFAULT_TILE,
    halo: int = DEFAULT_HALO,
    max_tiles: int = MAX_TILES,
    out: str = DEFAULT_OUT,
    job: Job | None = None,
) -> tuple[Path, Timings, dict]:
    """Super-resolve a raster 4x and return the written path, timings, and geometry."""
    spec = session.spec
    paths = [path, *dates]
    indexes = _check_model(session, dates, bands)
    halo = max(0, int(halo))
    timings = Timings()

    with ExitStack() as stack:
        datasets = [stack.enter_context(open_source(workspace, name)) for name in paths]
        reference = datasets[0]
        for name, dataset in zip(paths[1:], datasets[1:], strict=True):
            _same_grid(reference, dataset, name)
        for name, dataset in zip(paths, datasets, strict=True):
            if max(indexes) > dataset.count:
                raise ToolInputError(
                    f"{name} has {dataset.count} band(s); bands {indexes} were requested"
                )

        window = read_window(reference, bounds)
        cores = plan_cores(int(window.width), int(window.height), tile_size)
        if len(cores) > max_tiles:
            raise ToolInputError(
                f"this would run {len(cores)} tiles (limit {max_tiles}); pass bounds, "
                "clip the area first, or raise max_tiles"
            )

        target = workspace.resolve(out, write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        out_width = int(window.width) * SCALE
        out_height = int(window.height) * SCALE
        transform = window_transform(window, reference.transform) @ Affine.scale(1 / SCALE)
        profile = {
            "driver": "GTiff",
            "width": out_width,
            "height": out_height,
            "count": 3,
            "dtype": "uint8",
            "crs": reference.crs,
            "transform": transform,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "predictor": 2,
            "BIGTIFF": "IF_SAFER",
        }
        bounds_out = window_bounds(window, reference.transform)
        pixel_size = (abs(reference.transform.a) / SCALE, abs(reference.transform.e) / SCALE)
        crs_out = str(reference.crs)

        input_name = _inputs(session)
        if job is not None:
            job.progress(total=float(len(cores)))
        with rasterio.open(target, "w", **profile) as output:
            for index, (x, y, width, height) in enumerate(cores):
                if job is not None:
                    job.advance(detail=f"tile {index + 1}/{len(cores)}")
                source = Window(window.col_off + x, window.row_off + y, width, height)

                started = time.perf_counter()
                stacked, pad_top, pad_left = read_stack(datasets, source, indexes, halo)
                timings.read += time.perf_counter() - started
                timings.tiles += 1

                core = stacked[:, pad_top : pad_top + height, pad_left : pad_left + width]
                if core.size == 0 or not np.any(core):
                    timings.skipped_tiles += 1
                    continue

                started = time.perf_counter()
                result = session.encoder.run(None, {input_name: stacked[np.newaxis]})[0]
                timings.infer += time.perf_counter() - started
                expected = (1, 3, stacked.shape[1] * SCALE, stacked.shape[2] * SCALE)
                if result.shape != expected:
                    raise ToolInputError(
                        f"{spec.id} returned {result.shape}; expected {expected}"
                    )

                started = time.perf_counter()
                image = np.clip(result[0], 0.0, 1.0)
                image = (image * 255.0 + 0.5).astype(np.uint8)
                image = image[
                    :,
                    pad_top * SCALE : (pad_top + height) * SCALE,
                    pad_left * SCALE : (pad_left + width) * SCALE,
                ]
                output.write(
                    image,
                    window=Window(x * SCALE, y * SCALE, width * SCALE, height * SCALE),
                )
                timings.write += time.perf_counter() - started

            levels = [level for level in (2, 4, 8, 16) if max(out_width, out_height) // level >= 1]
            if levels:
                output.build_overviews(levels, Resampling.average)
                output.update_tags(ns="rio_overview", resampling="average")

    info = {
        "crs": crs_out,
        "bounds": [round(float(value), 6) for value in bounds_out],
        "source_pixels": [int(window.width), int(window.height)],
        "output_pixels": [out_width, out_height],
        "pixel_size": [round(float(pixel_size[0]), 6), round(float(pixel_size[1]), 6)],
    }
    return target, timings, info


__all__ = [
    "DEFAULT_HALO",
    "DEFAULT_OUT",
    "DEFAULT_TILE",
    "MAX_TILES",
    "SCALE",
    "Timings",
    "normalize",
    "plan_cores",
    "read_stack",
    "super_resolve_raster",
]
