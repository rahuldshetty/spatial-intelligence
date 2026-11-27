"""OpenSR SEN2-SR Sentinel-2 super-resolution, tiled through the official package.

The official model takes raw reflectance DN in B04/B03/B02/B08 order at 10 m and
returns the same four bands at 4x. The package applies measurement-consistency
back-projection inside ``predict_step``; this module only reads the co-registered
tiles, runs the model, and writes a four-band COG.
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
TILE = 128
HALO = 16
MAX_TILES = 64
DEFAULT_OUT = "results/superres.tif"
EXPECTED_PIXEL_SIZE = 10.0
PIXEL_SIZE_TOLERANCE = 0.5
DN_MAX = 10_000.0


@dataclass(slots=True)
class Timings:
    """Seconds spent reading, running the model, and writing."""

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


def plan_cores(width: int, height: int, tile: int = TILE) -> list[tuple[int, int, int, int]]:
    """Return non-overlapping ``(x, y, width, height)`` cores covering a raster."""
    tile = max(32, int(tile))
    return [
        (x, y, min(tile, width - x), min(tile, height - y))
        for y in range(0, height, tile)
        for x in range(0, width, tile)
    ]


def _check_grid(dataset: Any, path: str) -> None:
    """Refuse a raster that is not a projected 10 m Sentinel-2 grid."""
    if dataset.crs is None:
        raise ToolInputError(f"{path} has no CRS; SEN2-SR needs a georeferenced raster")
    if not dataset.crs.is_projected:
        raise ToolInputError(
            f"{path} is {dataset.crs}; SEN2-SR expects a projected 10 m Sentinel-2 grid"
        )
    x_size, y_size = abs(dataset.transform.a), abs(dataset.transform.e)
    if (
        abs(x_size - EXPECTED_PIXEL_SIZE) > PIXEL_SIZE_TOLERANCE
        or abs(y_size - EXPECTED_PIXEL_SIZE) > PIXEL_SIZE_TOLERANCE
    ):
        raise ToolInputError(
            f"{path} is {x_size:.1f} x {y_size:.1f} m; SEN2-SR expects 10 m bands. "
            "Resample the B04/B03/B02/B08 stack to 10 m first."
        )


def _select_bands(
    spec: Any, dataset: Any, bands: Sequence[int] | None, path: str
) -> list[int]:
    """Validate the four-band raw-DN contract and return indexes."""
    expected = len(spec.input_bands)
    if bands is None:
        if dataset.count != expected:
            raise ToolInputError(
                f"{path} has {dataset.count} band(s); {spec.id} takes {expected}: "
                f"{', '.join(spec.input_bands)}. Pass bands=[...] in that order."
            )
        indexes = list(range(1, expected + 1))
    else:
        indexes = list(bands)
        if len(indexes) != expected:
            raise ToolInputError(
                f"{spec.id} takes {expected} bands in this order: "
                f"{', '.join(spec.input_bands)}; got {len(indexes)}"
            )
        if any(index < 1 for index in indexes):
            raise ToolInputError("bands are 1-based")
        if max(indexes) > dataset.count:
            raise ToolInputError(
                f"{path} has {dataset.count} band(s); bands {indexes} were requested"
            )
    dtypes = {dataset.dtypes[index - 1] for index in indexes}
    if dtypes != {"uint16"}:
        raise ToolInputError(
            f"{path} has {sorted(dtypes)} bands; {spec.id} expects raw uint16 "
            "reflectance DN (B04/B03/B02/B08), not 8-bit TCI or float."
        )
    return indexes


def read_tile(
    dataset: Any,
    window: Window,
    bands: Sequence[int],
    halo: int = HALO,
) -> tuple[np.ndarray, int, int]:
    """Read one core with a halo, edge-padded to ``(4, h+2*halo, w+2*halo)``."""
    desired_left = int(window.col_off) - halo
    desired_top = int(window.row_off) - halo
    desired_width = int(window.width) + 2 * halo
    desired_height = int(window.height) + 2 * halo
    left = max(0, desired_left)
    top = max(0, desired_top)
    right = min(dataset.width, desired_left + desired_width)
    bottom = min(dataset.height, desired_top + desired_height)
    inner = Window(left, top, right - left, bottom - top)
    array = dataset.read(indexes=list(bands), window=inner).astype(np.float32)
    pads = (
        (0, 0),
        (top - desired_top, (desired_top + desired_height) - bottom),
        (left - desired_left, (desired_left + desired_width) - right),
    )
    try:
        array = np.pad(array, pads, mode="reflect")
    except ValueError:
        array = np.pad(array, pads, mode="edge")
    return np.ascontiguousarray(array), int(window.height), int(window.width)


def super_resolve_raster(
    workspace: Workspace,
    path: str,
    session: ModelSession,
    *,
    bands: Sequence[int] | None = None,
    bounds: Sequence[float] | None = None,
    max_tiles: int = MAX_TILES,
    out: str = DEFAULT_OUT,
    job: Job | None = None,
) -> tuple[Path, Timings, dict]:
    """Super-resolve a four-band Sentinel-2 stack 4x with the official SEN2-SR."""
    spec = session.spec
    if spec.scale != SCALE or spec.input_kind != "reflectance" or spec.output_bands != 4:
        raise ToolInputError(f"{spec.id} is not a 4-band Sentinel-2 super-resolution model")
    try:
        import torch
    except ImportError as exc:  # noqa: BLE001 - one actionable message
        raise ToolInputError(
            "torch is missing from this install, so SEN2-SR cannot run; "
            "reinstall spatial-intelligence (it ships opensr-srgan)"
        ) from exc
    timings = Timings()

    with ExitStack() as stack:
        dataset = stack.enter_context(open_source(workspace, path))
        _check_grid(dataset, path)
        indexes = _select_bands(spec, dataset, bands, path)

        window = read_window(dataset, bounds)
        cores = plan_cores(int(window.width), int(window.height))
        if len(cores) > max_tiles:
            raise ToolInputError(
                f"this would run {len(cores)} tiles (limit {max_tiles}); pass bounds, "
                "clip the area first, or raise max_tiles"
            )

        target = workspace.resolve(out, write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        out_width = int(window.width) * SCALE
        out_height = int(window.height) * SCALE
        transform = window_transform(window, dataset.transform) @ Affine.scale(1 / SCALE)
        profile = {
            "driver": "GTiff",
            "width": out_width,
            "height": out_height,
            "count": spec.output_bands,
            "dtype": "uint16",
            "crs": dataset.crs,
            "transform": transform,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            # Match the readable input COG: band interleave and a nodata value.
            # Pixel-interleaved 4-band uint16 with a predictor is rejected by
            # GeoLibre's browser GeoTIFF reader even though rasterio reads it.
            "interleave": "band",
            "nodata": 0,
            "BIGTIFF": "IF_SAFER",
        }
        bounds_out = window_bounds(window, dataset.transform)
        pixel_size = (abs(dataset.transform.a) / SCALE, abs(dataset.transform.e) / SCALE)
        crs_out = str(dataset.crs)

        if job is not None:
            job.progress(total=float(len(cores)))
        with rasterio.open(target, "w", **profile) as output:
            for index, (x, y, width, height) in enumerate(cores):
                if job is not None:
                    job.advance(detail=f"tile {index + 1}/{len(cores)}")
                source = Window(window.col_off + x, window.row_off + y, width, height)

                started = time.perf_counter()
                tile, valid_h, valid_w = read_tile(dataset, source, indexes)
                timings.read += time.perf_counter() - started
                timings.tiles += 1
                if not np.any(tile[:, HALO : HALO + valid_h, HALO : HALO + valid_w]):
                    timings.skipped_tiles += 1
                    continue

                started = time.perf_counter()
                with torch.inference_mode():
                    prediction = session.encoder.predict_step(
                        torch.from_numpy(tile[np.newaxis])
                    )
                timings.infer += time.perf_counter() - started
                result = prediction[0].detach().cpu().numpy()
                expected = (
                    spec.output_bands,
                    (valid_h + 2 * HALO) * SCALE,
                    (valid_w + 2 * HALO) * SCALE,
                )
                if result.shape != expected:
                    raise ToolInputError(
                        f"{spec.id} returned {result.shape}; expected {expected}"
                    )
                image = result[
                    :,
                    HALO * SCALE : (HALO + valid_h) * SCALE,
                    HALO * SCALE : (HALO + valid_w) * SCALE,
                ]
                image = (np.clip(image, 0.0, DN_MAX) + 0.5).astype(np.uint16)

                started = time.perf_counter()
                output.write(
                    image,
                    window=Window(
                        x * SCALE, y * SCALE, valid_w * SCALE, valid_h * SCALE
                    ),
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
        "scale": SCALE,
        "input_bands": list(spec.input_bands),
        "output_bands": list(spec.input_bands),
        "map_bands": [1, 2, 3],
    }
    return target, timings, info


__all__ = [
    "DEFAULT_OUT",
    "HALO",
    "MAX_TILES",
    "SCALE",
    "TILE",
    "Timings",
    "plan_cores",
    "read_tile",
    "super_resolve_raster",
]
