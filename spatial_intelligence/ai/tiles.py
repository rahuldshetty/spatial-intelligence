"""Reading a raster in tiles: the part both tasks share.

Segmentation and detection are different models with different outputs, but
they agree on everything around the model: which windows to read, how to turn a
window into 8-bit RGB, and how to decide which tile owns an object that the
overlap showed twice. Keeping that here means a fix to the tiling rules lands
for both, and neither task carries a copy of the other's raster handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window

from ..contracts.errors import ToolInputError
from ..workspace import Workspace

#: ImageNet statistics the models here were trained with: RGB, rescaled to
#: 0-1, normalized per channel.
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Source pixels per tile, and the overlap between neighbours.
DEFAULT_TILE_SIZE = 1024
DEFAULT_OVERLAP = 128
#: Tiles one call may process. A full Sentinel-2 scene tiles into ~120 of them,
#: which is tens of minutes of CPU; the caller clips first or raises this.
MAX_TILES = 64


@dataclass(slots=True)
class Tile:
    """One canvas-sized read of a raster, with the geometry to place results."""

    rgb: np.ndarray
    transform: Any
    key: str
    core: tuple[float, float, float, float]


@dataclass(slots=True)
class Timings:
    """Seconds spent in each stage, for the tool's report."""

    read: float = 0.0
    encode: float = 0.0
    decode: float = 0.0
    polygonize: float = 0.0
    tiles: int = 0
    prompts: int = 0
    objects: int = 0
    skipped_tiles: int = 0

    def as_dict(self) -> dict:
        return {
            "tiles": self.tiles,
            "skipped_tiles": self.skipped_tiles,
            "prompts": self.prompts,
            "objects": self.objects,
            "seconds": {
                "read": round(self.read, 2),
                "encode": round(self.encode, 2),
                "decode": round(self.decode, 2),
                "polygonize": round(self.polygonize, 2),
            },
        }


def stretch_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Stretch a float tile to 8-bit RGB using its 2nd-98th percentiles.

    Model quality follows contrast, and a raw Sentinel or Landsat
    surface-reflectance tile is a narrow band of values; the stretch is
    per-band so a scene with one bright band does not flatten the others.
    """
    if rgb.dtype == np.uint8:
        return rgb
    out = np.empty(rgb.shape, dtype=np.uint8)
    for index in range(rgb.shape[2]):
        band = rgb[:, :, index].astype(np.float32)
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            out[:, :, index] = 0
            continue
        low, high = np.percentile(finite, (2, 98))
        if not math.isfinite(low) or not math.isfinite(high) or high - low < 1e-9:
            out[:, :, index] = np.clip(band, 0, 255).astype(np.uint8)
            continue
        scaled = (band - low) * (255.0 / (high - low))
        out[:, :, index] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def read_window(dataset: Any, bounds: Sequence[float] | None) -> Window:
    """Return the window to process: the whole raster, or the one ``bounds`` names."""
    full = Window(0, 0, dataset.width, dataset.height)
    if bounds is None:
        return full
    if len(bounds) != 4:
        raise ToolInputError("bounds must be [west, south, east, north]")
    from rasterio.windows import from_bounds

    window = from_bounds(*bounds, transform=dataset.transform).round_offsets().round_lengths()
    window = window.intersection(full)
    if window.width <= 0 or window.height <= 0:
        raise ToolInputError("bounds fall outside the raster")
    return window


def read_rgb(dataset: Any, window: Window, bands: Sequence[int]) -> np.ndarray | None:
    """Read a window as 8-bit RGB, or ``None`` when there is nothing there."""
    indexes = list(bands)
    if len(indexes) == 1:
        indexes = indexes * 3
    elif len(indexes) == 2:
        indexes = [indexes[0], indexes[1], indexes[1]]
    array = dataset.read(indexes=indexes, window=window, boundless=False)
    if array.size == 0:
        return None
    rgb = np.transpose(array, (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = stretch_to_uint8(rgb)
    if float(rgb.std()) < 1.0:
        return None  # blank or fully nodata: nothing to find, and running costs seconds
    return np.ascontiguousarray(rgb[:, :, :3])


def open_source(workspace: Workspace, path: str) -> Any:
    """Open a raster for reading: a workspace file, or a remote COG by URL."""
    if str(path).startswith(("http://", "https://")):
        return rasterio.open(path)
    return rasterio.open(workspace.resolve(path, must_exist=True))


def axis_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Return window starts covering ``length`` with as little overlap as possible.

    The windows are spread evenly rather than stepped: a length that does not
    divide cleanly would otherwise leave a sliver at the end, and the even
    spread overlaps neighbours by less than the requested amount instead of
    nearly doubling one window.
    """
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    windows = max(2, math.ceil((length - tile) / step) + 1)
    span = length - tile
    return [round(index * span / (windows - 1)) for index in range(windows)]


def axis_boundaries(starts: list[int], tile: int, length: int) -> list[float]:
    """Return the split points between windows, at the middle of each overlap."""
    boundaries = [0.0]
    for left, right in zip(starts, starts[1:]):
        boundaries.append((left + tile + right) / 2.0)
    boundaries.append(float(length))
    return boundaries


def plan_tiles(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Return ``(x, y, width, height)`` windows covering a raster.

    ``overlap`` is a floor, not a promise: the last window in an axis is where
    the remainder lands, so two neighbours may share more than it asks for.
    """
    tile = max(256, tile_size)
    overlap = max(0, min(overlap, tile // 2))
    return [
        (x, y, min(tile, width - x), min(tile, height - y))
        for y in axis_starts(height, tile, overlap)
        for x in axis_starts(width, tile, overlap)
    ]


__all__ = [
    "DEFAULT_OVERLAP",
    "DEFAULT_TILE_SIZE",
    "IMAGE_MEAN",
    "IMAGE_STD",
    "MAX_TILES",
    "Tile",
    "Timings",
    "axis_boundaries",
    "axis_starts",
    "open_source",
    "plan_tiles",
    "read_rgb",
    "read_window",
    "stretch_to_uint8",
]
