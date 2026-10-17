"""Turn a captured canvas or a workspace image into a model-ready JPEG.

The map capture is a PNG, often several megabytes; vision endpoints and the
conversation history both prefer a bounded JPEG. Rasters get the same treatment
after a percentile stretch so a 16-bit scene is legible.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts.errors import ToolInputError

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"})
RASTER_SUFFIXES = frozenset({".tif", ".tiff", ".img", ".vrt", ".jp2"})
JPEG_QUALITY = 85


def encode_jpeg(image: Image.Image, max_side: int) -> tuple[bytes, int, int]:
    """Downscale to ``max_side`` and return JPEG bytes plus final dimensions."""
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue(), image.width, image.height


def encode_capture(data: bytes, max_side: int) -> tuple[bytes, int, int]:
    """Decode a captured PNG and return the bounded JPEG."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            return encode_jpeg(image, max_side)
    except Exception as exc:  # noqa: BLE001 - malformed browser payload
        raise ToolInputError(f"could not read the map capture: {exc}") from exc


def _from_raster(
    path: Path,
    max_side: int,
    bands: list[int] | None = None,
) -> tuple[Image.Image, int, int, list[int]]:
    """Read a raster (selected bands, default first three) for viewing."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as dataset:
        original_width, original_height = dataset.width, dataset.height
        if bands:
            indices = [int(value) for value in bands]
            if len(indices) not in (1, 2, 3):
                raise ToolInputError("bands must contain one, two, or three band numbers")
            if any(value < 1 or value > dataset.count for value in indices):
                raise ToolInputError(
                    f"bands must be between 1 and {dataset.count} for {path.name}"
                )
        else:
            indices = list(range(1, min(dataset.count, 3) + 1))
        if not indices:
            raise ToolInputError(f"{path.name} has no raster bands")
        scale = max(1, int(np.ceil(max(original_width, original_height) / max_side)))
        height = max(1, original_height // scale)
        width = max(1, original_width // scale)
        count = len(indices)
        pixels = dataset.read(
            indices,
            out_shape=(count, height, width),
            resampling=Resampling.average,
        )
    array = np.moveaxis(pixels, 0, -1).astype(np.float32)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 2:
        array = np.repeat(array[:, :, :1], 3, axis=2)
    stretched = np.empty_like(array)
    for index in range(3):
        band = array[:, :, index]
        low, high = np.nanpercentile(band, (2, 98))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            stretched[:, :, index] = 0
        else:
            stretched[:, :, index] = np.clip((band - low) / (high - low), 0, 1) * 255
    return (
        Image.fromarray(stretched.astype(np.uint8), mode="RGB"),
        original_width,
        original_height,
        indices,
    )


def load_image(
    path: Path,
    max_side: int,
    bands: list[int] | None = None,
) -> tuple[bytes, int, int, int, int, list[int]]:
    """Return JPEG bytes, displayed size, source size, and selected bands."""
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ToolInputError(
            f"unsupported image type {suffix or '(none)'}; use PNG, JPEG, WebP, or a GeoTIFF"
        )
    if suffix in RASTER_SUFFIXES:
        image, original_width, original_height, selected = _from_raster(
            path, max_side, bands
        )
    else:
        if bands:
            raise ToolInputError("bands only applies to multiband rasters, not photos")
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                original_width, original_height = source.size
        except Exception as exc:  # noqa: BLE001 - surfaces as a tool failure
            raise ToolInputError(f"could not read {path.name}: {exc}") from exc
        selected = [1, 2, 3]
    data, width, height = encode_jpeg(image, max_side)
    return data, width, height, original_width, original_height, selected


__all__ = [
    "IMAGE_SUFFIXES",
    "JPEG_QUALITY",
    "RASTER_SUFFIXES",
    "encode_capture",
    "encode_jpeg",
    "load_image",
]
