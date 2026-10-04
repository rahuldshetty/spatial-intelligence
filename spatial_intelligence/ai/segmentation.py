"""SAM-family promptable segmentation, run locally through ONNX Runtime.

One pipeline, two exported sessions: an image encoder that turns a canvas into
embeddings, and a prompt decoder that turns embeddings plus points (or boxes
encoded as two labelled points) into masks. The split is what makes the model
cheap to use — the encoder runs once per canvas, and every extra prompt is a
decoder call against embeddings it already has.

Large rasters are segmented in tiles. A SAM-family encoder only ever sees one
canvas, so a 10 000-pixel scene cannot be pushed through it whole; tiles are cut
with an overlap so objects on a seam are seen whole at least once, and each
feature is kept by the tile whose *core* (the overlap-free middle) contains it,
which keeps a seam object from being reported twice without any cross-tile
matching.

Geometry comes back in the raster's own CRS: masks are mapped from the decoder's
low-resolution grid through the tile transform, so a polygon lands on the
ground rather than in pixel space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.features import shapes
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from ..contracts.errors import ToolInputError
from ..contracts.progress import Job
from ..workspace import Workspace
from .manager import ModelSession
from .tiles import (
    DEFAULT_OVERLAP,
    DEFAULT_TILE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    MAX_TILES,
    Tile,
    Timings,
    axis_boundaries,
    axis_starts,
    open_source,
    plan_tiles,
    read_rgb,
    read_window,
    stretch_to_uint8,
)

#: Points sampled per side in automatic mode.
DEFAULT_POINTS_PER_SIDE = 16
#: Prompts decoded per batch. The decoder materializes three 256x256 masks per
#: prompt, so a batch of 64 costs ~50 MB and a full grid costs a few seconds.
DECODE_BATCH = 64

#: Automatic-mode filters, as SAM's own mask generator uses them.
DEFAULT_IOU_THRESHOLD = 0.85
DEFAULT_STABILITY_THRESHOLD = 0.9
#: Box IoU above which two masks from different prompts are the same object.
NMS_IOU = 0.7


@dataclass(slots=True)
class Mask:
    """One decoded mask at the decoder's resolution, with its scores."""

    mask: np.ndarray
    score: float
    stability: float
    box: tuple[float, float, float, float]


# -- preprocessing -------------------------------------------------------


def preprocess(rgb: np.ndarray, canvas: int) -> tuple[np.ndarray, float]:
    """Return ``(pixel_values, scale)`` for an RGB tile.

    The longest edge is resized to ``canvas`` and the result is padded at the
    top-left, exactly as the reference ``SamImageProcessor`` does — so
    ``scale`` is the only number needed to map a tile pixel onto the canvas,
    and back again.
    """
    from PIL import Image

    height, width = rgb.shape[:2]
    if height == 0 or width == 0:
        raise ToolInputError("cannot segment an empty window")
    scale = canvas / max(height, width)
    resized_width = max(1, min(canvas, round(width * scale)))
    resized_height = max(1, min(canvas, round(height * scale)))

    resized = np.asarray(
        Image.fromarray(rgb).resize((resized_width, resized_height), Image.BILINEAR),
        dtype=np.float32,
    )
    padded = np.zeros((canvas, canvas, 3), dtype=np.float32)
    padded[:resized_height, :resized_width] = resized / 255.0
    normalized = (padded - IMAGE_MEAN) / IMAGE_STD
    return normalized.transpose(2, 0, 1)[np.newaxis], scale


# -- inference -----------------------------------------------------------


def encode(session: ModelSession, pixel_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the image encoder once for a canvas."""
    outputs = session.encoder.run(None, {"pixel_values": pixel_values})
    return outputs[0], outputs[1]


def decode(
    session: ModelSession,
    embeddings: tuple[np.ndarray, np.ndarray],
    points: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode prompts in batches; ``points`` is ``(P, N, 2)`` in canvas pixels.

    A box prompt is two points labelled 2 and 3, which is how the prompt
    encoder the decoder carries represents a rectangle.
    """
    image_embeddings, positional = embeddings
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for start in range(0, len(points), DECODE_BATCH):
        chunk = slice(start, start + DECODE_BATCH)
        batch_points = points[chunk][np.newaxis].astype(np.float32)
        batch_labels = labels[chunk][np.newaxis].astype(np.int64)
        iou, pred = session.decoder.run(
            None,
            {
                "input_points": batch_points,
                "input_labels": batch_labels,
                "image_embeddings": image_embeddings,
                "image_positional_embeddings": positional,
            },
        )
        scores.append(iou[0])
        masks.append(pred[0])
    return np.concatenate(scores), np.concatenate(masks)


def stability_score(logits: np.ndarray) -> float:
    """Return how stable a mask is: the IoU of its >=1 and >=-1 thresholds.

    This is the mask generator's own filter: a mask whose boundary does not move
    when the threshold is nudged is a mask the model is confident about, while
    a ragged edge scores low whatever its predicted IoU says.
    """
    high = logits > 1.0
    low = logits > -1.0
    union = np.count_nonzero(high | low)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(high & low) / union)


def grid_points(
    width: int, height: int, canvas: int, points_per_side: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return single-point prompts on a regular grid over the image area."""
    side = max(2, min(64, points_per_side))
    xs = (np.arange(side) + 0.5) * (width / side)
    ys = (np.arange(side) + 0.5) * (height / side)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 1, 2)
    labels = np.ones((grid.shape[0], 1), dtype=np.int64)
    return grid.astype(np.float32), labels


def box_nms(
    boxes: np.ndarray, scores: np.ndarray, threshold: float = NMS_IOU
) -> list[int]:
    """Return the indices kept by greedy non-maximum suppression over boxes."""
    if len(boxes) == 0:
        return []
    x0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y0 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    intersection = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    union = areas[:, None] + areas[None, :] - intersection
    iou = np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)

    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)
    for index in np.argsort(-scores, kind="stable"):
        if suppressed[index]:
            continue
        keep.append(int(index))
        suppressed |= iou[index] > threshold
    return keep


def auto_masks(
    session: ModelSession,
    embeddings: tuple[np.ndarray, np.ndarray],
    points: np.ndarray,
    labels: np.ndarray,
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
) -> tuple[list[Mask], int]:
    """Decode a point grid, skipping points an accepted mask already covers.

    This is the mask generator's own saving: on a scene of forest, field, and
    lake, most grid points land on something already found, and decoding them
    again costs a decoder run each and returns the same object a second time.
    Points are pruned between batches, and the surviving masks are suppressed
    against each other as they arrive, so both the work and the duplicates go
    away together. Returns the masks and how many prompts were actually run.
    """
    kept: list[Mask] = []
    covered: np.ndarray | None = None
    decoded = 0
    stride = session.spec.canvas / 4.0
    for start in range(0, len(points), DECODE_BATCH):
        stop = min(start + DECODE_BATCH, len(points))
        wanted = list(range(start, stop))
        if covered is not None:
            wanted = [
                index
                for index in wanted
                if not covered[
                    int(points[index, 0, 1] / stride), int(points[index, 0, 0] / stride)
                ]
            ]
        if not wanted:
            continue
        selection = np.asarray(wanted)
        decoded += len(selection)
        for mask in decode_best(session, embeddings, points[selection], labels[selection]):
            if mask.score < iou_threshold or mask.stability < stability_threshold:
                continue
            kept.append(mask)
        if kept:
            keep = box_nms(
                np.asarray([mask.box for mask in kept], dtype=np.float64),
                np.asarray([mask.score for mask in kept], dtype=np.float64),
            )
            kept = [kept[index] for index in keep]
            covered = np.zeros_like(kept[0].mask)
            for mask in kept:
                covered |= mask.mask
    return kept, decoded


def decode_best(
    session: ModelSession,
    embeddings: tuple[np.ndarray, np.ndarray],
    points: np.ndarray,
    labels: np.ndarray,
) -> list[Mask]:
    """Decode prompts and keep each prompt's best-scoring mask."""
    scores, masks = decode(session, embeddings, points, labels)
    best: list[Mask] = []
    for index in range(scores.shape[0]):
        choice = int(np.argmax(scores[index]))
        logits = masks[index][choice]
        binary = logits > 0.0
        if not binary.any():
            continue
        rows = np.flatnonzero(binary.any(axis=1))
        cols = np.flatnonzero(binary.any(axis=0))
        best.append(
            Mask(
                mask=binary,
                score=float(scores[index][choice]),
                stability=stability_score(logits),
                box=(
                    float(cols[0]),
                    float(rows[0]),
                    float(cols[-1]),
                    float(rows[-1]),
                ),
            )
        )
    return best


# -- mask geometry -------------------------------------------------------


@lru_cache(maxsize=16)
def _pixel_index(height: int, width: int, scale: float, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column index maps from the decoder grid onto a tile.

    With the tile already resized by ``scale`` and the mask grid a quarter of
    the canvas, one nearest-neighbour lookup is the exact inverse of the
    preprocessing; interpolating the mask instead would blur its boundary.
    """
    rows = np.clip(
        np.floor((np.arange(height) + 0.5) * scale / 4.0).astype(np.int32), 0, grid - 1
    )
    cols = np.clip(
        np.floor((np.arange(width) + 0.5) * scale / 4.0).astype(np.int32), 0, grid - 1
    )
    return rows, cols


def mask_to_tile(mask: np.ndarray, scale: float, height: int, width: int) -> np.ndarray:
    """Map a decoder mask onto the tile's pixel grid."""
    rows, cols = _pixel_index(height, width, float(scale), mask.shape[0])
    return mask[rows[:, np.newaxis], cols[np.newaxis, :]]


def polygonize(
    tile_mask: np.ndarray, transform: Any, *, min_area_px: int = 0
) -> list[Any]:
    """Return the polygons of a tile-resolution mask, in the raster's CRS."""
    if min_area_px and np.count_nonzero(tile_mask) < min_area_px:
        return []
    polygons: list[Any] = []
    for geometry, value in shapes(
        tile_mask.astype(np.uint8), mask=tile_mask, transform=transform
    ):
        if not value:
            continue
        polygon = shapely.geometry.shape(geometry)
        if polygon.is_empty:
            continue
        polygons.append(polygon)
    return polygons


# -- prompts -------------------------------------------------------------


def _to_canvas(
    coords: Sequence[Sequence[float]],
    dataset_transform: Any,
    offset: tuple[float, float],
    scale: float,
    src_crs: Any,
    prompt_crs: str | None,
) -> np.ndarray:
    """Convert prompt coordinates into canvas pixels for one tile.

    Coordinates arrive in ``prompt_crs`` (longitude/latitude by default) or in
    the raster's own CRS, become dataset pixels through the dataset transform,
    and then ``offset`` moves them into this tile before the resize ``scale``
    maps them onto the canvas the encoder saw.
    """
    values = np.asarray(coords, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if prompt_crs and src_crs and str(prompt_crs) != str(src_crs):
        from pyproj import Transformer

        transformer = Transformer.from_crs(prompt_crs, src_crs, always_xy=True)
        xs, ys = transformer.transform(values[:, 0], values[:, 1])
        values = np.column_stack([xs, ys])
    cols, rows = (~dataset_transform) @ (values[:, 0], values[:, 1])
    canvas = np.column_stack(
        [np.asarray(cols) - offset[0], np.asarray(rows) - offset[1]]
    ) * scale
    return canvas.astype(np.float32)


def _box_points(
    boxes: Sequence[Sequence[float]],
    dataset_transform: Any,
    offset: tuple[float, float],
    scale: float,
    src_crs: Any,
    prompt_crs: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn ``[x0, y0, x1, y1]`` boxes into two-point prompts (labels 2 and 3)."""
    corners = [corner for box in boxes for corner in ((box[0], box[1]), (box[2], box[3]))]
    canvas = _to_canvas(corners, dataset_transform, offset, scale, src_crs, prompt_crs)
    points = canvas.reshape(-1, 2, 2)
    labels = np.tile(np.array([2, 3], dtype=np.int64), (points.shape[0], 1))
    return points, labels


def _point_prompts(
    points: Sequence[Sequence[float]],
    dataset_transform: Any,
    offset: tuple[float, float],
    scale: float,
    src_crs: Any,
    prompt_crs: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn ``[x, y]`` prompts into single foreground points."""
    canvas = _to_canvas(points, dataset_transform, offset, scale, src_crs, prompt_crs)
    return canvas.reshape(-1, 1, 2), np.ones((canvas.shape[0], 1), dtype=np.int64)


# -- the task ------------------------------------------------------------


def segment_raster(
    workspace: Workspace,
    path: str,
    session: ModelSession,
    *,
    bands: Sequence[int] = (1, 2, 3),
    bounds: Sequence[float] | None = None,
    mode: str = "auto",
    points: Sequence[Sequence[float]] | None = None,
    boxes: Sequence[Sequence[float]] | None = None,
    prompt_crs: str | None = "EPSG:4326",
    points_per_side: int = DEFAULT_POINTS_PER_SIDE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
    min_area_px: int = 0,
    max_area_fraction: float = 0.8,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_tiles: int = MAX_TILES,
    job: Job | None = None,
) -> tuple[gpd.GeoDataFrame, Timings]:
    """Segment a raster into polygons and return them with their timings.

    ``mode`` is ``"auto"`` (a point grid over every tile), ``"points"``, or
    ``"boxes"``; prompt-driven runs only the tiles a prompt falls in.
    ``bounds`` restricts the read to ``[west, south, east, north]``, which is
    what keeps a large scene affordable when the caller has an AOI.
    ``max_area_fraction`` drops masks covering more than that share of a tile —
    they are the model's "all of this is one object" answer, and keeping them
    paints the tile seam into the output.
    """
    if mode not in {"auto", "points", "boxes"}:
        raise ToolInputError(f"unknown segmentation mode {mode!r}; use auto, points, or boxes")
    if mode == "points" and not points:
        raise ToolInputError("mode='points' needs at least one point")
    if mode == "boxes" and not boxes:
        raise ToolInputError("mode='boxes' needs at least one box")

    timings = Timings()
    dataset = open_source(workspace, path)

    with dataset:
        raster_crs = dataset.crs
        window = read_window(dataset, bounds)
        width, height = int(window.width), int(window.height)
        tile = max(256, tile_size)
        gap = max(0, min(overlap, tile // 2))
        xs = axis_starts(width, tile, gap)
        ys = axis_starts(height, tile, gap)
        x_edges = axis_boundaries(xs, tile, width)
        y_edges = axis_boundaries(ys, tile, height)
        windows = [
            (x, y, min(tile, width - x), min(tile, height - y))
            for y in ys
            for x in xs
        ]
        if len(windows) > max_tiles:
            raise ToolInputError(
                f"this would run {len(windows)} tiles (limit {max_tiles}); clip the area "
                "you need with clip, pass bounds, or raise max_tiles"
            )

        features: list[dict] = []
        if job is not None:
            job.progress(total=float(len(windows)))
        for index, (dx, dy, dw, dh) in enumerate(windows):
            if job is not None:
                job.advance(detail=f"tile {index + 1}/{len(windows)}")
            row_index, column_index = divmod(index, len(xs))
            source_window = Window(window.col_off + dx, window.row_off + dy, dw, dh)
            start = time.perf_counter()
            rgb = read_rgb(dataset, source_window, bands)
            timings.read += time.perf_counter() - start
            timings.tiles += 1
            if rgb is None:
                timings.skipped_tiles += 1
                continue

            transform = window_transform(source_window, dataset.transform)
            tile_data = Tile(
                rgb=rgb,
                transform=transform,
                # Keyed by source geometry, so the same tile re-read by a later
                # prompt call finds its embeddings already computed.
                key=f"{path}|{source_window.col_off:.0f},{source_window.row_off:.0f}|{dw}x{dh}",
                core=(
                    x_edges[column_index],
                    y_edges[row_index],
                    x_edges[column_index + 1],
                    y_edges[row_index + 1],
                ),
            )
            canvas = session.spec.canvas
            pixel_values, scale = preprocess(tile_data.rgb, canvas)

            embeddings = session.cached_embedding(tile_data.key)
            if embeddings is None:
                start = time.perf_counter()
                embeddings = encode(session, pixel_values)
                timings.encode += time.perf_counter() - start
                session.cache_embedding(tile_data.key, embeddings)

            prompts = _tile_prompts(
                tile_data,
                mode=mode,
                points=points,
                boxes=boxes,
                prompt_crs=prompt_crs,
                src_crs=raster_crs,
                dataset_transform=dataset.transform,
                offset=(source_window.col_off, source_window.row_off),
                canvas=canvas,
                scale=scale,
                points_per_side=points_per_side,
            )
            if prompts is None:
                continue
            prompt_points, prompt_labels = prompts

            start = time.perf_counter()
            if mode == "auto":
                masks, decoded = auto_masks(
                    session,
                    embeddings,
                    prompt_points,
                    prompt_labels,
                    iou_threshold=iou_threshold,
                    stability_threshold=stability_threshold,
                )
                timings.prompts += decoded
            else:
                masks = decode_best(session, embeddings, prompt_points, prompt_labels)
                timings.prompts += len(prompt_points)
            timings.decode += time.perf_counter() - start

            kept = [
                mask
                for mask in masks
                if mask.score >= iou_threshold and mask.stability >= stability_threshold
            ]
            keep = box_nms(
                np.asarray([mask.box for mask in kept], dtype=np.float64)
                if kept
                else np.empty((0, 4)),
                np.asarray([mask.score for mask in kept], dtype=np.float64) if kept else np.empty(0),
            )
            start = time.perf_counter()
            for position in keep:
                mask = kept[position]
                tile_mask = mask_to_tile(mask.mask, scale, dh, dw)
                area = int(np.count_nonzero(tile_mask))
                if area < min_area_px:
                    continue
                # A mask that covers the whole tile is the model saying "this is
                # all one thing" — forest, or a homogeneous field. Kept, it
                # draws the tile seam across the scene as a polygon edge.
                if area > max_area_fraction * float(dw * dh):
                    continue
                for polygon in polygonize(tile_mask, transform):
                    # The core test is what stops one object being reported by
                    # every tile that overlapped it: a feature belongs to the
                    # single tile whose overlap-free interior holds it.
                    point = polygon.representative_point()
                    column, row = (~dataset.transform) @ (point.x, point.y)
                    if not (
                        tile_data.core[0] <= column < tile_data.core[2]
                        and tile_data.core[1] <= row < tile_data.core[3]
                    ):
                        continue
                    features.append(
                        {
                            "geometry": polygon,
                            "score": round(float(mask.score), 4),
                            "stability": round(float(mask.stability), 4),
                            "area_px": area,
                        }
                    )
            timings.polygonize += time.perf_counter() - start

    frame = gpd.GeoDataFrame(features, geometry="geometry", crs=raster_crs) if features else (
        gpd.GeoDataFrame(
            {"geometry": [], "score": [], "stability": [], "area_px": []},
            geometry="geometry",
            crs=raster_crs,
        )
    )
    return frame, timings


def _tile_prompts(
    tile: Tile,
    *,
    mode: str,
    points: Sequence[Sequence[float]] | None,
    boxes: Sequence[Sequence[float]] | None,
    prompt_crs: str | None,
    src_crs: Any,
    dataset_transform: Any,
    offset: tuple[float, float],
    scale: float,
    canvas: int,
    points_per_side: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the prompts of one tile, or ``None`` when it has none.

    Prompt-driven calls only run the tiles their prompts fall in, so a handful
    of clicks does not pay for the whole raster.
    """
    height, width = tile.rgb.shape[:2]
    if mode == "auto":
        scaled_w = min(canvas, round(width * scale))
        scaled_h = min(canvas, round(height * scale))
        return grid_points(scaled_w, scaled_h, canvas, points_per_side)

    if mode == "boxes":
        prompt_points, prompt_labels = _box_points(
            boxes or [], dataset_transform, offset, scale, src_crs, prompt_crs
        )
    else:
        prompt_points, prompt_labels = _point_prompts(
            points or [], dataset_transform, offset, scale, src_crs, prompt_crs
        )

    limit_x, limit_y = width * scale, height * scale
    inside = [
        index
        for index, prompt in enumerate(prompt_points)
        if 0 <= prompt[:, 0].min()
        and prompt[:, 0].max() < limit_x
        and 0 <= prompt[:, 1].min()
        and prompt[:, 1].max() < limit_y
    ]
    if not inside:
        return None
    selected = np.asarray(inside)
    return prompt_points[selected], prompt_labels[selected]


__all__ = [
    "DEFAULT_OVERLAP",
    "DEFAULT_POINTS_PER_SIDE",
    "DEFAULT_TILE_SIZE",
    "MAX_TILES",
    "Mask",
    "Tile",
    "Timings",
    "axis_boundaries",
    "axis_starts",
    "box_nms",
    "decode",
    "decode_best",
    "encode",
    "grid_points",
    "mask_to_tile",
    "plan_tiles",
    "polygonize",
    "preprocess",
    "read_rgb",
    "read_window",
    "segment_raster",
    "stability_score",
    "stretch_to_uint8",
]
