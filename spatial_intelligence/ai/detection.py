"""DETR-family object detection, run locally through ONNX Runtime.

One graph, one pass per canvas: the model takes an image, asks a fixed number of
queries, and answers each with a class and a box. There are no prompts — the
choice that matters is which answers to keep, which is what the confidence
threshold and the class filter are for.

The output format is why this is its own module. A DETR answers in normalized
centre-width-height against the canvas it saw, so every box needs the same
two-step mapping back — canvas to tile pixels, tile pixels to ground — and that
mapping is the one part of detection worth testing against a known box.

Tiling, tile ownership, and CRS handling are the segmentation pipeline's, from
:mod:`spatial_intelligence.ai.tiles`: both tasks are "run a model over a raster
too big to put through it in one go".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import geopandas as gpd
import numpy as np
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box as shapely_box

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
    Timings,
    axis_boundaries,
    axis_starts,
    open_source,
    read_rgb,
    read_window,
)

#: Box IoU above which two detections of the same class are one object.
NMS_IOU = 0.5
#: Score below which a detection is dropped. This is the DETR family's own
#: reference threshold; the model's real calibrations sit far above it (a bus is
#: 0.98) while its inventions land in the 0.4-0.9 band, so precision-hungry
#: callers raise it.
DEFAULT_CONFIDENCE = 0.5
#: Detections one call may return, best first.
MAX_DETECTIONS = 5000
#: Long edge of the canvas the model sees by default. The reference
#: preprocessing is 1333, thorough and slow on CPU; 800 is the same idea scaled
#: to a desktop, and the tool exposes it for a caller who wants either.
DEFAULT_MAX_SIDE = 800
#: Boxes on the resized canvas are rounded to this, because the ViT backbone
#: patches at 16 and a canvas that does not divide leaves a ragged edge.
PATCH = 16


@dataclass(slots=True)
class Detection:
    """One detected object, in tile pixels."""

    label: str
    class_id: int
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass(slots=True)
class Canvas:
    """The resized image the model saw: its size, and the scale to undo."""

    width: int
    height: int
    scale: float


def canvas_for(height: int, width: int, shortest_edge: int, longest_edge: int, max_side: int) -> Canvas:
    """Return the canvas a tile is resized onto.

    The shorter side is scaled to ``shortest_edge`` unless that would push the
    longer side past ``longest_edge``; ``max_side`` lowers both for a caller who
    prefers speed. The result is rounded down to a multiple of :data:`PATCH`.
    """
    shortest = min(shortest_edge, max(PATCH, shortest_edge * max_side // longest_edge))
    scale = min(shortest / min(height, width), max_side / max(height, width))
    return Canvas(
        width=max(PATCH, round(width * scale) // PATCH * PATCH),
        height=max(PATCH, round(height * scale) // PATCH * PATCH),
        scale=scale,
    )


def preprocess(rgb: np.ndarray, canvas: Canvas) -> np.ndarray:
    """Return the ``(1, 3, H, W)`` tensor the graph takes."""
    from PIL import Image

    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ToolInputError("cannot detect on an empty window")
    resized = np.asarray(
        Image.fromarray(rgb).resize((canvas.width, canvas.height), Image.BILINEAR),
        dtype=np.float32,
    )
    normalized = (resized / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return normalized.transpose(2, 0, 1)[np.newaxis]


def decode(
    outputs: Sequence[np.ndarray],
    *,
    confidence: float,
    labels: Sequence[str],
    classes: Sequence[str] | None = None,
) -> list[Detection]:
    """Turn a graph's raw output into detections on the resized canvas.

    The scores are a softmax over the class slots — verified against this
    export, not assumed: sigmoid, the other detector convention, puts every
    score under 0.1 on a picture of a bus, while softmax answers 0.998 for it.
    The graph's last slot is reserved for "no object", and a query that means
    nothing spends its probability there, so dropping that slot is what makes a
    query's best real class comparable to the threshold. Boxes stay normalized
    here; :func:`to_tile_pixels` gives them pixels.
    """
    if len(outputs) < 2:
        raise ToolInputError(
            f"the model returned {len(outputs)} outputs; detection needs logits and boxes"
        )
    logits, boxes = np.asarray(outputs[0]), np.asarray(outputs[1])
    if logits.ndim != 3 or boxes.shape[-1] != 4 or boxes.ndim != 3:
        raise ToolInputError(
            f"the model returned {logits.shape} and {boxes.shape}; this pipeline expects "
            "(batch, queries, classes) and (batch, queries, 4)"
        )
    logits, boxes = logits[0], boxes[0]
    if logits.shape[1] < len(labels):
        raise ToolInputError(
            f"the model predicts {logits.shape[1]} classes but the catalog lists "
            f"{len(labels)} labels for it"
        )

    scores = _softmax(logits)
    if scores.shape[1] > len(labels):
        scores = scores[:, : len(labels)]
    class_ids = np.argmax(scores, axis=1)
    best = scores[np.arange(scores.shape[0]), class_ids]

    wanted = {name.strip().lower() for name in classes or () if name.strip()}
    found: list[Detection] = []
    for index in np.argsort(-best, kind="stable"):
        score = float(best[index])
        if score < confidence:
            break
        label = labels[int(class_ids[index])]
        if label == "N/A" or (wanted and label.lower() not in wanted):
            continue
        center_x, center_y, box_width, box_height = (float(value) for value in boxes[index])
        found.append(
            Detection(
                label=label,
                class_id=int(class_ids[index]),
                score=score,
                box=(
                    center_x - box_width / 2,
                    center_y - box_height / 2,
                    center_x + box_width / 2,
                    center_y + box_height / 2,
                ),
            )
        )
        if len(found) >= MAX_DETECTIONS:
            break
    return found


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Return the row-wise softmax of ``logits``, shifted for stability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def to_tile_pixels(
    detections: Sequence[Detection], canvas: Canvas, width: int, height: int
) -> list[Detection]:
    """Map normalized boxes onto the tile's pixel grid, clamped to the tile.

    A box is normalized against the canvas the model saw, so it scales back by
    the canvas size and divides by the resize factor; clamping to the tile keeps
    a box from claiming ground the tile never covered.
    """
    mapped: list[Detection] = []
    for detection in detections:
        x0 = min(max(detection.box[0] * canvas.width / canvas.scale, 0.0), float(width))
        y0 = min(max(detection.box[1] * canvas.height / canvas.scale, 0.0), float(height))
        x1 = min(max(detection.box[2] * canvas.width / canvas.scale, 0.0), float(width))
        y1 = min(max(detection.box[3] * canvas.height / canvas.scale, 0.0), float(height))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue  # a box thinner than a pixel is not an object
        mapped.append(
            Detection(
                label=detection.label,
                class_id=detection.class_id,
                score=detection.score,
                box=(x0, y0, x1, y1),
            )
        )
    return mapped


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    """IoU of two ``(x0, y0, x1, y1)`` boxes."""
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def owned_by_core(column: float, row: float, core: tuple[float, float, float, float]) -> bool:
    """Whether a detection at ``(column, row)`` belongs to the tile with ``core``.

    Tiles overlap so an object on a seam is seen whole by at least one of them,
    and the cores — the overlap-free middle of each tile — partition the raster.
    A detection is kept by the single tile whose core holds its centre, which is
    what stops a seam object being reported twice without any cross-tile
    matching.
    """
    return core[0] <= column < core[2] and core[1] <= row < core[3]


def nms_indices(
    class_ids: Sequence[int],
    boxes: Sequence[tuple[float, float, float, float]],
    scores: Sequence[float],
    threshold: float = NMS_IOU,
) -> list[int]:
    """Return the indices kept by greedy non-maximum suppression, best first.

    Suppression is per class: a car beside a boat overlaps in image space but is
    not a duplicate of it, and collapsing the two would delete the one asked for.
    """
    order = sorted(range(len(boxes)), key=lambda index: -scores[index])
    kept: list[int] = []
    for index in order:
        if any(
            class_ids[other] == class_ids[index]
            and iou(boxes[other], boxes[index]) > threshold
            for other in kept
        ):
            continue
        kept.append(index)
    return kept


def box_nms(detections: Sequence[Detection], threshold: float = NMS_IOU) -> list[Detection]:
    """Return the detections kept by greedy non-maximum suppression."""
    keep = nms_indices(
        [item.class_id for item in detections],
        [item.box for item in detections],
        [item.score for item in detections],
        threshold,
    )
    return [detections[index] for index in keep]


def detect_raster(
    workspace: Workspace,
    path: str,
    session: ModelSession,
    *,
    bands: Sequence[int] = (1, 2, 3),
    bounds: Sequence[float] | None = None,
    classes: Sequence[str] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    max_side: int = DEFAULT_MAX_SIDE,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_tiles: int = MAX_TILES,
    job: Job | None = None,
) -> tuple[gpd.GeoDataFrame, Timings]:
    """Detect objects in a raster and return them with their timings."""
    spec = session.spec
    if not spec.labels:
        raise ToolInputError(f"{spec.id} carries no class labels; it is not a detection model")

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
            (x, y, min(tile, width - x), min(tile, height - y)) for y in ys for x in xs
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

            canvas = canvas_for(
                dh, dw, spec.shortest_edge, spec.longest_edge, max_side
            )
            start = time.perf_counter()
            outputs = session.encoder.run(None, {"pixel_values": preprocess(rgb, canvas)})
            timings.encode += time.perf_counter() - start

            start = time.perf_counter()
            found = box_nms(
                to_tile_pixels(
                    decode(
                        outputs,
                        confidence=confidence,
                        labels=spec.labels,
                        classes=classes,
                    ),
                    canvas,
                    dw,
                    dh,
                )
            )
            timings.decode += time.perf_counter() - start
            timings.objects += len(found)

            transform = window_transform(source_window, dataset.transform)
            core = (
                x_edges[column_index],
                y_edges[row_index],
                x_edges[column_index + 1],
                y_edges[row_index + 1],
            )
            for detection in found:
                x0, y0, x1, y1 = detection.box
                column = window.col_off + dx + (x0 + x1) / 2
                row = window.row_off + dy + (y0 + y1) / 2
                if not owned_by_core(column, row, core):
                    continue
                (gx0, gy0), (gx1, gy1) = transform @ (x0, y0), transform @ (x1, y1)
                ground = (
                    min(gx0, gx1),
                    min(gy0, gy1),
                    max(gx0, gx1),
                    max(gy0, gy1),
                )
                features.append(
                    {
                        "geometry": shapely_box(*ground),
                        "label": detection.label,
                        "class_id": detection.class_id,
                        "score": round(detection.score, 4),
                        "box_px": [round(value, 1) for value in detection.box],
                        "_ground": ground,
                    }
                )

    # Ownership decides which tile reports an object, but a tile that sees a
    # clipped object can put its centre just across the core boundary and report
    # the same object again. One suppression pass over the assembled boxes is
    # what a detector pipeline always ends with, and it is what removes those.
    if features:
        keep = nms_indices(
            [feature["class_id"] for feature in features],
            [feature["_ground"] for feature in features],
            [feature["score"] for feature in features],
        )
        features = [features[index] for index in keep]
        for feature in features:
            feature.pop("_ground")

    empty = {
        "geometry": [],
        "label": [],
        "class_id": [],
        "score": [],
        "box_px": [],
    }
    frame = (
        gpd.GeoDataFrame(features, geometry="geometry", crs=raster_crs)
        if features
        else gpd.GeoDataFrame(empty, geometry="geometry", crs=raster_crs)
    )
    return frame, timings


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MAX_SIDE",
    "MAX_DETECTIONS",
    "NMS_IOU",
    "Canvas",
    "Detection",
    "box_nms",
    "canvas_for",
    "decode",
    "detect_raster",
    "iou",
    "nms_indices",
    "owned_by_core",
    "preprocess",
    "to_tile_pixels",
]
