"""Object detection: find known objects in imagery with a local detector.

One problem, one tool, one file. Detection is not promptable — the model has a
fixed list of classes and answers with boxes — so its arguments are a class
filter and a confidence floor, and its output is labelled boxes rather than
polygons. Keeping it beside the segmentation tool would mean two arguments that
share nothing but the word "model".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...ai import detection
from ...ai.catalog import find as find_model
from ...ai.manager import get_manager
from ...contracts.effects import Effect
from ...contracts.errors import ToolInputError
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool
from .ai_common import publish, task_job


@pack(category="detection")
class DetectionPack:
    """Detection of known objects in rasters, on this machine."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._manager = get_manager(runtime)

    @tool(
        effects=frozenset({Effect.WORKSPACE_WRITE, Effect.NETWORK}),
        kind=ToolKind.REPORTING,
        timeout=1800,
    )
    def detect_objects(
        self,
        path: str,
        model_id: str = "yolos-tiny",
        classes: list[str] | None = None,
        confidence: float = detection.DEFAULT_CONFIDENCE,
        bounds: list[float] | None = None,
        bands: list[int] | None = None,
        max_side: int = detection.DEFAULT_MAX_SIDE,
        tile_size: int = 1024,
        max_tiles: int = 64,
        keep_crs: bool = False,
        out: str = "results/detections.geojson",
    ) -> dict:
        """Detect known objects in a raster as labelled boxes, with a local model.

        The default model is yolos-tiny, a COCO-80 detector: it knows person,
        bicycle, car, motorcycle, airplane, bus, train, truck, boat, and the
        indoor COCO set — and it knows nothing about trees, buildings, or land
        cover, which is what segment_image is for. It is also weak on 10 m
        satellite pixels, where a car is one dark pixel: it earns its keep on
        aerial and drone imagery, where vehicles, boats and aircraft are real
        objects. ai_models lists what is downloaded (yolos-tiny pulls ~26 MB).

        classes filters the output to those names, e.g. ["car", "truck", "boat"];
        leave it out to keep every class above confidence. confidence is the
        score floor: 0.5 is the DETR family's reference default, and this model's
        real objects sit far above it (a bus scores 0.98) while its inventions
        land between 0.4 and 0.9 — raise it to 0.9 when precision matters, drop
        it to 0.3 when you would rather see everything and filter later.
        max_side caps the long edge of each canvas the model sees: the reference
        is 1333, 800 is the default here, and a smaller value is faster and
        blinder. bounds and tile_size behave as in segment_image.

        Writes GeoJSON under results/ (WGS84 unless keep_crs) with one box per
        detection — label, class_id, score, box_px — and returns its path, the
        count per label, the score range and per-stage seconds. found=0 without
        writing when nothing passes. Call add_geojson afterwards to show it, and
        segment_image when the thing you want has no COCO name.
        """
        spec = find_model(model_id)
        if not spec.labels:
            raise ToolInputError(
                f"{spec.id} is a {spec.task} model with no classes, not a detector; "
                "ai_models lists which model serves which task"
            )

        with task_job(self._rt, self._manager, model_id, f"detect {out}") as (job, session):
            frame, timings = detection.detect_raster(
                self._rt.workspace,
                path,
                session,
                bands=bands or (1, 2, 3),
                bounds=bounds,
                classes=classes,
                confidence=confidence,
                max_side=max_side,
                tile_size=tile_size,
                max_tiles=max_tiles,
                job=job,
            )

            summary: dict[str, Any] = {
                "model": spec.id,
                "found": len(frame),
                "timings": timings.as_dict(),
            }
            if len(frame) == 0:
                summary["message"] = (
                    "no detection passed the confidence threshold; lower confidence, "
                    "raise max_side, or drop classes to see everything the model offers"
                )
                return summary

            counts = frame["label"].value_counts().to_dict()
            summary["counts"] = {str(label): int(count) for label, count in counts.items()}
            summary.update(publish(self._rt, frame, out=out, keep_crs=keep_crs))
            job.done(artifact=self._rt.workspace.relative(Path(summary["path"])))
            return summary


__all__ = ["DetectionPack"]
