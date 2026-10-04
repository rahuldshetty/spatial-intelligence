"""Segmentation: turn a raster into object polygons with a local SAM model.

One problem, one tool, one file. Segmentation is promptable — it answers "what
is this thing, and where does it end" for anything the caller can point at or
let the model find — which is why its arguments are prompts and thresholds and
its output is polygons with scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ...ai import segmentation
from ...ai.catalog import find as find_model
from ...ai.manager import get_manager
from ...contracts.effects import Effect
from ...contracts.errors import ToolInputError
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool
from .ai_common import publish, task_job


@pack(category="segmentation")
class SegmentationPack:
    """Promptable segmentation of rasters, on this machine."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._manager = get_manager(runtime)

    @tool(
        effects=frozenset({Effect.WORKSPACE_WRITE, Effect.NETWORK}),
        kind=ToolKind.REPORTING,
        timeout=1800,
    )
    def segment_image(
        self,
        path: str,
        model_id: str = "slimsam-77",
        mode: Literal["auto", "points", "boxes"] = "auto",
        points: list[list[float]] | None = None,
        boxes: list[list[float]] | None = None,
        prompt_crs: str = "EPSG:4326",
        bounds: list[float] | None = None,
        bands: list[int] | None = None,
        points_per_side: int = 16,
        iou_threshold: float = 0.85,
        min_area_px: int = 64,
        keep_crs: bool = False,
        tile_size: int = 1024,
        max_tiles: int = 64,
        out: str = "results/segments.geojson",
    ) -> dict:
        """Segment a raster into object polygons with a local SAM-family model.

        mode="auto" is the default and needs no prompt: it tiles a point grid
        over the image (points_per_side, 16 by default) and returns every object
        the model finds — fields, water, buildings, cloud. mode="points" takes
        points=[[x, y], ...] and mode="boxes" takes boxes=[[x0, y0, x1, y1], ...];
        both are in longitude/latitude unless prompt_crs names the raster's own
        CRS, and only the tiles a prompt falls in are run.

        bounds=[west, south, east, north] limits the work to an area of a large
        raster, which is much cheaper than clipping first. bands picks the three
        bands to segment (default 1,2,3 when present; a single band is treated as
        grayscale). iou_threshold and min_area_px trade completeness for
        precision: lower the threshold or the area to find more, raise them to
        drop weak and speck-sized masks. Tiles beyond max_tiles are refused
        rather than run for an hour; raise it or pass bounds for a piece.

        Writes GeoJSON under results/ (WGS84 unless keep_crs) and returns its
        path, the feature count, the score range, and per-stage seconds. It
        returns found=0 without writing when nothing passes the thresholds.
        Call add_geojson afterwards to show the layer on the map.
        """
        spec = find_model(model_id)
        if spec.task != "segmentation":
            raise ToolInputError(
                f"{spec.id} is a {spec.task} model, not a segmenter; ai_models lists "
                "which model serves which task"
            )

        with task_job(self._rt, self._manager, model_id, f"segment {out}") as (job, session):
            frame, timings = segmentation.segment_raster(
                self._rt.workspace,
                path,
                session,
                bands=bands or (1, 2, 3),
                bounds=bounds,
                mode=mode,
                points=points,
                boxes=boxes,
                prompt_crs=prompt_crs,
                points_per_side=points_per_side,
                iou_threshold=iou_threshold,
                min_area_px=min_area_px,
                tile_size=tile_size,
                max_tiles=max_tiles,
                job=job,
            )

            summary: dict = {
                "model": spec.id,
                "mode": mode,
                "found": len(frame),
                "timings": timings.as_dict(),
            }
            if len(frame) == 0:
                summary["message"] = (
                    "no objects passed the thresholds; lower iou_threshold or "
                    "min_area_px, raise points_per_side, or run mode='points' on "
                    "a known object"
                )
                return summary

            summary.update(publish(self._rt, frame, out=out, keep_crs=keep_crs))
            job.done(artifact=self._rt.workspace.relative(Path(summary["path"])))
            return summary


__all__ = ["SegmentationPack"]
