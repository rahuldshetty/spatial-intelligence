"""Local AI models: pull one once, then segment rasters in this workspace.

The tools stay thin — model residency is the manager's job
(:mod:`spatial_intelligence.ai.manager`), the pipeline is the segmentation
module's — and the docstrings carry everything the model needs to choose a
model, prompt it, and read the result.
"""

from __future__ import annotations

from typing import Any, Literal

from ...ai import models, segmentation
from ...ai.manager import get_manager
from ...ai.models import find as find_model
from ...contracts.effects import Effect
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool


@pack(category="ai")
class AIPack:
    """Local models for this workspace, run through ONNX Runtime on this machine."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._manager = get_manager(runtime)

    # -- catalog and memory -----------------------------------------------

    @tool(core=True, effects=frozenset({Effect.READ}))
    def ai_models(self, task: str = "") -> list[dict]:
        """List the local AI models, what is downloaded, and what is in memory.

        Each entry has id, task ("segmentation"), prompts it accepts
        (point/box/grid), license, bytes, downloaded, loaded, provider, and the
        seconds it has been idle. Filter with task when the request names one.
        Call this before segment_image to see whether a download is needed, and
        to check what is already costing memory.
        """
        return self._manager.status(task=task.strip() or None)

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}), kind=ToolKind.REPORTING)
    def ai_pull_model(self, model_id: str, verify: bool = False) -> dict:
        """Download a model into the local cache, once, and verify it.

        Files land under ``<GEOAI_HOME>/.models`` and are checked against the
        sha256 recorded in the catalog, so a rerun skips what is already there;
        set verify=True to re-hash the files on disk. Sizes: slimsam-77 is about
        40 MB. segment_image downloads on its own when a model is missing, so
        this is only needed to fetch ahead of time or to repair a bad file.
        """
        spec = find_model(model_id)
        job = self._rt.reporter.job(
            "download", f"{spec.id} ({spec.total_bytes / 1e6:.0f} MB)", unit="bytes",
            total=float(spec.total_bytes),
        )
        with job:
            result = self._manager.pull(spec, verify_hashes=verify, progress=_advance(job))
            job.done(artifact=str(models.spec_dir(spec)))
        result["downloaded_state"] = self._manager.status()
        return result

    @tool(effects=frozenset({Effect.READ}))
    def ai_unload_model(self, model_id: str = "") -> dict:
        """Free the memory a loaded model holds; an empty model_id unloads all.

        Loaded models stay resident until they are idle past the TTL, so call
        this after a heavy segmentation on a machine that needs the memory back.
        A model still in use is refused rather than pulled out from under a run.
        """
        return self._manager.unload(model_id)

    # -- the task ----------------------------------------------------------

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
        job = self._rt.reporter.job("ai", f"segment {out}", unit="tiles")
        with job:
            with self._manager.reserve(model_id, progress=_advance(job)) as session:
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

            summary: dict[str, Any] = {
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

            source_crs = str(frame.crs)
            if not keep_crs:
                frame = frame.to_crs("EPSG:4326")
            # Imported here, not at module scope: geo.vector reaches the map
            # layer, which imports this package (same reason geo.raster defers).
            from ...geo.vector import _write

            written = _write(self._rt.workspace, out, frame)
            summary.update(
                {
                    "path": self._rt.record_artifact(written),
                    "crs": str(frame.crs),
                    "source_crs": source_crs,
                    "bounds": [round(float(value), 6) for value in frame.total_bounds],
                    "score": {
                        "min": round(float(frame["score"].min()), 4),
                        "max": round(float(frame["score"].max()), 4),
                    },
                }
            )
            job.done(artifact=self._rt.workspace.relative(written))
            return summary


def _advance(job: Any):
    """Return a progress callback that reports downloaded bytes into ``job``."""

    def report(done: float, total: float, label: str) -> None:
        job.progress(done, total=total, detail=label)

    return report


__all__ = ["AIPack"]
