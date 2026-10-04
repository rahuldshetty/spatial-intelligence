"""What a model-backed task tool does besides run its model.

Segmentation and detection differ in their pipelines and their arguments, but
both have to open a progress job, hold the model resident for the whole run,
write the result layer into the workspace, and report the same handful of
numbers back. That shared part lives here, once, so neither task pack owns a
copy of it and a fix to the artifact handling lands for both.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import geopandas as gpd

from ...ai.manager import ModelManager, ModelSession
from ...contracts.progress import Job
from ..runtime import ToolRuntime


def advance(job: Job):
    """Return a progress callback that reports downloaded bytes into ``job``."""

    def report(done: float, total: float, label: str) -> None:
        job.progress(done, total=total, detail=label)

    return report


@contextmanager
def task_job(
    runtime: ToolRuntime,
    manager: ModelManager,
    model_id: str,
    label: str,
) -> Iterator[tuple[Job, ModelSession]]:
    """Open a progress job and hold ``model_id`` loaded for the block.

    The job covers the download the manager may have to make and the tiles the
    caller then reports into it, and the reservation is what guarantees the model
    is not evicted halfway through a run for being over the residency cap.
    """
    job = runtime.reporter.job("ai", label, unit="tiles")
    with job:
        with manager.reserve(model_id, progress=advance(job)) as session:
            yield job, session


def publish(
    runtime: ToolRuntime,
    frame: gpd.GeoDataFrame,
    *,
    out: str,
    keep_crs: bool,
) -> dict:
    """Write a result layer for the map and return its summary.

    GeoJSON goes out in WGS84 unless the caller asked to keep the raster's CRS,
    which is what the map expects and what every other layer tool produces.
    """
    source_crs = str(frame.crs)
    if not keep_crs:
        frame = frame.to_crs("EPSG:4326")
    # Imported here, not at module scope: geo.vector reaches the map layer, which
    # imports the tool packages (the same reason geo.raster defers).
    from ...geo.vector import _write

    written = _write(runtime.workspace, out, frame)
    return {
        "path": runtime.record_artifact(written),
        "crs": str(frame.crs),
        "source_crs": source_crs,
        "bounds": [round(float(value), 6) for value in frame.total_bounds],
        "score": {
            "min": round(float(frame["score"].min()), 4),
            "max": round(float(frame["score"].max()), 4),
        },
    }


__all__ = ["advance", "publish", "task_job"]
