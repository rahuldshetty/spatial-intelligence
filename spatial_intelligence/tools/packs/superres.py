"""Super-resolution: upscale a Sentinel-2 10 m stack to 2.5 m with OpenSR SEN2-SR."""

from __future__ import annotations

from typing import Any

from ...ai import superres
from ...ai.catalog import find as find_model
from ...ai.manager import get_manager
from ...contracts.effects import Effect
from ...contracts.errors import ToolInputError
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool
from .ai_common import task_job


@pack(category="super-resolution")
class SuperResolutionPack:
    """4x upscaling of Sentinel-2 rasters, on this machine."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._manager = get_manager(runtime)

    @tool(
        effects=frozenset({Effect.WORKSPACE_WRITE, Effect.NETWORK}),
        kind=ToolKind.REPORTING,
        timeout=1800,
    )
    def super_resolve(
        self,
        path: str,
        model_id: str = "sen2-sr",
        bands: list[int] | None = None,
        bounds: list[float] | None = None,
        max_tiles: int = superres.MAX_TILES,
        out: str = superres.DEFAULT_OUT,
    ) -> dict:
        """Upscale a Sentinel-2 four-band 10 m stack to 2.5 m with OpenSR SEN2-SR.

        The model is Sentinel-2 specific and takes **B04, B03, B02, B08** raw
        reflectance DN at 10 m: uint16 values, not divided by 10000. It returns
        the same four bands at 2.5 m, with measurement-consistency back-projection.
        A three-band TCI/RGB raster, an 8-bit image, or a float stack is refused,
        because that is not the model's input.

        path is a four-band GeoTIFF in the correct order. If the raster has more
        than four bands, pass bands=[...] selecting red, green, blue, and NIR in
        that order (for a standard 12-band S2 stack that is usually [4, 3, 2, 8]).
        bands must be 10 m; a 20 m or geographic input is refused instead of
        silently producing the wrong pixel size. bounds=[west, south, east, north]
        limits the work. max_tiles refuses a run before writing.

        Writes a four-band uint16 GeoTIFF at 4x under results/ (default
        results/superres.tif). Bands 1-3 are RGB for the map; band 4 is NIR.
        The added detail is generated, not measured. Returns the path, CRS,
        bounds, pixel size, tile count, and per-stage seconds. Call add_raster
        with bands=[1, 2, 3] afterwards, then fit_bounds.
        """
        spec = find_model(model_id)
        if spec.task != "super-resolution":
            raise ToolInputError(
                f"{spec.id} is a {spec.task} model, not a super-resolution model; "
                "ai_models lists which model serves which task"
            )

        with task_job(self._rt, self._manager, model_id, f"super-resolve {out}") as (job, session):
            target, timings, info = superres.super_resolve_raster(
                self._rt.workspace,
                path,
                session,
                bands=bands,
                bounds=bounds,
                max_tiles=max_tiles,
                out=out,
                job=job,
            )
            summary: dict[str, Any] = {
                "model": spec.id,
                "device": session.provider,
                "path": self._rt.record_artifact(target),
                "timings": timings.as_dict(),
            }
            summary.update(info)
            summary["message"] = (
                "output bands are B04/B03/B02/B08 at 2.5 m; "
                "call add_raster with bands=[1, 2, 3] to show RGB; detail is generated"
            )
            job.done(artifact=self._rt.workspace.relative(target))
            return summary


__all__ = ["SuperResolutionPack"]
