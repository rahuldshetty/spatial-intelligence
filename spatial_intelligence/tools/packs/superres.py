"""Super-resolution: upscale a raster 4x with a local Satlas ESRGAN model."""

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
    """4x upscaling of rasters, on this machine."""

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
        model_id: str = "satlas-esrgan-1s2",
        dates: list[str] | None = None,
        bands: list[int] | None = None,
        bounds: list[float] | None = None,
        tile_size: int = superres.DEFAULT_TILE,
        halo: int = superres.DEFAULT_HALO,
        max_tiles: int = superres.MAX_TILES,
        out: str = superres.DEFAULT_OUT,
    ) -> dict:
        """Upscale a raster 4x with a local Satlas ESRGAN model.

        Trained on Sentinel-2 10 m imagery and run at 4x: output is 2.5 m RGB.
        Detail is generated, not recovered, so use it for display and
        interpretation, not measurement. The default model takes one RGB raster
        (L1C TCI is the matched input); the -2s2/-4s2/-8s2/-16s2 models take 2,
        4, 8, or 16 dates, and -10m/-20m/-60m take 8 dates with extra S2 bands.

        path is the first date and defines the grid. For a multi-date model pass
        the remaining co-registered rasters in dates; a different CRS, size, or
        transform is refused rather than resampled. bands selects the raster
        bands in the model's per-date order, defaulting to 1..N: -10m is TCI RGB
        then B08, -20m adds B05/B06/B07/B11/B12, -60m adds B01/B09/B10. 8-bit
        input is divided by 255; 16-bit Sentinel-2 by 8160.

        bounds=[west, south, east, north] limits the work to that box. tile_size
        is the core in source pixels; halo is context that is cropped from the
        output.
        max_tiles refuses a run before writing; raise it or pass bounds for a
        larger area. Writes a tiled GeoTIFF at 4x under results/ (default
        results/superres.tif) and returns its path, CRS, bounds, output pixel
        size, tile count, and per-stage seconds. Call add_raster afterwards to
        show it, then fit_bounds.
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
                dates=dates or (),
                bands=bands,
                bounds=bounds,
                tile_size=tile_size,
                halo=halo,
                max_tiles=max_tiles,
                out=out,
                job=job,
            )
            summary: dict[str, Any] = {
                "model": spec.id,
                "scale": spec.scale,
                "device": session.provider,
                "path": self._rt.record_artifact(target),
                "timings": timings.as_dict(),
            }
            summary.update(info)
            summary["message"] = (
                "detail is synthetic; call add_raster to show the result, then fit_bounds"
            )
            job.done(artifact=self._rt.workspace.relative(target))
            return summary


__all__ = ["SuperResolutionPack"]
