"""Raster tools: inspect, convert, warp, clip, stretch, and index rasters."""

from __future__ import annotations

from ...contracts.effects import Effect
from ...geo import raster
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool


@pack(category="raster", effects=frozenset({Effect.WORKSPACE_WRITE}))
class RasterPack:
    """Rasterio-backed raster tools; every output lands under ``results/``."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime

    # -- inspection (read-only) -------------------------------------------

    @tool(effects=frozenset({Effect.READ}))
    def raster_info(self, path: str) -> dict:
        """Return CRS, transform, size, bands, dtypes, nodata and bounds.

        ``path`` may be a remote COG/GeoTIFF URL: only its metadata is fetched.
        """
        return raster.raster_info(self._rt.workspace, path)

    @tool(effects=frozenset({Effect.READ}))
    def raster_stats(self, path: str, band: int = 1) -> dict:
        """Return min/max/mean/std/percentiles and a 256-bin histogram for a band.

        ``path`` may be a remote COG URL, but the whole band is read; on a full
        satellite tile that is a large download, so clip it first.
        """
        return raster.raster_stats(self._rt.workspace, path, band)

    @tool(effects=frozenset({Effect.READ}))
    def sample_point(self, path: str, lng: float, lat: float, band: int = 1) -> dict:
        """Sample one pixel value at a coordinate; returns ``{value, lng, lat}``.

        ``path`` may be a remote COG URL: one pixel costs one small range request.
        """
        return raster.sample_point(self._rt.workspace, path, lng, lat, band)

    # -- writers -----------------------------------------------------------

    @tool(kind=ToolKind.REPORTING)
    def to_cog(self, path: str, out: str, resample: str = "nearest") -> str:
        """Convert a raster to a Cloud Optimized GeoTIFF; returns the absolute path."""
        total = raster.raster_info(self._rt.workspace, path)["count"]
        job = self._rt.reporter.job(
            "raster", f"to_cog {out}", unit="steps", total=total
        )
        with job:
            written = raster.to_cog(self._rt.workspace, path, out, resample, job=job)
            job.done(artifact=self._rt.workspace.relative(written))
        return self._rt.record_artifact(written)

    @tool(kind=ToolKind.REPORTING)
    def reproject(
        self, path: str, out: str, dst_crs: str, resampling: str = "nearest"
    ) -> str:
        """Reproject a raster to ``dst_crs`` (e.g. ``"EPSG:4326"``)."""
        total = raster.raster_info(self._rt.workspace, path)["count"]
        job = self._rt.reporter.job(
            "raster", f"reproject {out}", unit="steps", total=total
        )
        with job:
            written = raster.reproject(
                self._rt.workspace, path, out, dst_crs, resampling, job=job
            )
            job.done(artifact=self._rt.workspace.relative(written))
        return self._rt.record_artifact(written)

    @tool(kind=ToolKind.REPORTING)
    def clip(
        self,
        path: str,
        out: str,
        bounds: list[float] | None = None,
        mask_geojson: str | None = None,
        bounds_crs: str = "EPSG:4326",
    ) -> str:
        """Clip a raster to ``[west,south,east,north]`` or a mask GeoJSON.

        ``path`` may be a remote COG URL, and only the bytes the clip window covers
        are fetched. This is the cheap way to work with a catalog scene: a full
        Sentinel-2 band is ~200 MB, while a town-sized window is a few MB, so clip
        first and run the analysis on the clipped copy.

        ``bounds`` are longitude/latitude (``[west, south, east, north]``), and a
        projected raster such as a UTM satellite tile is converted for you; pass
        ``bounds_crs`` with the raster's own CRS only if the numbers are already
        projected.
        """
        job = self._rt.reporter.job(
            "raster", f"clip {out}", unit="steps", total=1
        )
        with job:
            written = raster.clip(
                self._rt.workspace, path, out, bounds, mask_geojson, bounds_crs, job=job
            )
            job.done(artifact=self._rt.workspace.relative(written))
        return self._rt.record_artifact(written)

    @tool(kind=ToolKind.REPORTING)
    def rescale(
        self,
        path: str,
        out: str,
        vmin: float | None = None,
        vmax: float | None = None,
        method: str = "percentile",
        pmin: float = 2,
        pmax: float = 98,
        nodata: float | None = None,
    ) -> str:
        """Stretch a raster to uint8. Returns the absolute path."""
        job = self._rt.reporter.job(
            "raster", f"rescale {out}", unit="steps", total=1
        )
        with job:
            written = raster.rescale(
                self._rt.workspace,
                path,
                out,
                vmin,
                vmax,
                method,
                pmin,
                pmax,
                nodata,
                job=job,
            )
            job.done(artifact=self._rt.workspace.relative(written))
        return self._rt.record_artifact(written)

    @tool(requires_approval=True)
    def band_math(
        self,
        path: str,
        out: str,
        expression: str,
        bands: dict[str, int] | None = None,
    ) -> str:
        """Evaluate a NumPy expression over named bands (e.g. an index).

        ``expression`` is evaluated with NumPy bound to ``np`` and each entry of
        ``bands`` bound to its band array, e.g. ``"(nir - red) / (nir + red)"``
        with ``bands={"nir": 4, "red": 3}``.

        This runs arbitrary NumPy code, so it only runs once the user approved
        dangerous mode (File -> Settings); prefer ``run_python`` with numpy for
        band math instead.
        """
        self._rt.require_approval(
            "band_math", hint="Use run_python with numpy for band math instead."
        )
        written = raster.band_math(
            self._rt.workspace, path, out, expression, bands
        )
        return self._rt.record_artifact(written)

    @tool()
    def gdal_translate(
        self, path: str, out: str, options: dict[str, str] | None = None
    ) -> str:
        """Copy a raster, applying GDAL creation options.

        ``options`` are GDAL creation options as ``key=value`` — ``TILED=YES``,
        ``COMPRESS=ZSTD``, ``PREDICTOR=2``, ``BIGTIFF=YES``, ``NUM_THREADS=ALL_CPUS``,
        ``driver=COG`` to write a Cloud-Optimized GeoTIFF, ``driver=PNG`` for another
        container. This is the escape hatch for the creation settings no other tool
        exposes; it cannot subset, resize, rescale, or change the pixel type, and it
        says so rather than writing a file that ignores the request.
        """
        written = raster.gdal_translate(self._rt.workspace, path, out, options)
        return self._rt.record_artifact(written)

    # -- analysis ----------------------------------------------------------

    @tool()
    def spectral_index(
        self,
        path: str,
        out: str,
        index: str = "ndvi",
        bands: dict[str, int | str] | None = None,
        scale: float = 1.0,
    ) -> str:
        """Compute a named spectral index into a float32 raster (no approval needed).

        ``index`` is one of ndvi, gndvi, ndwi, ndmi, ndbi, nbr, evi, savi.
        ``bands`` maps the index's band names to a 1-based band number in this
        raster (``{"nir": 8, "red": 4}``) or to another single-band raster
        (``{"nir": "data/B08.tif", "red": "data/B04.tif"}``) — catalog imagery
        arrives one COG per band, so clip the bands you need (a small window, not
        the whole ~200 MB tile) and pass those paths, or mix a band path with band
        numbers. ``scale`` divides each band first, which only
        matters for the indices with an additive term: a Sentinel-2 L2A COG holds
        reflectance scaled by 10000, so ``scale=10000`` is what makes evi and savi
        correct. Prefer this over ``band_math``, which needs dangerous mode.
        """
        written = raster.spectral_index(
            self._rt.workspace, path, out, index, bands, scale
        )
        return self._rt.record_artifact(written)

    @tool()
    def zonal_stats(
        self,
        path: str,
        zones: str,
        out: str,
        stats: list[str] | None = None,
        band: int = 1,
        prefix: str = "raster",
        all_touched: bool = False,
        layer: str | None = None,
    ) -> str:
        """Summarize a raster band inside each zone polygon, as new attributes.

        ``stats`` is any of count, mean, min, max, sum, std, median (default
        count/mean/min/max), written as ``<prefix>_<stat>`` columns on a copy of
        the zones layer. Nodata pixels are excluded, so a zone over no data gets
        ``count = 0`` and empty statistics rather than a zero mean. Pass
        ``layer`` when the zones are one layer of a GeoPackage.
        """
        written = raster.zonal_stats(
            self._rt.workspace, path, zones, out, stats, band, prefix, all_touched, layer
        )
        return self._rt.record_artifact(written)

    @tool(kind=ToolKind.REPORTING)
    def hillshade(
        self,
        path: str,
        out: str,
        band: int = 1,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        z_factor: float = 1.0,
    ) -> str:
        """Shade a DEM from a light source, as a 0-255 grayscale raster.

        ``azimuth`` is the light's compass direction (315 = north-west, the
        cartographic default) and ``altitude`` its height above the horizon.
        Shadowed pixels are 0, the same value nodata takes. ``z_factor``
        exaggerates relief; the DEM's own grid spacing is used, with a geographic
        raster measured in meters at its centre latitude.
        """
        job = self._rt.reporter.job("raster", f"hillshade {out}", unit="steps", total=1)
        with job:
            written = raster.hillshade(
                self._rt.workspace, path, out, band, azimuth, altitude, z_factor
            )
            job.done(artifact=self._rt.workspace.relative(written))
        return self._rt.record_artifact(written)

    @tool()
    def slope(
        self, path: str, out: str, band: int = 1, units: str = "degrees", z_factor: float = 1.0
    ) -> str:
        """Return slope steepness; ``units`` is ``degrees`` or ``percent``.

        Nearby values mean very different things: 5° is gentle, 30° is steep, and
        45° is a 100 % grade. A geographic DEM is measured with its grid spacing
        converted to meters at the raster's centre latitude.
        """
        written = raster.slope(self._rt.workspace, path, out, band, units, z_factor)
        return self._rt.record_artifact(written)

    @tool()
    def aspect(self, path: str, out: str, band: int = 1, z_factor: float = 1.0) -> str:
        """Return the compass direction a slope faces: 0 = north, 90 = east, clockwise.

        Flat pixels take 0, which is the convention rather than a measurement.
        Useful with ``hillshade`` and for solar or aspect-based site screening.
        """
        written = raster.aspect(self._rt.workspace, path, out, band, z_factor)
        return self._rt.record_artifact(written)

    @tool(kind=ToolKind.REPORTING)
    def polygonize(self, path: str, out: str, band: int = 1) -> str:
        """Vectorize a discrete raster: one polygon per connected same-value region.

        Each polygon carries its raster ``value``. Meant for classified or
        integer rasters — a continuous raster yields one polygon per run of
        identical values, so threshold or reclassify it first.
        """
        job = self._rt.reporter.job("raster", f"polygonize {out}", unit="steps", total=1)
        with job:
            written = raster.polygonize(self._rt.workspace, path, out, band, job=job)
            job.done(artifact=self._rt.workspace.relative(written))
        return self._rt.record_artifact(written)

    @tool(kind=ToolKind.REPORTING)
    def contour(
        self,
        path: str,
        out: str,
        interval: float = 10.0,
        base: float = 0.0,
        band: int = 1,
        simplify: float = 0.0,
    ) -> str:
        """Vectorize a continuous raster into contour lines every ``interval`` units.

        Use this to turn a DEM, rainfall surface, or any continuous raster into
        lines you can label, measure, or ship as vectors: ``interval`` is in the
        raster's units (meters of elevation for a DEM), ``base`` aligns the ladder
        (10 m contours from 100 m: ``base=100``), and ``simplify`` is a tolerance in
        pixels that removes marching squares' stair-steps. Each line carries its
        ``value``. The band must have no nodata — clip to the data area first.
        """
        job = self._rt.reporter.job("raster", f"contour {out}", unit="steps", total=1)
        with job:
            written = raster.contour(
                self._rt.workspace, path, out, interval, base, band, simplify, job=job
            )
            job.done(artifact=self._rt.workspace.relative(written))
        return self._rt.record_artifact(written)

    @tool()
    def compose_rgb(
        self,
        red: str,
        green: str,
        blue: str,
        out: str,
        stretch: bool = True,
    ) -> str:
        """Stack three single-band rasters into one 8-bit RGB GeoTIFF.

        Use this when a sensor publishes one file per band (Landsat Collection 2
        or Sentinel-1 GRD downloaded from a catalog): ``add_raster`` shows a
        three-band file in colour, while a single-band file renders grey. The
        three inputs must share one grid and CRS — reproject and clip them to the
        same extent first. ``stretch`` scales each band by its 2nd-98th
        percentile so the composite looks like imagery.
        """
        written = raster.compose_rgb(
            self._rt.workspace, red, green, blue, out, stretch=stretch
        )
        return self._rt.record_artifact(written)


__all__ = ["RasterPack"]
