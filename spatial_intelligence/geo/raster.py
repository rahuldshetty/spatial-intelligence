"""Raster services backed by rasterio.

All outputs land under ``results/``. GDAL itself comes with rasterio, which
:func:`gdal_translate` uses for the creation options the other tools do not
expose.

Services are plain functions over a
:class:`~spatial_intelligence.workspace.Workspace` (plus, for the writers that
touch more than one band, an optional progress
:class:`~spatial_intelligence.contracts.progress.Job`). They never read the
ambient runtime, and they return the :class:`~pathlib.Path` they wrote: recording
the output in the workspace manifest is the calling pack's job.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.shutil  # noqa: F401 - rasterio does not import this submodule itself
from rasterio import mask, warp
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform

from ..contracts.errors import ToolInputError, WorkspaceError
from ..contracts.progress import Job
from ..workspace import Workspace

#: Bins in the histogram :func:`raster_stats` returns.
HISTOGRAM_BINS = 256
#: Pixels one remote source may cover before a whole-array read is refused.
#: A catalog COG for a full satellite tile is ~10800x10800, and upcasting it to
#: float64 costs about a gigabyte, so a tool that would read all of it asks for a
#: clip instead of taking the machine down with it.
MAX_REMOTE_PIXELS = 50_000_000
#: Most contour levels one :func:`contour` call may produce.
MAX_CONTOUR_LEVELS = 200
#: Smallest block a tiled raster must use to count as COG-shaped. A COG reader
#: fetches whole tiles over range requests, so a striped file (one row per block)
#: would make it read the entire image for the first screenful.
COG_MIN_BLOCK = 256

#: Options that are ``gdal_translate`` *API* arguments rather than creation options.
#: GDAL's creation-option path ignores the ones a driver does not know, so without
#: the bindings these would be silently dropped and produce a file that does not
#: answer the request; the tool refuses them and names what does the job instead.
_TRANSLATE_ONLY_OPTIONS = {
    "OUTPUTTYPE": "converting the pixel type: read and rewrite it in run_python, or use rescale for uint8",
    "OUTSIZE": "resizing: use reproject with a target resolution",
    "WIDTH": "resizing: use reproject with a target resolution",
    "HEIGHT": "resizing: use reproject with a target resolution",
    "PROJWIN": "subsetting to a window: use clip with bounds",
    "SRCWIN": "subsetting to a pixel window: use clip with bounds",
    "SCALEPARAMS": "rescaling values: use rescale",
    "EXPONENT": "value scaling: use rescale",
    "UNSCALE": "value scaling: use rescale",
    "ANODATA": "setting nodata: write it in run_python with rasterio",
    "ASRS": "overriding the CRS: use reproject",
    "AULLR": "georeferencing by corner coordinates: use run_python with rasterio",
}

#: Band names each spectral index needs, and how it combines them.
SPECTRAL_INDICES: dict[str, tuple[str, ...]] = {
    "ndvi": ("nir", "red"),
    "gndvi": ("nir", "green"),
    "ndwi": ("green", "nir"),
    "ndmi": ("nir", "swir16"),
    "ndbi": ("swir16", "nir"),
    "nbr": ("nir", "swir22"),
    "evi": ("nir", "red", "blue"),
    "savi": ("nir", "red"),
}

#: Statistics :func:`zonal_stats` can compute per zone.
ZONAL_STATS = ("count", "mean", "min", "max", "sum", "std", "median")


def _is_remote(path: str) -> bool:
    """Whether ``path`` is a URL rather than a workspace-relative file."""
    return str(path).startswith(("http://", "https://"))


def _source(
    workspace: Workspace, path: str, *, whole: bool = False
) -> rasterio.DatasetReader:
    """Open a raster for reading: a workspace file, or a remote COG by URL.

    A URL is handed to GDAL, which reads it over HTTP range requests, so a
    windowed tool such as :func:`clip` fetches only the bytes its window covers
    instead of the whole file. ``whole`` marks the callers that will read every
    pixel: for those, a remote source larger than :data:`MAX_REMOTE_PIXELS` is
    refused with the instruction to clip first.
    """
    if _is_remote(path):
        dataset = rasterio.open(path)
        if whole and dataset.width * dataset.height > MAX_REMOTE_PIXELS:
            width, height = dataset.width, dataset.height
            dataset.close()
            raise ToolInputError(
                f"{path!r} is {width}x{height} pixels, and this tool reads all of it; "
                "clip the area you need first (clip accepts a remote URL and fetches only "
                "that window), then run this on the clipped copy"
            )
        return dataset
    return rasterio.open(str(workspace.resolve(path, must_exist=True)))


def _target(workspace: Workspace, path: str) -> Path:
    """Resolve a write target, confined to ``results/``, ``maps/``, or ``data/``."""
    target = workspace.resolve(path, write=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _write_vector(workspace: Workspace, out: str, frame: Any) -> Path:
    """Write a vector output through the vector tools' own writer.

    ``geo.vector`` owns the extension-to-driver table and refuses an extension it
    cannot write; deferring the import keeps one answer to which formats exist,
    and the import is local because ``geo.vector`` reaches the map layer, which
    imports this module.
    """
    from .vector import _write

    return _write(workspace, out, frame)


def _advance(job: Job | None, detail: str) -> None:
    """Advance an optional progress job by one step."""
    if job is not None:
        job.advance(detail=detail)


def raster_info(workspace: Workspace, path: str) -> dict:
    """Return CRS, transform, size, bands, dtypes, nodata, bounds, and COG status.

    ``cog`` reports whether the file is laid out as a Cloud-Optimized GeoTIFF,
    which is what the map needs. ``None`` means it could not be read as a raster,
    or that the source is a URL, whose layout is not checked here.
    """
    cog = None if _is_remote(path) else looks_cog(workspace.resolve(path, must_exist=True))
    with _source(workspace, path) as ds:
        return {
            "cog": cog,
            "crs": str(ds.crs),
            "transform": list(ds.transform)[:6],
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtypes": list(ds.dtypes),
            "nodata": ds.nodata,
            "bounds": list(ds.bounds),
        }


def raster_stats(workspace: Workspace, path: str, band: int = 1) -> dict:
    """Return min/max/mean/std/percentiles and a 256-bin histogram for a band."""
    with _source(workspace, path, whole=True) as ds:
        arr = ds.read(band, masked=True).astype("float64")
        data = arr.compressed()
        if data.size == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "p2": None,
                "p98": None,
                "histogram": [],
            }
        hist, _ = np.histogram(data, bins=HISTOGRAM_BINS)
        return {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "p2": float(np.nanpercentile(data, 2)),
            "p98": float(np.nanpercentile(data, 98)),
            "histogram": [int(x) for x in hist],
        }


def looks_cog(path: Path) -> bool | None:
    """Whether a raster is laid out the way a COG reader needs it.

    The check is structural — tiled, with blocks big enough to be worth a range
    request — because that is what the map's COG client depends on, and because it
    is the property the failed conversion was about: a striped GeoTIFF pulls the
    whole image for the first screenful. ``rio-cogeo``'s validator is deliberately
    not used here: it calls a 64x64 striped file a valid COG, so it cannot be the
    gate (it remains available to generated code, and as a second opinion when a
    file is checked by hand).

    Returns ``None`` when the file cannot be read as a raster at all — the one
    case where a caller should keep its own behavior rather than convert.
    """
    try:
        with rasterio.open(path) as ds:
            if not ds.is_tiled:
                return False
            block = ds.block_shapes[0] if ds.block_shapes else (0, 0)
            return min(block) >= COG_MIN_BLOCK
    except Exception:
        return None


def to_cog_if_needed(
    workspace: Workspace, path: str, *, job: Job | None = None
) -> tuple[Path, bool]:
    """Return a COG for ``path``, converting a striped raster beside it.

    The embedded GeoLibre app loads a COG: handed a striped GeoTIFF it either
    fails or asks its Python sidecar to convert the file, and this harness embeds
    the static app without that sidecar, so the conversion cannot succeed. Doing
    it here — with rasterio's COG driver, which handles small rasters and NaN
    nodata — is what keeps any raster output loadable on the map.

    The copy is written under ``results/`` like every other output, so a
    conversion of a source in ``data/`` does not put an unrecorded file beside it.

    Returns ``(path, converted)``: the file to load, and whether it is a new copy.
    """
    source = workspace.resolve(path, must_exist=True)
    if looks_cog(source) is not False:
        # Already COG-shaped, or not readable as a raster at all: no copy either way.
        return source, False
    suffix = source.suffix or ".tif"
    target = workspace.results / f"{source.stem}_cog{suffix}"
    written = to_cog(workspace, workspace.relative(source), workspace.relative(target), job=job)
    return written, True


def to_cog(
    workspace: Workspace,
    path: str,
    out: str,
    resample: str = "nearest",
    *,
    job: Job | None = None,
) -> Path:
    """Convert a raster to a Cloud Optimized GeoTIFF; returns the absolute path."""
    dst = _target(workspace, out)
    with _source(workspace, path, whole=True) as src:
        profile = src.profile.copy()
        profile.update(
            driver="COG",
            tiled=True,
            compress="deflate",
            overview_resampling=resample,
        )
        with rasterio.open(dst, "w", **profile) as ds:
            for i in range(1, src.count + 1):
                ds.write(src.read(i), i)
                _advance(job, f"band {i}/{src.count}")
    return dst


def _window_for_bounds(
    src: rasterio.DatasetReader, bounds: list[float], bounds_crs: str
) -> Any:
    """Return the pixel window that covers ``bounds`` in ``src``.

    The numbers are read as ``bounds_crs`` (longitude/latitude by default) and
    reprojected into the raster's CRS when the two differ, then clamped to the
    raster: a box that covers more than the raster yields the raster, and a box
    that misses it entirely is refused.
    """
    if len(bounds) != 4 or not all(math.isfinite(float(value)) for value in bounds):
        raise ToolInputError("bounds must be four finite numbers: [west, south, east, north]")
    west, south, east, north = (float(value) for value in bounds)
    if west >= east or south >= north:
        raise ToolInputError("bounds must be [west, south, east, north]")

    raster_crs = src.crs
    requested = str(bounds_crs or "").strip()
    same = (
        raster_crs is None
        or not requested
        or raster_crs.to_string().upper() == requested.upper()
        or (requested.upper() == "EPSG:4326" and raster_crs.is_geographic)
    )
    if same:
        left, bottom, right, top = west, south, east, north
    else:
        if requested.upper() == "EPSG:4326" and not (
            -180.0 <= west <= 180.0 and -180.0 <= east <= 180.0
            and -90.0 <= south <= 90.0 and -90.0 <= north <= 90.0
        ):
            raise ToolInputError(
                f"bounds {bounds} are outside longitude/latitude range but this raster is "
                f"{raster_crs}; pass bounds_crs={raster_crs} if they are already projected"
            )
        try:
            left, bottom, right, top = warp.transform_bounds(
                requested, raster_crs, west, south, east, north
            )
        except Exception as exc:
            raise ToolInputError(
                f"could not convert {requested} bounds to the raster's {raster_crs}: {exc}"
            ) from exc
    try:
        window = from_bounds(left, bottom, right, top, transform=src.transform)
    except Exception as exc:
        raise ToolInputError(f"could not turn bounds {bounds} into a raster window: {exc}") from exc
    whole = rasterio.windows.Window(0, 0, src.width, src.height)
    try:
        clipped = window.intersection(whole)
    except rasterio.errors.WindowError as exc:
        # An intersection that misses the raster raises rather than returning an
        # empty window; both readings mean the same thing to the caller.
        raise ToolInputError(
            f"bounds {bounds} ({requested or 'raster CRS'}) do not overlap this raster "
            f"({raster_crs}): {[round(value, 4) for value in src.bounds]}"
        ) from exc
    if clipped.width < 1 or clipped.height < 1:
        raise ToolInputError(
            f"bounds {bounds} ({requested or 'raster CRS'}) do not overlap this raster "
            f"({raster_crs}): {[round(value, 4) for value in src.bounds]}"
        )
    return clipped


def reproject(
    workspace: Workspace,
    path: str,
    out: str,
    dst_crs: str,
    resampling: str = "nearest",
    *,
    job: Job | None = None,
) -> Path:
    """Reproject a raster to ``dst_crs`` (e.g. ``"EPSG:4326"``)."""
    dst = _target(workspace, out)
    with _source(workspace, path, whole=True) as src:
        transform, width, height = warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            driver="GTiff",
            compress="deflate",
        )
        with rasterio.open(dst, "w", **kwargs) as ds:
            for i in range(1, src.count + 1):
                warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(ds, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=getattr(
                        warp.Resampling, resampling, warp.Resampling.nearest
                    ),
                )
                _advance(job, f"band {i}/{src.count}")
    return dst


def clip(
    workspace: Workspace,
    path: str,
    out: str,
    bounds: list[float] | None = None,
    mask_geojson: str | None = None,
    bounds_crs: str = "EPSG:4326",
    *,
    job: Job | None = None,
) -> Path:
    """Clip a raster to ``[west,south,east,north]`` or a mask GeoJSON.

    ``bounds`` are longitude/latitude by default, because that is what every
    other bound in this harness means (the map, a drawn box, a catalog search).
    A projected raster — a UTM satellite tile, most often — is handled by
    converting the box into its CRS first, so a lng/lat request works on it
    directly. Pass ``bounds_crs`` set to the raster's own CRS to give projected
    coordinates instead. A box that does not overlap the raster is an error
    rather than an empty file.
    """
    dst = _target(workspace, out)
    with _source(workspace, path) as src:
        if bounds is not None:
            window = _window_for_bounds(src, bounds, bounds_crs)
            kwargs = src.meta.copy()
            kwargs.update(
                height=window.height,
                width=window.width,
                transform=window_transform(window, src.transform),
                driver="GTiff",
                compress="deflate",
            )
            with rasterio.open(dst, "w", **kwargs) as ds:
                ds.write(src.read(window=window))
                _advance(job, "window written")
        elif mask_geojson is not None:
            mask_path = workspace.resolve(mask_geojson, must_exist=True)
            with open(mask_path, encoding="utf-8") as f:
                geoms = json.load(f)
            if "features" in geoms:
                geoms = [feat["geometry"] for feat in geoms["features"]]
            else:
                geoms = [geoms]
            out_image, out_transform = mask.mask(src, geoms, crop=True)
            kwargs = src.meta.copy()
            kwargs.update(
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                driver="GTiff",
                compress="deflate",
            )
            with rasterio.open(dst, "w", **kwargs) as ds:
                ds.write(out_image)
                _advance(job, "mask written")
        else:
            raise WorkspaceError("clip requires either bounds or mask_geojson")
    return dst


def rescale(
    workspace: Workspace,
    path: str,
    out: str,
    vmin: float | None = None,
    vmax: float | None = None,
    method: str = "percentile",
    pmin: float = 2,
    pmax: float = 98,
    nodata: float | None = None,
    *,
    job: Job | None = None,
) -> Path:
    """Stretch a raster to uint8. Returns the absolute path."""
    dst = _target(workspace, out)
    with _source(workspace, path, whole=True) as src:
        arr = src.read(1).astype("float64")
        if nodata is not None and src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        if vmin is None or vmax is None:
            valid = arr[np.isfinite(arr)]
            if method == "percentile":
                vmin = vmin if vmin is not None else float(np.nanpercentile(valid, pmin))
                vmax = vmax if vmax is not None else float(np.nanpercentile(valid, pmax))
            else:
                vmin = vmin if vmin is not None else float(valid.min())
                vmax = vmax if vmax is not None else float(valid.max())
        scaled = np.clip((arr - vmin) / (vmax - vmin) * 255.0, 0, 255).astype("uint8")
        if nodata is not None:
            scaled = np.where(np.isnan(arr), 0, scaled)
        profile = src.profile.copy()
        profile.update(dtype="uint8", count=1, driver="COG", compress="deflate")
        if nodata is not None:
            profile.update(nodata=0)
        with rasterio.open(dst, "w", **profile) as ds:
            ds.write(scaled, 1)
            _advance(job, "band 1/1")
    return dst


def band_math(
    workspace: Workspace,
    path: str,
    out: str,
    expression: str,
    bands: dict[str, int] | None = None,
) -> Path:
    """Evaluate a NumPy expression over named bands (e.g. an index)."""
    dst = _target(workspace, out)
    with _source(workspace, path, whole=True) as src:
        band_arrays: dict[str, np.ndarray] = {}
        for name, idx in (bands or {}).items():
            band_arrays[name] = src.read(idx).astype("float64")
        result = eval(expression, {"np": np}, band_arrays)  # noqa: S307 - tool contract
        result = np.asarray(result, dtype="float64")
        profile = src.profile.copy()
        profile.update(
            dtype=result.dtype.name, count=1, driver="GTiff", compress="deflate"
        )
        with rasterio.open(dst, "w", **profile) as ds:
            ds.write(result, 1)
    return dst


def sample_point(
    workspace: Workspace, path: str, lng: float, lat: float, band: int = 1
) -> dict:
    """Sample one pixel value at a coordinate; returns ``{value, lng, lat}``."""
    with _source(workspace, path) as ds:
        values = list(ds.sample([(lng, lat)], indexes=[band]))
        value = values[0][0] if values else None
        if value is not None:
            value = float(value)
        return {"value": value, "lng": lng, "lat": lat}


def gdal_translate(
    workspace: Workspace,
    path: str,
    out: str,
    options: dict[str, str] | None = None,
) -> Path:
    """Copy a raster, applying GDAL creation options.

    ``options`` are GDAL **creation options** as ``key=value``: ``TILED=YES``,
    ``COMPRESS=ZSTD``, ``PREDICTOR=2``, ``BLOCKSIZE=512``, ``BIGTIFF=YES``,
    ``NUM_THREADS=ALL_CPUS``, ``driver=COG`` to write a Cloud-Optimized GeoTIFF,
    ``driver=PNG`` for another container — the settings no other tool here
    exposes. It goes through rasterio, which carries GDAL itself.

    What it is not is a general ``gdal_translate``: subsetting, resizing, pixel-type
    conversion, and value scaling are *arguments of that program*, not creation
    options, so GDAL would ignore them and write a file that does not answer the
    request. Those are refused here, naming the tool that does the job instead.
    """
    source = workspace.resolve(path, must_exist=True)
    dst = _target(workspace, out)
    settings = {str(key).upper().replace("_", ""): str(value) for key, value in (options or {}).items()}
    # `driver` names the output format rather than a creation option, so it is
    # passed through as that argument instead of being uppercased into an option
    # every driver would silently ignore.
    driver = settings.pop("DRIVER", None)
    unsupported = {
        name: hint for name, hint in _TRANSLATE_ONLY_OPTIONS.items() if name in settings
    }
    if unsupported:
        raise ToolInputError(
            "these are gdal_translate arguments rather than creation options, so GDAL "
            "would ignore them: "
            + "; ".join(f"{name} -> {hint}" for name, hint in sorted(unsupported.items()))
        )
    try:
        # strict=False: GDAL decides which creation options a driver accepts, and
        # a driver ignoring one should not fail an otherwise good copy. An
        # unknown *driver*, on the other hand, still raises.
        rasterio.shutil.copy(str(source), str(dst), driver=driver, strict=False, **settings)
    except Exception as exc:
        raise ToolInputError(f"gdal_translate failed for {path!r}: {exc}") from exc
    return dst



def polygonize(
    workspace: Workspace,
    path: str,
    out: str,
    band: int = 1,
    *,
    job: Job | None = None,
) -> Path:
    """Vectorize a discrete raster: one polygon per connected same-value region.

    Meant for classified or integer rasters. A continuous raster will produce one
    polygon per run of identical values, which is rarely what a caller wants, so
    reclassify or threshold it first. Each polygon carries its raster ``value``.
    """
    import geopandas as gpd  # noqa: PLC0415 - keeps the module import light
    from rasterio.features import shapes  # noqa: PLC0415

    with _source(workspace, path, whole=True) as src:
        _check_bands(src, {"band": band})
        arr = src.read(band, masked=True)
        keep = ~np.ma.getmaskarray(arr) & np.isfinite(np.ma.getdata(arr))
        geoms = [
            {"type": "Feature", "properties": {"value": float(value)}, "geometry": geometry}
            for geometry, value in shapes(arr.filled(0), mask=keep, transform=src.transform)
        ]
        _advance(job, f"{len(geoms)} polygon(s)")
        if not geoms:
            raise WorkspaceError("polygonize found no regions: the band is empty or uniform")
        frame = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
    return _write_vector(workspace, out, frame)


def spectral_index(
    workspace: Workspace,
    path: str,
    out: str,
    index: str = "ndvi",
    bands: dict[str, int | str] | None = None,
    scale: float = 1.0,
    L: float = 0.5,
) -> Path:
    """Compute a named spectral index into a float32 raster.

    ``bands`` maps the index's band names either to a 1-based band number in
    ``path`` (``{"nir": 8, "red": 4}`` for a stacked Sentinel-2 file) or to
    another workspace raster that holds that band alone
    (``{"nir": "data/B08.tif", "red": "data/B04.tif"}``) — which is how the
    catalogs publish imagery, one COG per band. Sources may be mixed. Every band
    read from a separate file must share the reference raster's grid and CRS.

    ``scale`` divides every band first, which matters for the indices with an
    additive term: a Sentinel-2 L2A COG holds reflectance scaled by 10000, so
    ``scale=10000`` is what makes EVI and SAVI correct (a normalized difference is
    unaffected). ``L`` is SAVI's soil-brightness constant. Invalid pixels — a
    masked source pixel, or a division that cannot be evaluated — become
    ``nodata``.
    """
    key = str(index).strip().lower()
    required = SPECTRAL_INDICES.get(key)
    if required is None:
        raise ToolInputError(
            f"unknown index {index!r}; choose one of {', '.join(sorted(SPECTRAL_INDICES))}"
        )
    mapping: dict[str, int | str] = dict(bands or {})
    missing = [name for name in required if name not in mapping]
    if missing:
        raise ToolInputError(
            f"{key} needs bands {', '.join(required)}; pass bands={{"
            + ", ".join(f'"{name}": <band number or path>' for name in required)
            + "}"
        )
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale == 0:
        raise ToolInputError("scale must be a non-zero finite number")

    dst = _target(workspace, out)
    with _source(workspace, path, whole=True) as src:
        values: dict[str, np.ndarray] = {}
        invalid = np.zeros((src.height, src.width), dtype=bool)
        for name in required:
            reference = mapping[name]
            if isinstance(reference, str):
                with _source(workspace, reference, whole=True) as other:
                    if other.count != 1:
                        raise ToolInputError(
                            f"{reference!r} has {other.count} bands; a per-band source "
                            "must be a single-band raster"
                        )
                    if not _same_grid(src, other):
                        raise ToolInputError(
                            f"band {name!r} in {reference!r} does not share {path!r}'s grid "
                            "and CRS; reproject and clip the bands to one grid first"
                        )
                    array = other.read(1, masked=True).astype("float64")
            else:
                number = int(reference)
                _check_bands(src, {name: number})
                array = src.read(number, masked=True).astype("float64")
            values[name] = array.filled(np.nan) / scale
            invalid |= np.ma.getmaskarray(array)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.asarray(_index_formula(key, values, L), dtype="float64")
        result[invalid | ~np.isfinite(result)] = np.nan
        _write_float(dst, src, result)
    return dst


def _index_formula(index: str, values: dict[str, np.ndarray], L: float) -> np.ndarray:
    """Return the index's array from its named bands (all reflectance-scaled)."""
    if index == "ndvi":
        nir, red = values["nir"], values["red"]
        return (nir - red) / (nir + red)
    if index == "gndvi":
        nir, green = values["nir"], values["green"]
        return (nir - green) / (nir + green)
    if index == "ndwi":
        green, nir = values["green"], values["nir"]
        return (green - nir) / (green + nir)
    if index == "ndmi":
        nir, swir = values["nir"], values["swir16"]
        return (nir - swir) / (nir + swir)
    if index == "ndbi":
        swir, nir = values["swir16"], values["nir"]
        return (swir - nir) / (swir + nir)
    if index == "nbr":
        nir, swir = values["nir"], values["swir22"]
        return (nir - swir) / (nir + swir)
    if index == "evi":
        nir, red, blue = values["nir"], values["red"], values["blue"]
        return 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)
    if index == "savi":
        nir, red = values["nir"], values["red"]
        return ((nir - red) / (nir + red + L)) * (1.0 + L)
    # Explicit rather than a fallthrough: a name added to SPECTRAL_INDICES without
    # a formula here must not quietly compute SAVI and be written as that index.
    raise ToolInputError(f"no formula for index {index!r}")


def _same_grid(reference: rasterio.DatasetReader, other: rasterio.DatasetReader) -> bool:
    """Whether ``other`` shares ``reference``'s pixel grid and CRS.

    Every band of an index or a composite ends up in the reference raster's grid
    (that is where the output profile comes from), so a per-band file has to match
    it — not merely match the other bands.
    """
    return (other.width, other.height, other.crs, other.transform) == (
        reference.width,
        reference.height,
        reference.crs,
        reference.transform,
    )


def _check_bands(src: rasterio.DatasetReader, mapping: dict[str, int]) -> None:
    """Refuse a band number the dataset does not have."""
    for name, number in mapping.items():
        if not 1 <= number <= src.count:
            raise ToolInputError(
                f"band {number} for {name!r} is outside this raster's {src.count} band(s)"
            )


def zonal_stats(
    workspace: Workspace,
    path: str,
    zones: str,
    out: str,
    stats: list[str] | None = None,
    band: int = 1,
    prefix: str = "raster",
    all_touched: bool = False,
    layer: str | None = None,
) -> Path:
    """Summarize a raster band inside each zone polygon, as new zone attributes.

    ``stats`` is any of ``count``, ``mean``, ``min``, ``max``, ``sum``, ``std``,
    ``median`` (default ``count``, ``mean``, ``min``, ``max``), written as
    ``<prefix>_<stat>`` columns on a copy of the zones. ``all_touched`` counts
    every pixel the zone boundary touches, not just the ones whose centre falls
    inside. Nodata pixels are excluded, so a zone covering no data gets
    ``count = 0`` and empty statistics rather than a zero mean.
    """
    requested = list(stats) if stats else ["count", "mean", "min", "max"]
    unknown = [name for name in requested if name not in ZONAL_STATS]
    if unknown:
        raise ToolInputError(
            f"unknown statistic(s) {unknown}; choose from {', '.join(ZONAL_STATS)}"
        )
    # Read the zones through the vector tools' reader so a container with several
    # layers is refused (naming them) instead of silently using the first one,
    # which would report plausible numbers for the wrong layer.
    from .vector import _read

    frame = _read(workspace, zones, layer)
    if getattr(frame, "geometry", None) is None:
        raise ToolInputError(f"{zones!r} holds no geometry; zonal_stats needs polygon zones")
    with _source(workspace, path) as src:
        _check_bands(src, {"band": band})
        if frame.crs is None:
            raise ToolInputError(
                "the zones have no CRS; assign one (reproject_vector needs a source CRS too) "
                "so they can be aligned with the raster"
            )
        aligned = frame.to_crs(src.crs) if frame.crs != src.crs else frame
        for name in requested:
            frame[f"{prefix}_{name}"] = [
                _zone_stat(src, geometry, band, name, all_touched)
                for geometry in aligned.geometry
            ]
    return _write_vector(workspace, out, frame)


def _zone_stat(
    src: rasterio.DatasetReader,
    geometry: Any,
    band: int,
    name: str,
    all_touched: bool,
) -> float | None:
    """Return one statistic of the raster inside ``geometry``, or ``None``."""
    if geometry is None or geometry.is_empty:
        return 0.0 if name == "count" else None
    try:
        data, _ = mask.mask(src, [geometry], crop=True, all_touched=all_touched, filled=False)
        values = np.ma.getdata(data[0])[~np.ma.getmaskarray(data[0])]
    except (ValueError, OSError, rasterio.errors.RasterioError) as exc:
        # A zone that misses the raster has no window to read; it still has a count
        # of zero rather than an unknown one. Every other read failure is a real
        # error and must not masquerade as an empty zone.
        if "do not overlap" not in str(exc):
            raise ToolInputError(f"could not read the raster under a zone: {exc}") from exc
        values = np.array([], dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0 if name == "count" else None
    if name == "count":
        return float(values.size)
    if name == "mean":
        return float(values.mean())
    if name == "min":
        return float(values.min())
    if name == "max":
        return float(values.max())
    if name == "sum":
        return float(values.sum())
    if name == "std":
        return float(values.std())
    return float(np.median(values))


def hillshade(
    workspace: Workspace,
    path: str,
    out: str,
    band: int = 1,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> Path:
    """Shade a DEM from a light source; 0-255, as GDAL's hillshade renders it.

    ``azimuth`` is the light's compass direction (315 = north-west, the
    cartographic default) and ``altitude`` its height above the horizon in
    degrees. Pixels with no data are 0, which is also the value a fully shadowed
    pixel takes.
    """
    _check_light(azimuth, altitude, z_factor)
    dst = _target(workspace, out)
    with _source(workspace, path, whole=True) as src:
        dz_dx, dz_dy, valid = _gradients(src, band, z_factor)
        zenith = np.radians(90.0 - altitude)
        slope = np.arctan(np.hypot(dz_dx, dz_dy))
        facing = np.arctan2(-dz_dx, -dz_dy)
        shade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
            np.radians(azimuth) - facing
        )
        result = np.where(valid, np.clip(shade, 0.0, 1.0) * 255.0, 0.0).astype("uint8")
        nodata = 0 if not valid.all() else None
        profile = src.profile.copy()
        profile.update(dtype="uint8", count=1, driver="COG", compress="deflate")
        if nodata is not None:
            profile.update(nodata=nodata)
        with rasterio.open(dst, "w", **profile) as ds:
            ds.write(result, 1)
    return dst


def slope(
    workspace: Workspace,
    path: str,
    out: str,
    band: int = 1,
    units: str = "degrees",
    z_factor: float = 1.0,
) -> Path:
    """Return slope steepness; ``units`` is ``degrees`` or ``percent``.

    A geographic raster (``EPSG:4326``) is measured with its grid spacing
    converted to meters at the raster's centre latitude, so a global DEM gives
    sensible degrees without first being reprojected.
    """
    if units not in ("degrees", "percent"):
        raise ToolInputError("units must be degrees or percent")
    _check_z_factor(z_factor)
    dst = _target(workspace, out)
    with _source(workspace, path, whole=True) as src:
        dz_dx, dz_dy, valid = _gradients(src, band, z_factor)
        gradient = np.hypot(dz_dx, dz_dy)
        result = (
            np.degrees(np.arctan(gradient)) if units == "degrees" else gradient * 100.0
        )
        return _write_float(dst, src, np.where(valid, result, np.nan))


def aspect(
    workspace: Workspace,
    path: str,
    out: str,
    band: int = 1,
    z_factor: float = 1.0,
) -> Path:
    """Return the compass direction a slope faces: 0 = north, 90 = east, clockwise.

    Flat pixels take 0, which is the convention rather than a measurement.
    """
    _check_z_factor(z_factor)
    dst = _target(workspace, out)
    with _source(workspace, path, whole=True) as src:
        dz_dx, dz_dy, valid = _gradients(src, band, z_factor)
        facing = np.degrees(np.arctan2(-dz_dx, -dz_dy)) % 360.0
        return _write_float(dst, src, np.where(valid, facing, np.nan))


def _check_light(azimuth: float, altitude: float, z_factor: float) -> None:
    """Refuse light and scaling parameters that cannot produce a shade."""
    if not 0.0 <= azimuth <= 360.0 or not math.isfinite(azimuth):
        raise ToolInputError("azimuth must be between 0 and 360 degrees")
    if not 0.0 <= altitude <= 90.0 or not math.isfinite(altitude):
        raise ToolInputError("altitude must be between 0 and 90 degrees")
    _check_z_factor(z_factor)


def _check_z_factor(z_factor: float) -> None:
    """Refuse a vertical exaggeration that cannot scale a gradient."""
    if not math.isfinite(z_factor) or z_factor <= 0:
        raise ToolInputError("z_factor must be a positive, finite number")


def _gradients(
    src: rasterio.DatasetReader, band: int, z_factor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(dz/dx east, dz/dy north, valid)`` for one band, in raster units.

    Horne's 3×3 window in the form GDAL's terrain tools use: the edges are
    replicated so a border pixel gets a gradient instead of a hole, and the
    spacing is converted to meters when the raster is geographic, because an
    unprojected DEM's degrees would otherwise read as a ~100000× exaggeration.
    A pixel touching nodata is reported invalid rather than given an invented
    gradient.
    """
    _check_bands(src, {"band": band})
    array = src.read(band, masked=True).astype("float64")
    valid = ~np.ma.getmaskarray(array) & np.isfinite(np.ma.getdata(array))
    padded = np.pad(np.where(valid, np.ma.getdata(array), np.nan), 1, mode="edge")
    north_west, north_, north_east = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    west_, east_ = padded[1:-1, :-2], padded[1:-1, 2:]
    south_west, south_, south_east = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]

    dx, dy = _cell_size_meters(src)
    dz_dx = ((north_east + 2.0 * east_ + south_east) - (north_west + 2.0 * west_ + south_west)) / (
        8.0 * dx
    )
    dz_dy = ((north_west + 2.0 * north_ + north_east) - (south_west + 2.0 * south_ + south_east)) / (
        8.0 * dy
    )
    dz_dx *= z_factor
    dz_dy *= z_factor
    valid &= np.isfinite(dz_dx) & np.isfinite(dz_dy)
    return dz_dx, dz_dy, valid


def _cell_size_meters(src: rasterio.DatasetReader) -> tuple[float, float]:
    """Return the raster's pixel spacing in meters (east, north)."""
    dx = abs(float(src.transform.a))
    dy = abs(float(src.transform.e))
    crs = src.crs
    if crs is None or not crs.is_geographic:
        return dx or 1.0, dy or 1.0
    center_lat = float((src.bounds.top + src.bounds.bottom) / 2.0)
    return (
        dx * 111_320.0 * math.cos(math.radians(center_lat)) or 1.0,
        dy * 110_540.0,
    )


def contour(
    workspace: Workspace,
    path: str,
    out: str,
    interval: float = 10.0,
    base: float = 0.0,
    band: int = 1,
    simplify: float = 0.0,
    *,
    job: Job | None = None,
) -> Path:
    """Vectorize a continuous raster into contour lines every ``interval`` units.

    Levels run from the first multiple of ``interval`` at or above ``base`` up to
    the band's maximum, and each line carries its ``value``. ``simplify`` is a
    tolerance in *pixels* (converted to the raster's units) that removes the
    stair-steps marching squares produces; 0 keeps every vertex.

    Nodata is refused rather than contoured: a masked hole would put a spurious
    line around itself, so clip or sketch the raster to the area that has values.
    The lines come back in the raster's CRS.
    """
    if not math.isfinite(interval) or interval <= 0:
        raise ToolInputError("interval must be a positive, finite number")
    if not math.isfinite(base):
        raise ToolInputError("base must be a finite number")
    if not math.isfinite(simplify) or simplify < 0:
        raise ToolInputError("simplify must be a non-negative, finite number of pixels")

    import geopandas as gpd  # noqa: PLC0415 - keeps the module import light
    from shapely.geometry import LineString, mapping  # noqa: PLC0415
    from skimage import measure  # noqa: PLC0415 - optional image stack

    with _source(workspace, path, whole=True) as src:
        _check_bands(src, {"band": band})
        data = src.read(band, masked=True).astype("float64")
        if np.ma.getmaskarray(data).any():
            raise ToolInputError(
                "contour needs a band with no nodata pixels, because a masked hole "
                "would be outlined as if it were data; clip the raster to the region "
                "that has values first"
            )
        values = np.ma.getdata(data)
        levels = _contour_levels(values, interval, base)
        tolerance = simplify * max(abs(float(src.transform.a)), abs(float(src.transform.e)))
        features = []
        for level in levels:
            for segment in measure.find_contours(values, float(level)):
                xs, ys = rasterio.transform.xy(
                    src.transform, segment[:, 0], segment[:, 1], offset="center"
                )
                line = LineString(list(zip(xs, ys)))
                if tolerance > 0:
                    line = line.simplify(tolerance)
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"value": float(level)},
                        "geometry": mapping(line),
                    }
                )
            _advance(job, f"level {level:g}")
        if not features:
            raise ToolInputError(
                f"no contour lines at interval {interval:g} within the band's "
                f"{float(values.min()):g}..{float(values.max()):g}"
            )
        frame = gpd.GeoDataFrame.from_features(features, crs=src.crs)
    return _write_vector(workspace, out, frame)


def _contour_levels(values: np.ndarray, interval: float, base: float) -> np.ndarray:
    """Return the contour levels a band's value range calls for.

    The count is computed arithmetically and checked before anything is
    allocated: a very small interval would otherwise ask NumPy for billions of
    levels and die on the allocation instead of reporting the interval.
    """
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ToolInputError("the band has no finite value to contour")
    if high <= low:
        raise ToolInputError(f"the band is flat at {low:g}; there is no relief to contour")
    first = base + math.ceil((low - base) / interval) * interval
    count = int(math.floor((high - first) / interval)) + 1 if first <= high else 0
    if count <= 0:
        raise ToolInputError(
            f"no contour level falls between {low:g} and {high:g} at interval "
            f"{interval:g} from base {base:g}; use a smaller interval"
        )
    if count > MAX_CONTOUR_LEVELS:
        raise ToolInputError(
            f"an interval of {interval:g} over {low:g}..{high:g} makes {count} "
            f"levels (limit {MAX_CONTOUR_LEVELS}); use a larger interval"
        )
    return first + np.arange(count) * interval


def compose_rgb(
    workspace: Workspace,
    red: str,
    green: str,
    blue: str,
    out: str,
    stretch: bool = True,
    pmin: float = 2,
    pmax: float = 98,
) -> Path:
    """Stack three single-band rasters into one 8-bit RGB GeoTIFF.

    This is what makes a multi-band sensor that publishes one file per band —
    Landsat Collection 2 and Sentinel-1 GRD on Planetary Computer, for instance —
    renderable as a colour composite on the map, since ``add_raster`` needs one
    file with three bands. All three inputs must already share a grid and CRS
    (``reproject`` and ``clip`` them to one first). ``stretch`` scales each band
    between its ``pmin``/``pmax`` percentiles, which is what makes the result
    look like imagery rather than a dark rectangle.
    """
    dst = _target(workspace, out)
    with _source(workspace, red, whole=True) as red_ds, _source(
        workspace, green, whole=True
    ) as green_ds, _source(workspace, blue, whole=True) as blue_ds:
        for name, source in (("green", green_ds), ("blue", blue_ds)):
            if not _same_grid(red_ds, source):
                raise ToolInputError(
                    f"the {name} raster does not share the red raster's grid and CRS; "
                    "reproject and clip them to one grid first"
                )
        profile = red_ds.profile.copy()
        profile.update(dtype="uint8", count=3, driver="COG", compress="deflate")
        with rasterio.open(dst, "w", **profile) as dest:
            for index, source in enumerate((red_ds, green_ds, blue_ds), start=1):
                band = source.read(1, masked=True).astype("float64")
                values = band.compressed()
                if values.size == 0:
                    raise ToolInputError(f"band {index} has no valid pixels to stretch")
                if stretch:
                    low = float(np.percentile(values, pmin))
                    high = float(np.percentile(values, pmax))
                else:
                    low, high = float(values.min()), float(values.max())
                if high <= low:
                    high = low + 1.0
                scaled = np.clip((np.ma.getdata(band) - low) / (high - low) * 255.0, 0, 255)
                dest.write(
                    np.where(np.ma.getmaskarray(band), 0, scaled).astype("uint8"), index
                )
    return dst


def _write_float(dst: Path, src: rasterio.DatasetReader, array: np.ndarray) -> Path:
    """Write a single-band float32 COG whose nodata is NaN."""
    profile = src.profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        driver="COG",
        compress="deflate",
        nodata=float("nan"),
    )
    with rasterio.open(dst, "w", **profile) as ds:
        ds.write(array.astype("float32"), 1)
    return dst


__all__ = [
    "SPECTRAL_INDICES",
    "ZONAL_STATS",
    "aspect",
    "band_math",
    "clip",
    "compose_rgb",
    "contour",
    "gdal_translate",
    "hillshade",
    "looks_cog",
    "polygonize",
    "raster_info",
    "raster_stats",
    "reproject",
    "rescale",
    "sample_point",
    "slope",
    "spectral_index",
    "to_cog",
    "to_cog_if_needed",
    "zonal_stats",
]
