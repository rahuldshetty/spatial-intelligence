"""Compact capability discovery without exposing every integration as a tool."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the module import-cheap
    from .tools.registry import ToolRegistry


@dataclass(frozen=True)
class Capability:
    """One thing the harness can do, and how the agent reaches it."""

    id: str
    title: str
    summary: str
    keywords: tuple[str, ...]
    implementation: str
    tools: tuple[str, ...] = ()
    geolibre_plugins: tuple[str, ...] = ()
    status: str = "available"
    fallback: str | None = None


# This is intentionally metadata, not another set of model-visible tool schemas.
# It lets the agent find a relevant capability first; executable packs can later
# be injected dynamically as the GeoLibre bridge grows.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="workspace.files",
        title="Workspace files",
        summary="Import, download, find, inspect, and write workspace data.",
        keywords=("file", "import", "download", "workspace", "data"),
        implementation="backend",
        tools=("list_files", "find_files", "read_file", "write_file", "download", "download_files"),
    ),
    Capability(
        id="map.layers",
        title="Map layers and styling",
        summary="Add and style raster, vector, tile, and WMS layers.",
        keywords=(
            "map", "plot", "display", "layer", "style", "raster", "vector",
            "heatmap", "density", "swipe", "compare", "before", "after",
        ),
        implementation="backend+geolibre",
        tools=(
            "add_raster",
            "add_vector",
            "add_vector_to_map",
            "add_geojson",
            "add_heatmap",
            "swipe_compare",
            "style_layer",
            "fit_bounds",
        ),
    ),
    Capability(
        id="raster.processing",
        title="Raster processing",
        summary=(
            "Inspect and derive rasters: clip, reproject, rescale, spectral indices, "
            "zonal statistics, terrain (hillshade/slope/aspect), vectorize, and COGs."
        ),
        keywords=(
            "raster", "imagery", "satellite", "clip", "reproject", "cog", "band",
            "ndvi", "index", "zonal", "statistics", "hillshade", "slope", "aspect",
            "terrain", "dem", "elevation", "polygonize", "vectorize", "composite", "stretch",
            "remote", "url", "cog", "window", "subset", "contour", "isolines",
            "skimage", "filter", "morphology", "segmentation",
        ),
        implementation="backend",
        tools=(
            "raster_info",
            "raster_stats",
            "clip",
            "reproject",
            "rescale",
            "band_math",
            "to_cog",
            "spectral_index",
            "zonal_stats",
            "hillshade",
            "slope",
            "aspect",
            "polygonize",
            "contour",
            "compose_rgb",
            "gdal_translate",
            "sample_point",
        ),
    ),
    Capability(
        id="vector.processing",
        title="Vector processing",
        summary=(
            "Inspect and analyze vector data: layers in a GeoPackage, buffer, clip, "
            "dissolve, overlay, joins, selection, aggregation, grids, and topology repair."
        ),
        keywords=(
            "vector", "building", "road", "parcel", "polygon", "point", "line",
            "buffer", "clip", "dissolve", "overlay", "intersect", "union", "difference",
            "join", "select", "filter", "aggregate", "count", "sum", "grid", "hexagon",
            "voronoi", "centroid", "hull", "simplify", "geojson", "geopackage", "gpkg",
            "shapefile", "attribute", "topology", "invalid", "geometry",
        ),
        implementation="backend",
        tools=(
            "list_layers",
            "read_vector",
            "check_geometry",
            "fix_geometry",
            "reproject_vector",
            "buffer",
            "clip_vector",
            "dissolve",
            "overlay",
            "spatial_join",
            "attribute_join",
            "select_by_value",
            "select_by_location",
            "aggregate",
            "centroids",
            "convex_hull",
            "bounding_box",
            "simplify",
            "explode",
            "voronoi",
            "points_along",
            "grid",
            "export_vector",
            "add_heatmap",
        ),
    ),
    Capability(
        id="python.execution",
        title="Python sandbox",
        summary="Run Python snippets for data processing (pandas, geopandas, rasterio, …) and page the output.",
        keywords=(
            "python", "code", "execute", "run", "script", "sandbox", "compute",
            "calculate", "process", "analyze", "pandas", "geopandas", "numpy",
        ),
        implementation="backend",
        tools=("run_python", "inspect_output", "query_output", "python_help"),
    ),
    Capability(
        id="catalog.disaster-imagery",
        title="Open disaster imagery",
        summary="Search, download into data/, and map Vantor and OpenAerialMap disaster imagery.",
        keywords=(
            "disaster", "flood", "landslide", "earthquake", "before", "after",
            "planet", "vantor", "aerial", "stac", "imagery",
        ),
        implementation="backend using GeoLibre catalog contracts",
        tools=(
            "search_vantor_events",
            "search_vantor_imagery",
            "search_openaerialmap",
            "add_catalog_scene",
            "download_catalog_scene",
        ),
        geolibre_plugins=(
            "Vantor Open Data",
            "OpenAerialMap",
        ),
        status="available",
    ),
    Capability(
        id="catalog.satellite-imagery",
        title="Satellite imagery and elevation catalogs (STAC)",
        summary="Search Sentinel, Landsat, NAIP, Sentinel-1, and DEM collections, then download and map the best scene.",
        keywords=(
            "satellite", "sentinel", "landsat", "naip", "hls", "dem", "elevation",
            "stac", "catalog", "imagery", "reflectance", "ndvi", "sar", "multispectral",
        ),
        implementation="backend using the public STAC APIs GeoLibre's STAC panel browses",
        tools=(
            "list_stac_catalogs",
            "search_stac_collections",
            "search_stac_scenes",
            "add_catalog_scene",
            "download_catalog_scene",
        ),
        geolibre_plugins=("STAC Catalogs", "Planetary Computer"),
        status="available",
    ),
    Capability(
        id="catalog.planet-stac",
        title="Planet Open Data and other STAC APIs",
        summary="Discover Planet disaster releases, or reach a STAC API this build does not name.",
        keywords=("disaster", "planet", "stac", "catalog", "release"),
        implementation="geolibre-plugin",
        geolibre_plugins=(
            "Planet Open Data",
            "STAC Catalogs",
        ),
        status="interactive_handoff",
        fallback=(
            "Discovery of Planet's disaster releases is a GeoLibre panel, not a call "
            "this build can make: open the STAC Catalogs panel (Planet Open Data is "
            "the same panel pinned to Planet's releases), add the scenes you want, "
            "then continue from the persisted map layers. Once a release or any other "
            "catalog publishes a STAC API URL, pass it to search_stac_scenes instead."
        ),
    ),
    Capability(
        id="map.compare",
        title="Before/after comparison",
        summary=(
            "Compare two map layers side by side through GeoLibre's Swipe plugin, "
            "including a layer against the basemap itself."
        ),
        keywords=("compare", "comparison", "before", "after", "swipe", "change"),
        implementation="backend+geolibre",
        geolibre_plugins=("Swipe",),
        tools=("swipe_compare",),
    ),
    Capability(
        id="map.terrain",
        title="Terrain and elevation",
        summary=(
            "Derive relief from a DEM (hillshade, slope, aspect) and explore it with "
            "imagery in GeoLibre's terrain view."
        ),
        keywords=("terrain", "elevation", "dem", "3d", "slope", "aspect", "hillshade", "landslide"),
        implementation="backend+geolibre",
        geolibre_plugins=("Terrain",),
        tools=("hillshade", "slope", "aspect"),
    ),
    Capability(
        id="catalog.overture",
        title="Overture Maps",
        summary="Find buildings, places, and transportation data for the current area.",
        keywords=("overture", "building", "infrastructure", "road", "place"),
        implementation="geolibre-plugin",
        geolibre_plugins=("Overture Maps",),
        status="interactive_handoff",
        fallback="Open the Overture Maps plugin, extract the desired theme, then analyze the added layer with vector tools.",
    ),
)


def _terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    # Lightweight singular aliases are enough for deterministic routing without
    # adding a stemming/NLP dependency.
    terms.update(term[:-1] for term in list(terms) if len(term) > 3 and term.endswith("s"))
    return terms


def _ranked(goal: str, limit: int) -> list[Capability]:
    """Return the capabilities that best match ``goal``, best first."""
    terms = _terms(goal)
    ranked: list[tuple[int, Capability]] = []
    for capability in CAPABILITIES:
        haystack = _terms(" ".join((*capability.keywords, capability.title)))
        score = len(terms & haystack)
        if score:
            ranked.append((score, capability))
    if not ranked:
        ranked = [(0, capability) for capability in CAPABILITIES[:4]]
    ranked.sort(key=lambda item: (-item[0], item[1].id))
    return [capability for _, capability in ranked[: max(1, min(limit, 8))]]


def _tools_of(capability: Capability, registry: ToolRegistry | None) -> list[dict[str, Any]]:
    """Return the tools this capability names that ``registry`` actually has.

    The catalog is static metadata, so a capability may name a tool a given
    build does not ship (a GeoLibre-side integration, a pack that is not built
    yet). The registry is authoritative: unknown names are dropped rather than
    summarised speculatively. Without a registry at all, names are still
    reported, unsummarised, so the prose stays useful before the session wires
    one up.
    """
    if registry is None:
        return [{"name": name, "summary": None} for name in capability.tools]
    entries: list[dict[str, Any]] = []
    for name in capability.tools:
        if name not in registry:
            continue
        entries.append({"name": name, "summary": registry.get(name).summary})
    return entries


def discover(registry: ToolRegistry | None, goal: str, limit: int = 5) -> list[dict]:
    """Rank the capability catalog against ``goal`` and annotate it.

    Every result carries the capability's prose plus the registry summary of
    each tool it names, so the model sees what this build can actually execute
    instead of a catalog of names that may not exist here. ``registry`` may be
    ``None`` while the session is still assembling the tool surface.
    """
    return [
        {
            "id": capability.id,
            "title": capability.title,
            "summary": capability.summary,
            "keywords": list(capability.keywords),
            "implementation": capability.implementation,
            "tools": _tools_of(capability, registry),
            "geolibre_plugins": list(capability.geolibre_plugins),
            "status": capability.status,
            "fallback": capability.fallback,
        }
        for capability in _ranked(goal, limit)
    ]


__all__ = ["CAPABILITIES", "Capability", "discover"]
