# Spatial Intelligence

A geospatial-analysis agent in a custom web UI: a live
[GeoLibre](https://geolibre.app) map on the left, and a Jupyter/Colab-style
notebook panel on the right. Prompt an agent that drives the map and runs
GDAL/rasterio and GeoPandas work behind the scenes, or run raw Python directly.

## Setup

```bash
make venv          # creates .venv (uv, or python -m venv) and installs it editable
cp .env.example .env    # Windows: copy .env.example .env
```

By hand, if you prefer:

```bash
uv venv .venv && uv pip install -e .          # or: python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .  # Windows; .venv/bin/python on macOS/Linux
```

`make venv` installs the package editable, so `spatial_intelligence` imports
from anywhere and source edits take effect without reinstalling. `make dev` and
`make test` also export `PYTHONPATH=.`, so the checkout always wins over any
installed copy.

`.env` settings (loaded automatically at startup):

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Provider API key (OpenAI / Azure / compatible endpoints). |
| `OPENAI_BASE_URL` | Optional custom OpenAI-compatible endpoint (Azure OpenAI, LiteLLM, vLLM, llama.cpp, Ollama, OpenRouter, DeepSeek). Leave empty for OpenAI's default. With a base URL set, `openai:` and `openai-chat:` use the Chat Completions API and `openai-responses:` the Responses API; the API key becomes optional. |
| `GEOAI_MODEL` | Model string. Default `openai:gpt-4o`. Also `openai-chat:qwen3` (a Chat Completions endpoint), `openai-responses:gpt-5`, `anthropic:claude-sonnet-4-5`, `google-gla:gemini-2.5-pro`, `ollama:llama3.1`. |
| `GEOAI_WORKSPACE` | Optional default workspace name (overrides the notebook default). |
| `GEOAI_HOME` | Optional data root (defaults to the repo root in a checkout). |
| `GEOAI_MAX_RETRIES` | Transient model/API attempts per prompt run before reporting an error (default `5`). |
| `GEOAI_PORT` | HTTP port (default `8000`). |
| `GEOAI_NO_BROWSER` | Set to `1` to skip opening the browser on start. |

## Run

```bash
make dev            # or: make run
```

equivalent to:

```bash
.venv/Scripts/python.exe -m spatial_intelligence.server
```

On the first start the app downloads the pinned `marked` browser bundle into
`spatial_intelligence/web/vendor/`. That asset is ignored by Git and reused on
later starts, so only the initial startup needs network access.

## Test

```bash
make test
```

runs the whole suite with stdlib `unittest` discovery:

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

That runs `tests/spatial_intelligence/`, which covers the app: the tool
registry and its packs, the workspace/notebook/trace documents, the map and
layer services, the raster/vector/catalog services, the `run_python` sandbox,
the session (cells, runs, progress jobs, interactions, retries), the server
routes, and the web components.

## Usage

- **Map (left)** — the live GeoLibre map. Agent tool calls mutate it in place and
  it persists to `maps/current.geolibre.json` after every change. It follows the
  app theme and is only rebuilt when you switch workspace, so it is never torn
  down mid-session.
- **File menu (top)** — `New` creates a workspace, `Open` switches to an existing
  one, `Save` persists the current map project, `Settings` edits the model, the
  theme, the transient-attempt cap, and whether agent steps are recorded. The
  toggle on the right of the menu bar enables dangerous mode.
- **Cells tab** — a notebook-like prompt. Pick `Prompt` to send a message to the
  agent (which runs the tools), or `Python` to execute a command directly in the
  kernel. Prompt requests serialize as Markdown, and the agent's generated
  tool calls, tool outputs, and final response are appended as provenance cells
  in the saved notebook without duplicating them in the interactive Cells tab.
  This recording can be disabled under File → Settings. When a meaningful choice
  is ambiguous, the agent can pause the prompt and show a structured radio,
  multi-select, confirmation, or custom-text form; the submitted choice is saved
  in the notebook and the same agent conversation resumes. Each prompt cell is
  independent: it starts a fresh plan and does not replay prior cells' messages,
  while the map state carries over. The run's token usage is shown next to its
  output. Traces (steps, plan, usage) persist as JSONL under `traces/`, so a
  reopened workspace shows each cell's plan and progress again. The agent keeps a
  per-cell task plan (via Pydantic AI Harness `Planning`, captured in the trace)
  and compacts its context in-run (clear old tool results, then summarize).
  A failing tool call never ends the prompt: a rejected call or a raised tool
  error comes back to the agent as that call's own failed result, so it can
  correct course and continue, while a transient network failure (a dropped
  connection, a timeout, a 5xx) is retried in place before the agent is told
  anything.
- **Progress** — every long-running operation (a URL download, a raster warp, a
  catalog search, a `run_python` snippet) reports a progress job to the browser:
  one progress cell per job, attached to the cell that started it, with a label,
  a state, a percentage or an indeterminate bar, and the resulting path when it
  finishes. Jobs whose owning cell is gone render as standalone cells.
- **Data tab** — a tree explorer of the workspace files, with **Import files** and
  **Import folder** buttons that copy into the workspace's `data/` folder (plus a
  URL download field with progress shown in the Cells tab). Imported data is
  automatically included in the agent's context, so a later prompt can refer to
  it directly.
- **Open disaster imagery** — the agent can search the same public Vantor Open
  Data and OpenAerialMap contracts used by GeoLibre, present matching scenes as
  an inline choice form, download the selected source COG into `data/`, and add
  that workspace-local copy to the live map. Each download appears as its own
  progress cell; multiple independent downloads can run in parallel. Search
  metadata is cached under `traces/catalog-scenes.json` so a paused selection
  survives a restart.
- **Satellite imagery and elevation (STAC)** — the same flow covers any STAC
  catalog the GeoLibre STAC panel browses. `list_stac_catalogs` and
  `search_stac_collections` find a collection (`sentinel-2-l2a`, `landsat-c2-l2`,
  `naip`, `sentinel-1-grd`, `cop-dem-glo-30`, …), `search_stac_scenes` searches
  it by area, date range, and cloud cover — newest or clearest first — and the
  returned `scene_key` feeds the same download and `add_catalog_scene` tools the
  disaster catalogs use. AWS Earth Search assets are public; Planetary Computer
  asset URLs are signed with a per-collection SAS token that is fetched when a
  URL is used. A scene keeps the catalog's own unsigned href — that is what its
  `scene_key` and its cached row are built from — so no token reaches the scene
  cache or a workspace file, and a scene found yesterday still downloads today.
- **Vector toolbox** — a GeoPackage is read layer by layer: `list_layers` names
  what a container holds, and `read_vector`/`export_vector` take a `layer` to pick
  one. The analysis tools mirror GeoLibre's own vector toolbox, so the same
  operation produces the same result here as in the app: buffer (optionally
  dissolved), clip, dissolve, overlay (intersection / difference / union),
  spatial and attribute joins, select by value or by location, aggregate by
  attribute, centroids, convex hull and bounding box, simplify, explode, Voronoi
  and Delaunay diagrams, regular grids (rectangle or hexagon), points along
  lines, and `check_geometry`/`fix_geometry` for invalid geometry. Units are
  stated per tool: buffer measures in meters through a local UTM projection,
  while simplify's tolerance, points_along's interval, and a grid's cell size are
  in layer units (degrees for `EPSG:4326`).
- **Rasters land on the map as COGs** — the embedded app loads a Cloud-Optimized
  GeoTIFF, and a striped one is handed to its Python sidecar converter, which this
  harness does not run. `add_raster` therefore converts a local GeoTIFF to a COG
  copy in `results/` (`<name>_cog.tif`, written with rasterio's COG driver), reports
  it as its own progress cell, records it in the outputs manifest, and loads
  that copy, recording both paths in the layer metadata; a file that is already a
  COG is left alone. The tools that produce map-facing products — `spectral_index`,
  `slope`, `aspect`, `hillshade`, `rescale`, `compose_rgb` — write COGs directly.
- **Raster analysis** — beyond clip/reproject/rescale, the raster pack derives
  products in one call each: `spectral_index` (NDVI, GNDVI, NDWI, NDMI, NDBI,
  NBR, EVI, SAVI) reading either band numbers of one file or per-band COGs the
  catalogs publish, `zonal_stats` (a band summarized per zone polygon, joined
  back onto the zones), `hillshade`/`slope`/`aspect` (a DEM, with a geographic
  raster's spacing converted to meters), `polygonize` (a classified raster to
  polygons), and `compose_rgb` (three single-band bands stacked into one colour
  file, which is what makes a per-band sensor like Landsat render in colour).
  `contour` turns a continuous surface into isolines (a DEM into 10 m contours,
  with a pixel tolerance for the marching-squares stair-steps). `add_raster` takes
  `bands` to choose what is drawn, and `add_heatmap` renders a point layer as a
  density surface. Rasters can be read straight from a remote
  COG URL: `clip`, `raster_info`, `raster_stats`, and `sample_point` go through
  GDAL's HTTP range reads, so clipping a catalog scene to a town-sized window
  fetches a few megabytes instead of the whole ~200 MB band (tools that must read
  every pixel refuse a remote source above 50 M pixels and ask for a clip).
  `clip` takes longitude/latitude bounds like every other tool and converts them
  into a projected raster's CRS, so a UTM satellite tile is clipped with the same
  AOI the rest of the harness speaks. `swipe_compare` sets up GeoLibre's swipe
  control between two layers (or a layer and `__basemap__`) for before/after
  comparison.

## Package layout

```
spatial_intelligence/
  contracts/      types shared by every layer: effects, progress jobs, errors
  settings/       data root, .env, model resolution, persisted settings
  workspace/      the workspace tree, notebook document, run traces, file I/O
  map/            the live GeoLibre document, snapshots, the style model, the iframe bridge
  geo/            rasterio and GeoPandas services, open-data catalog clients
  pythonruntime/  the run_python sandbox, output store, and API help
  tools/          the tool registry (@tool/@pack), the runtime, and the packs
  agent/          prompt, model resolution, run loop, interaction protocol
  session/        notebook session, run queue, progress jobs, event bus, facade
  server/         FastAPI app factory, routers, asset bootstrap
  web/            the ESM front end (pages + components) and the stylesheet
```

Rules the layers keep:

- **A tool is a declaration.** `@tool(category=..., effects=...)` on a function,
  or on a `@pack` class method when the tool needs services. `tools/build.py`
  states which packs exist and in what order. Everything else about a tool —
  whether it is always visible, whether a failed run may be replayed after it,
  whether it needs approval, whether it reports progress — is read from the
  registry, never restated in a second list.
- **Services are plain functions or classes with explicit dependencies.** They
  receive a workspace, a job, or a reporter; they never read ambient state.
- **Tools report, they do not render.** Report progress with
  `rt.reporter.job(kind, label)` and record a written output with
  `rt.record_artifact(path)`; the session turns both into browser events.
- **Layering**: contracts ← services ← tools ← agent ← session ← server, with
  `web/` talking only HTTP and SSE. Imports point one way.
- **A layer style lives in two places.** GeoLibre resolves a layer as
  `defaults < layer.style < project.styles[layerId]`, and its app re-materialises
  the project-level copy on every save — so a writer that touches only
  `layer.style` is silently outranked. `map/styles.py` owns every style write:
  it translates foreign key names, rejects keys the app would ignore, and keeps
  both copies in sync.

## Workspace layout

Each workspace lives at `workspaces/<name>/`:

```
data/     user inputs (imports, downloads, dropped files) — read here
results/  your outputs (GeoTIFF/COG, GeoJSON, tables) — write here
maps/     saved .geolibre.json projects
traces/   per-prompt-cell agent run logs (.jsonl): steps, messages, token usage
notebook.ipynb  user cells plus optionally recorded agent tool/response cells
workspace.json   manifest (outputs + version)
```

## Notes

- `gdal_translate` copies a raster with GDAL creation options (`TILED=YES`,
  `COMPRESS=ZSTD`, `PREDICTOR=2`, `BIGTIFF=YES`, `driver=COG`) — the settings no
  other tool exposes. It goes through `rasterio.shutil.copy`, i.e. the GDAL rasterio
  bundles (3.10.3), so nothing extra is needed: the `gdal` PyPI package (the `osgeo`
  bindings) publishes **no wheels on any platform** and is not used here. Arguments
  that are not creation options (subsetting, resizing, rescaling, pixel types) are
  refused with the tool that does that job, rather than silently ignored by GDAL.
- `raster_info` reports `cog`: whether a file is readable as a Cloud-Optimized
  GeoTIFF, which is what the embedded map needs. COG validation and creation are
  also available to generated code through `rio-cogeo`.
- `run_python` runs in the kernel process behind a cooperative AST guard: the
  geospatial stack and basic stdlib are importable — including scikit-image,
  scipy, tifffile, and PIL, so image processing (filters, morphology,
  segmentation, texture, blob detection) is available to generated code without a
  dedicated tool — while subprocess, network, dynamic execution, and raw command
  calls are rejected. Snippets run with the
  working directory at the active workspace root, so a relative path means the
  same thing inside a snippet as it does in every other tool (`data/x` is
  `<workspace>/data/x`); snippets are serialized while they hold that directory.
  Dangerous mode (the menu-bar toggle) lifts the guard entirely. `band_math`
  evaluates a NumPy expression and therefore also requires approval;
  `run_python` with numpy is the alternative while dangerous mode is off.
- Raster and vector tools write under `results/`, `maps/`, or `data/` only, and
  every tool path is confined to the active workspace.

## Packaging

`packaging/` builds a self-contained Linux AppImage of this app — a PyInstaller
one-dir bundle built inside a manylinux container and wrapped with appimagetool:

```bash
make release          # -> dist/spatial-intelligence-<version>-x86_64.AppImage
```

The spec collects everything PyInstaller cannot see statically: the
`spatial_intelligence` submodules, the `web/` assets, pydantic-ai's
name-resolved providers, geolibre/pydantic-ai-harness data, and the geo-stack's
data files (proj.db, GDAL). The image also fetches the pinned `marked` bundle at
build time, because the bundle directory inside a mounted AppImage is read-only
and the app cannot download it on first launch. `packaging/README.md` has the
build stages, the data root, and the aarch64 notes.
