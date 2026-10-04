# 🛰️ Spatial Intelligence

A chat-driven geospatial analyst: a live [GeoLibre](https://geolibre.app) map on
the left, a notebook on the right, and an agent that works in your files.

Ask for an outcome — *"map the flooded area in this AOI and measure the fields it
hit"* — and the agent picks the tools, finds and downloads the imagery, runs the
analysis, and puts the result on the map. Every step is visible while it runs, and
every output is a real file in the workspace you can inspect or reuse.

## 🧭 What it can do

**📡 Imagery and data**

- Search and download open imagery: STAC catalogs (Sentinel-2, Landsat, NAIP,
  Sentinel-1, DEMs), Vantor disaster releases, OpenAerialMap.
- Import local files or folders, or download by URL, into the workspace.
- Search the web for documentation, provider terms, or current facts.

**🧮 Raster analysis**

- Clip, reproject, rescale, spectral indices (NDVI, GNDVI, NDWI, NDMI, NDBI,
  NBR, EVI, SAVI), zonal statistics, terrain (hillshade, slope, aspect),
  contour, polygonize, RGB composites, COG conversion. A striped GeoTIFF is
  converted to a COG copy when the map needs one.
- Remote COG URLs are read by HTTP range request, so clipping a scene to a
  town-sized window costs a few megabytes instead of the whole band.

**✂️ Vector analysis**

- Buffer, clip, dissolve, overlay, spatial and attribute joins, select,
  aggregate, centroids, convex hull, simplify, explode, Voronoi/Delaunay,
  grids, points along lines, geometry repair.

**🤖 Local AI models**

- Segmentation (SAM family) and object detection (COCO) run locally on ONNX
  Runtime — no service to stand up. Models download once into `.models/` and are
  unloaded when idle.
- Heavy research libraries (torchgeo, terratorch, torch) can be exposed to the
  Python sandbox; the agent reads their API references on demand and writes its
  own inference code, so new models need no new tool.
- Weights always land in `.models/`, hash-verified. The sandbox itself has no
  network.

**🗺️ The map**

- The agent authors the live map: raster and vector layers, styling, heatmaps,
  swipe comparison. Changes persist to the workspace's `.geolibre.json`.
- Long operations report progress; the agent can pause and ask you a structured
  question when a choice would change the result.

**📁 The workspace**

- Everything is a file you own: inputs in `data/`, outputs in `results/`,
  projects in `maps/`, run traces in `traces/`, cells in `notebook.ipynb`.
- `run_python` runs your own Python against the same geospatial stack when no
  tool fits.

## 📦 Installation

Requires Python 3.11 or newer, and network access on first start.

```bash
git clone https://github.com/spatialint-org/spatial-intelligence.git
cd spatial-intelligence
make venv                  # creates .venv and installs the package editable
cp .env.example .env       # then set OPENAI_API_KEY (or your endpoint)
```

By hand:

```bash
uv venv .venv && uv pip install -e .                    # or: python -m venv .venv
.venv/Scripts/python.exe -m spatial_intelligence.server # Windows; .venv/bin/python elsewhere
```

Optional: the research sandbox.

```bash
uv pip install -e ".[geoai]"  # torchgeo, terratorch, torch — ~150 MB of wheels
```

Everything the app presents as built in — including local segmentation and object
detection on ONNX Runtime — installs with the base package. The `geoai` extra is
deliberately the only extra: `torch` alone is a 124 MB wheel and far more
installed, and a CUDA machine wants torch from the CUDA index rather than the CPU
wheel a plain dependency would pin, so it stays opt-in. It takes current torchgeo
on Python 3.12+ and the last compatible releases on 3.11.

GDAL ships inside rasterio; the `gdal` PyPI package (the `osgeo` bindings) is not
used and cannot be installed from a wheel.

## 🚀 Getting started

```bash
make dev        # or: .venv/Scripts/python.exe -m spatial_intelligence.server
```

The browser opens at `http://127.0.0.1:8000`.

1. **File → New workspace** and give it a name. A workspace is the folder the
   agent reads and writes.
2. **Data tab** — import files, or paste a URL to download. Imports are picked up
   by the agent automatically.
3. **Cells tab → Prompt** and ask for an outcome. For example:
   - *"Find a recent Sentinel-2 scene over Phewa Lake, Nepal with under 10% cloud,
     clip it to the lake, and map it."*
   - *"Segment the fields, water and buildings in data/scene.tif and show the
     polygons."*
   - *"How much of the AOI burned between these two dates?"*
4. Watch the progress cells; results land in `results/` and appear in the Data
   tab in the same turn.
5. **Python** in the Cells tab runs a snippet directly in the same sandbox the
   agent uses, if you would rather type code.

Notes: the first start downloads the pinned `marked` bundle for the UI (cached
afterwards). Local AI models download on first use — about 40 MB for
segmentation, 26 MB for detection.

## ⚙️ Configuration

Settings come from `.env` in the data root (loaded at startup) and from
**File → Settings**, which persists model, theme, retry cap, and step recording
to `settings.json`. **Dangerous mode** (the toggle at the right of the menu bar)
lifts the sandbox guard; leave it off unless you are running your own code.

**Model**

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Provider key (OpenAI, Azure, or a compatible endpoint). |
| `OPENAI_BASE_URL` | — | Custom OpenAI-compatible endpoint (Azure OpenAI, LiteLLM, vLLM, llama.cpp, Ollama, OpenRouter, DeepSeek). With it set, the key is optional. |
| `GEOAI_MODEL` | `openai:gpt-4o` | Model string, e.g. `anthropic:claude-sonnet-4-5`, `google-gla:gemini-2.5-pro`, `ollama:llama3.1`. |
| `GEOAI_MAX_RETRIES` | `5` | Attempts for a transient model/API failure before a run is reported as failed. Tool errors are handed to the agent instead. |
| `GEOAI_MAX_REQUESTS` | `200` | Model requests one run may spend before it stops cleanly, keeping the work already done. Raise it for a long analysis; the default is 4× pydantic-ai's own cap of 50, which a fetch-then-tile-then-map run can exceed. |
| `GEOAI_CONTEXT_WINDOW` | resolved from the model id | The window in tokens compaction targets (half of it) and where the app stops a run whose history no longer fits. Set it for a self-hosted or proxy endpoint whose model id no registry describes — otherwise 200k is assumed, and a smaller real window fails the request. |

**Paths and storage**

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEOAI_HOME` | the checkout root in a dev clone, else `~/.local/share/spatial-intelligence` | Data root: `workspaces/`, `settings.json`, `.env`, `.models/`, `.skills/`. |
| `GEOAI_MODELS_DIR` | `<GEOAI_HOME>/.models` | Where models and weights are cached. |
| `GEOAI_SKILLS_DIR` | `<GEOAI_HOME>/.skills` | Per-machine skills cache; the shipped trees live in the package. |
| `GEOAI_OFFLINE` | off | Set to `1` to refuse every download. |
| `HF_TOKEN` | — | Hugging Face token, for gated repositories. |

**Runtime**

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEOAI_PORT` | `8000` | HTTP port. |
| `GEOAI_NO_BROWSER` | off | Set to `1` to skip opening the browser on start. |
| `GEOAI_MODEL_CACHE_SIZE` | `2` | Local models kept in memory at once. |
| `GEOAI_MODEL_TTL` | `300` | Seconds a loaded model may stay idle before it can be unloaded. |
| `GEOAI_ONNX_THREADS` | up to 4 | ONNX Runtime threads for local models. |
| `GEOAI_WORKSPACE` | — | Default workspace name to open. |

## 🗂️ Workspace layout

Each workspace lives at `workspaces/<name>/`:

```
data/     inputs: imports, downloads, dropped files — read here
results/  outputs: GeoTIFF/COG, GeoJSON, tables — written here
maps/     saved .geolibre.json map projects
traces/   per-cell agent run logs: steps, plan, token usage
notebook.ipynb   your cells, plus recorded agent steps
workspace.json   manifest: outputs and version
```

Tool paths are confined to the active workspace, and writes go under `results/`,
`maps/`, or `data/` only.

## 🛠️ Development

```bash
make test      # python -m unittest discover -s tests -t . -p "test_*.py"
make release   # AppImage build (Docker); see packaging/README.md
```

```
spatial_intelligence/
  contracts/      types shared by every layer: effects, progress jobs, errors
  settings/       data root, .env, model resolution, persisted settings
  workspace/      the workspace tree, notebook document, run traces, file I/O
  map/            the live GeoLibre document, snapshots, styles, iframe bridge
  geo/            rasterio and GeoPandas services, open-data catalog clients
  ai/             model catalog, on-disk store, residency, segmentation/detection
  pythonruntime/  run_python sandbox, output store, API help, skills reader
  skills/         API references for heavy packages (authored + generated)
  tools/          the tool registry (@tool/@pack) and its packs
  agent/          prompt, model resolution, run loop, interaction protocol
  session/        notebook session, run queue, progress jobs, event bus
  server/         FastAPI app factory, routers, asset bootstrap
  web/            the browser front end (ESM modules + stylesheet)
```

Tools are declarations: `@tool(...)` on a function, or on a `@pack` class method,
registered in `tools/build.py`. Visibility, replay safety, approval, and progress
are all read from the registry rather than restated per tool. Imports point one
way — contracts ← services ← tools ← agent ← session ← server — and `web/` talks
only HTTP and SSE.
