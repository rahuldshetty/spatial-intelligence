# Packaging: Spatial Intelligence as a single Linux AppImage (Route B)

Builds a self-contained `spatial-intelligence-<version>-x86_64.AppImage`: Python,
the full geospatial stack (rasterio, GeoPandas, pyogrio, pyproj, shapely), the
web front end, and the `run_python` sandbox bundled in one file. No install
step, no pip, no network needed at setup — copy it to a machine and run.

## Build

Requires Docker with BuildKit. From the repo root:

```bash
packaging/build-appimage.sh            # writes dist/spatial-intelligence-<version>-x86_64.AppImage
```

or directly:

```bash
docker build -f packaging/Dockerfile.build --output dist .
```

The build runs inside `quay.io/pypa/manylinux_2_28_x86_64` (glibc 2.28), so
the artifact runs on any distro from roughly 2019 on (Ubuntu 20.04+,
Debian 10+, Fedora, Arch, openSUSE). Building on a newer base would inherit
its glibc and break older hosts — keep the manylinux base.

Two things the image does that the dev checkout does not:

- It downloads the pinned `marked` Markdown bundle into
  `spatial_intelligence/web/vendor/` before PyInstaller runs. The bundle
  directory inside a mounted AppImage is read-only, so the app's first-launch
  bootstrap cannot write there — the asset has to ship inside the image.
  `spatial_intelligence/web/vendor` is excluded from the build context so a
  copy on the build host cannot decide what ships.
- It pins CPython 3.12 (`uv venv --python 3.12`) rather than taking uv's
  default, so the bundled interpreter is the one the project is developed on.

### Registry access and the base image

`pypa` publishes the manylinux images **only on quay.io** — there is no Docker
Hub mirror. The base is therefore a build argument, so a blocked or throttled
registry never needs a Dockerfile edit:

```bash
packaging/build-appimage.sh dist \
    --build-arg BASE_IMAGE=quay.io/pypa/manylinux_2_28_x86_64:2026.09.05-1
```

Pinning a dated tag instead of `:latest` also makes a build reproducible; the
available tags are listed at
<https://quay.io/repository/pypa/manylinux_2_28_x86_64?tab=tags>.

A failure while *resolving the base* (`failed to resolve source metadata for
quay.io/...: read: connection reset by peer`) is a problem on the build host's
network path, not in this Dockerfile — quay.io answers an unauthenticated
`GET /v2/` with `401`. In order:

1. Retry. Transient resets are the common case:
   `docker pull quay.io/pypa/manylinux_2_28_x86_64:latest`.
2. Confirm reachability from the host running Docker:
   `curl -sI https://quay.io/v2/` should print `401 Unauthorized`. A timeout or
   a reset means DNS, a proxy, or a firewall on that path.
3. If the daemon is behind a proxy, configure it for the *daemon* as well
   (`HTTPS_PROXY` in `/etc/systemd/system/docker.service.d/http-proxy.conf` on
   Linux, or Docker Desktop → Settings → Resources → Proxies). On WSL an
   oversized MTU is a known cause of TLS resets: `ip link set dev eth0 mtu 1400`.
4. Use a different base. Any image with **glibc 2.28 or older** works, because
   the interpreter comes from uv (`uv venv --python 3.12` fetches
   python-build-standalone) rather than from the base:
   `--build-arg BASE_IMAGE=rockylinux:8`. Rocky 8 is glibc 2.28, so the
   portability floor is unchanged; the only requirement is that `curl` exists in
   the image for the appimagetool download. Third-party GHCR mirrors of the
   manylinux images exist too, but they are not published by pypa — check the
   digest before trusting one.

Do not move to a newer base to work around a registry problem: the AppImage
inherits the base's glibc, and a newer one silently raises the minimum host
version.

### Other architectures

For aarch64, build on an ARM host (or with emulation) with the aarch64 base and
swap the appimagetool download:

```bash
packaging/build-appimage.sh dist \
    --build-arg BASE_IMAGE=quay.io/pypa/manylinux_2_28_aarch64:latest
```

then change `appimagetool-x86_64.AppImage` to `appimagetool-aarch64.AppImage` in
`packaging/Dockerfile.build`. The interpreter needs no change: the build uses
uv's python-build-standalone at `/opt/venv`, not the base's `/opt/python`.

## Run

```bash
chmod +x spatial-intelligence-0.0.1-x86_64.AppImage
./spatial-intelligence-0.0.1-x86_64.AppImage
```

The server starts on `http://127.0.0.1:8000/` and opens the browser.
If the desktop lacks FUSE2 (`libfuse2`), run:

```bash
./spatial-intelligence-0.0.1-x86_64.AppImage --appimage-extract-and-run
```

## Data and configuration

Everything the app writes lives under one data root (see `GEOAI_HOME` in
`spatial_intelligence/settings/env.py`):

| Path | Purpose |
| --- | --- |
| `~/.local/share/spatial-intelligence/workspaces/` | workspaces (`data/`, `results/`, `maps/`, `traces/`) |
| `~/.local/share/spatial-intelligence/settings.json` | model, theme, retries, dangerous mode |
| `~/.local/share/spatial-intelligence/.env` | provider key and model (same schema as `.env.example`) |

The root was `~/.local/share/geo-ai` before the project was renamed. An
installation that already has data there keeps using it: the old directory wins
while it exists and the new one does not, so workspaces and settings survive the
rename, and a fresh install gets the new name. Only a checkout (a `.git` or
`workspaces/` next to the package) writes next to the sources, which a frozen
bundle never is.

Environment overrides (as in the dev checkout): `GEOAI_HOME` (data root),
`GEOAI_PORT` (default 8000), `GEOAI_NO_BROWSER=1` (headless),
`GEOAI_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`.

First run: copy `.env.example` to `~/.local/share/spatial-intelligence/.env`
(`~/.local/share/geo-ai/.env` for an install predating the rename) and fill in
the provider key — or set the variables in the environment before launching.
Without `OPENAI_BASE_URL`, opening/creating a workspace fails with a
provider error ("Set the `OPENAI_API_KEY` environment variable...") until a
key is configured; with a base URL pointing at a local endpoint the key is
optional. The rest of the app (map, python cells, file data) works without one.

## Build pipeline (Dockerfile.build, stage by stage)

1. `manylinux_2_28` + CPython 3.12, `uv pip install . pyinstaller`
2. the pinned Marked bundle, fetched into the source tree
3. `pyinstaller packaging/spatial-intelligence.spec` — one-dir bundle at
   `dist/spatial-intelligence/`. The spec collects everything PyInstaller
   cannot see statically: the `spatial_intelligence` submodules reached only
   through import strings, the `web/` assets, pydantic-ai's name-resolved
   providers/models, pydantic-ai-harness and geolibre data, and the geo-stack
   wheels' data files (proj.db, GDAL resources) plus uvicorn's
   import-string-loaded loop/protocol modules.
4. AppDir assembly (`usr/bin/spatial-intelligence`, desktop file, icon,
   `.DirIcon`) and `appimagetool --appimage-extract-and-run` to produce the
   AppImage.
5. `FROM scratch AS artifacts` — the `--output` export stage.
