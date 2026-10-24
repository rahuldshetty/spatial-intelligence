# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Spatial Intelligence one-directory bundle (Route B).
# Built inside packaging/Dockerfile.build; entry point is the repo-root app.py.

import os

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# The web front end is plain assets (ES modules, CSS, the marked bundle the
# build downloads), so it ships as data rather than as importable modules.
datas = [(os.path.join(ROOT, "spatial_intelligence", "web"), "spatial_intelligence/web")]
binaries = []
hiddenimports = [
    # uvicorn resolves loop/protocol modules by import string at runtime
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.lifespan.on",
    # dotenv is imported inside a try/except in settings/env.py; without it a
    # user's .env file would be ignored entirely in a frozen build.
    "dotenv",
    # Orthority is imported lazily by the orthorectification service, and its
    # OpenCV dependency carries binaries that need the cv2 hook.
    "orthority",
    "cv2",
]

# Several modules are reached only through import strings (the uvicorn app
# target, the provider/model name resolution), which PyInstaller's static
# analysis cannot see. Pull in every submodule of the package explicitly so the
# frozen bundle contains the whole app rather than only what app.py imports.
hiddenimports += collect_submodules("spatial_intelligence")

# Dynamic imports and bundled data PyInstaller cannot see statically:
# pydantic-ai resolves providers/models by name (infer_model/infer_provider),
# pydantic-ai-harness and geolibre ship data files, and the geo-stack wheels
# carry data (proj.db, GDAL resources) next to their bundled shared libraries.
for pkg in (
    "pydantic_ai",
    "pydantic_ai_harness",
    "geolibre",
    "orthority",
    "rasterio",
    "pyogrio",
    "shapely",
    "pyproj",
    "geopandas",
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# Packages that read their own dist metadata (importlib.metadata) at import.
# pydantic-ai-slim is the distribution that actually carries pydantic-ai's
# modules and version; the "pydantic-ai" wheel only depends on it.
for dist in (
    "genai_prices",
    "pydantic_ai",
    "pydantic_ai_slim",
    "pydantic_ai_harness",
    "logfire",
    "geolibre",
    "orthority",
):
    datas += copy_metadata(dist)

a = Analysis(
    [os.path.join(ROOT, "app.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="spatial-intelligence",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="spatial-intelligence",
)
