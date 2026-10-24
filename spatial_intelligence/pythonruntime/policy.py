"""The run_python import and helper policy selected in Tool Configuration.

The sandbox has two independent surfaces the user can narrow:

* **Packages** — import roots and pre-bound module aliases. Disabling
  ``numpy`` refuses ``import numpy`` in safe mode and removes both ``numpy``
  and ``np`` from the snippet namespace.
* **Methods** — the convenience facades ``ws``, ``vision``, and ``models``.
  A disabled method raises when called; when every method of a facade is
  disabled the facade is not bound at all.

Tool internals are deliberately unaffected: the packs import their own
dependencies at module import time, and only ``run_python`` consults this
policy. Dangerous mode still bypasses the import guard, as it always has.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import packages


@dataclass(frozen=True, slots=True)
class ImportOption:
    """One package/import root the runtime page lists."""

    name: str
    group: str
    purpose: str = ""


@dataclass(frozen=True, slots=True)
class MethodOption:
    """One helper method the runtime page lists."""

    name: str
    group: str
    purpose: str


#: Core import roots, grouped for the configuration page. Every root in
#: ``sandbox.ALLOWED_IMPORTS`` must appear here (a test pins that), so the page
#: cannot silently omit something safe mode allows.
CORE_IMPORTS: tuple[ImportOption, ...] = (
    # Raster, vector, and array data.
    ImportOption("numpy", "Data & rasters", "array math and band operations"),
    ImportOption("pandas", "Data & rasters", "tabular data and attribute tables"),
    ImportOption("geopandas", "Data & rasters", "vector layers and geometry frames"),
    ImportOption("rasterio", "Data & rasters", "raster I/O, windows, and masks"),
    ImportOption("rioxarray", "Data & rasters", "xarray rasters with CRS aware operations"),
    ImportOption("xarray", "Data & rasters", "labelled N-dimensional arrays"),
    ImportOption("shapely", "Data & rasters", "geometry construction and predicates"),
    ImportOption("pyproj", "Data & rasters", "coordinate reference systems and transforms"),
    # Image processing and science.
    ImportOption("skimage", "Imaging & science", "filters, morphology, and segmentation"),
    ImportOption("scipy", "Imaging & science", "ND image filters and statistics"),
    ImportOption("tifffile", "Imaging & science", "TIFF tags and plain image stacks"),
    ImportOption("PIL", "Imaging & science", "image decoding and RGB conversion"),
    ImportOption("rio_cogeo", "Imaging & science", "COG validation and creation"),
    # Standard library: one flat group, because these are enabling primitives
    # rather than features a user chooses between.
    ImportOption("os", "Standard library"),
    ImportOption("sys", "Standard library"),
    ImportOption("pathlib", "Standard library"),
    ImportOption("shutil", "Standard library"),
    ImportOption("time", "Standard library"),
    ImportOption("glob", "Standard library"),
    ImportOption("csv", "Standard library"),
    ImportOption("tempfile", "Standard library"),
    ImportOption("json", "Standard library"),
    ImportOption("math", "Standard library"),
    ImportOption("re", "Standard library"),
    ImportOption("collections", "Standard library"),
    ImportOption("datetime", "Standard library"),
    ImportOption("functools", "Standard library"),
    ImportOption("itertools", "Standard library"),
    ImportOption("statistics", "Standard library"),
    ImportOption("fractions", "Standard library"),
    ImportOption("decimal", "Standard library"),
    ImportOption("copy", "Standard library"),
    ImportOption("random", "Standard library"),
    ImportOption("string", "Standard library"),
    ImportOption("typing", "Standard library"),
    ImportOption("enum", "Standard library"),
    ImportOption("contextlib", "Standard library"),
    ImportOption("warnings", "Standard library"),
    ImportOption("uuid", "Standard library"),
    ImportOption("textwrap", "Standard library"),
    ImportOption("struct", "Standard library"),
    ImportOption("dataclasses", "Standard library"),
    ImportOption("xml", "Standard library"),
    ImportOption("bisect", "Standard library"),
    ImportOption("heapq", "Standard library"),
    ImportOption("operator", "Standard library"),
    ImportOption("numbers", "Standard library"),
)

#: Alias -> import root. Disabling a package removes every alias, so the
#: namespace cannot keep a live module handle the guard would refuse to import.
ALIAS_ROOTS: dict[str, str] = {
    "np": "numpy",
    "pd": "pandas",
    "gpd": "geopandas",
    "xr": "xarray",
}


METHOD_OPTIONS: tuple[MethodOption, ...] = (
    MethodOption(
        "ws.resolve",
        "Workspace helper",
        "resolve a workspace-relative path with confinement checks",
    ),
    MethodOption(
        "ws.read_text",
        "Workspace helper",
        "read a text file through the confined workspace facade",
    ),
    MethodOption(
        "ws.write_text",
        "Workspace helper",
        "write a text file under results/, maps/, or data/",
    ),
    MethodOption(
        "ws.list_files",
        "Workspace helper",
        "list files in the workspace",
    ),
    MethodOption(
        "vision.ask",
        "Vision helper",
        "ask the configured image-capable model about the map or a file",
    ),
    MethodOption(
        "models.dir",
        "Model cache",
        "return the model cache root or one repository directory",
    ),
    MethodOption(
        "models.path",
        "Model cache",
        "return a downloaded model file's absolute path",
    ),
    MethodOption(
        "models.list",
        "Model cache",
        "list cached repositories or the files in one",
    ),
)

#: Default methods are every method above.
ALL_METHODS = frozenset(option.name for option in METHOD_OPTIONS)


@dataclass(frozen=True, slots=True)
class PythonPolicy:
    """What one session's ``run_python`` may import and which helpers it binds."""

    disabled_packages: frozenset[str] = frozenset()
    disabled_methods: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, config: dict | None) -> "PythonPolicy":
        """Build the policy from a normalized (or raw) tool configuration."""
        python = config.get("python") if isinstance(config, dict) else {}
        if not isinstance(python, dict):
            python = {}
        packages_disabled = python.get("disabled_packages")
        methods_disabled = python.get("disabled_methods")
        return cls(
            disabled_packages=frozenset(
                str(name) for name in packages_disabled if str(name).strip()
            )
            if isinstance(packages_disabled, (list, tuple, set, frozenset))
            else frozenset(),
            disabled_methods=frozenset(
                str(name) for name in methods_disabled if str(name).strip()
            )
            if isinstance(methods_disabled, (list, tuple, set, frozenset))
            else frozenset(),
        )

    def allows_import(self, root: str) -> bool:
        """Whether ``import root`` is allowed in safe mode."""
        if root in self.disabled_packages:
            return False
        package = packages.package_for(root)
        return package is None or package.import_name not in self.disabled_packages

    def allows_method(self, name: str) -> bool:
        """Whether a facade method is callable in this session."""
        return name not in self.disabled_methods

    def any_method(self, prefix: str) -> bool:
        """Whether any known method under ``prefix`` remains enabled."""
        return any(
            option.name.startswith(prefix) and self.allows_method(option.name)
            for option in METHOD_OPTIONS
        )

    def summary(self) -> dict:
        """Return the disabled sets for diagnostics and the prompt note."""
        return {
            "disabled_packages": sorted(self.disabled_packages),
            "disabled_methods": sorted(self.disabled_methods),
        }


def import_groups() -> list[str]:
    """Package group labels in display order."""
    return ["Data & rasters", "Imaging & science", "Standard library", "Optional extras"]


def method_groups() -> list[str]:
    """Method group labels in display order."""
    return ["Workspace helper", "Vision helper", "Model cache"]


__all__ = [
    "ALIAS_ROOTS",
    "ALL_METHODS",
    "CORE_IMPORTS",
    "ImportOption",
    "METHOD_OPTIONS",
    "MethodOption",
    "PythonPolicy",
    "import_groups",
    "method_groups",
]
