"""Optional heavy packages the sandbox may import when they are installed.

The core sandbox stack ships with the app; these are the research libraries a
workspace only pays for when it installs the ``geoai`` extra. The table is the
single source for four answers: what ``run_python`` may import, what the executor
binds, what ``python_help`` can resolve, and what the system prompt advertises.
Only installed packages are ever advertised — naming an absent one invites the
agent to plan around something this machine cannot run.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from dataclasses import dataclass
from functools import lru_cache

#: The extra that installs every optional package below.
EXTRA = "geoai"
#: Most package lines the prompt block may carry before it summarises the rest.
MAX_PROMPT_LINES = 8


@dataclass(frozen=True, slots=True)
class SandboxPackage:
    """One optional library: how to import it and what it is for."""

    key: str
    import_name: str
    purpose: str
    roots: tuple[str, ...] = ()


PACKAGES: tuple[SandboxPackage, ...] = (
    SandboxPackage(
        key="torchgeo",
        import_name="torchgeo",
        purpose="geospatial datasets, samplers, multispectral pretrained weights, Lightning tasks",
    ),
    SandboxPackage(
        key="terratorch",
        import_name="terratorch",
        purpose="EO foundation models (Prithvi, SatMAE, ...) with task heads and fine-tuning",
    ),
    SandboxPackage(
        key="torch",
        import_name="torch",
        purpose="the tensor library both of the above run on",
    ),
    SandboxPackage(
        key="torchvision",
        import_name="torchvision",
        purpose="vision backbones and pretrained RGB weights",
    ),
    SandboxPackage(
        key="timm",
        import_name="timm",
        purpose="backbone zoo; the way torchgeo's multispectral weights are loaded",
    ),
    SandboxPackage(
        key="transformers",
        import_name="transformers",
        purpose="Hugging Face model classes and processors",
    ),
    SandboxPackage(
        key="segment_geospatial",
        import_name="segment_geospatial",
        purpose="SamGeo: text- and point-prompted segmentation of imagery",
    ),
)

#: Kinds that make a snippet long-running: the prompt says so, so a plan can
#: account for minutes per tile instead of discovering it mid-run.
HEAVY_KEYS = frozenset({"torch", "torchgeo", "terratorch"})


@lru_cache(maxsize=None)
def installed(package: SandboxPackage) -> bool:
    """Whether ``package`` can be imported in this interpreter."""
    try:
        return importlib.util.find_spec(package.import_name) is not None
    except (ImportError, ValueError):  # a broken install is not an installed one
        return False


def installed_packages() -> tuple[SandboxPackage, ...]:
    """Every optional package present on this machine, in table order."""
    return tuple(package for package in PACKAGES if installed(package))


def optional_roots() -> frozenset[str]:
    """Import roots the optional packages add to the sandbox allowlist."""
    return frozenset(
        root
        for package in installed_packages()
        for root in (package.roots or (package.import_name,))
    )


def package_for(root: str) -> SandboxPackage | None:
    """Return the optional package an import root belongs to, if any."""
    for package in PACKAGES:
        if root == package.import_name or root in package.roots:
            return package
    return None


def version(package: SandboxPackage) -> str:
    """Return the installed version, or an empty string when it cannot be read."""
    try:
        return importlib.metadata.version(package.import_name.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        return ""


def import_error(root: str) -> str | None:
    """Return the message for importing ``root`` when it is a missing extra.

    A known optional package that is not installed deserves better than "not
    allowed": the snippet would be legal here, so the answer is the install
    command rather than a dead end.
    """
    package = package_for(root)
    if package is None or installed(package):
        return None
    return (
        f"{package.import_name} is not installed on this machine; install it with "
        f"pip install 'spatial-intelligence[{EXTRA}]' to use it in run_python"
    )


def bindings() -> dict:
    """Return ``{name: module}`` for every installed optional package.

    Imported lazily and individually: a package that fails to import is skipped
    rather than taking the whole sandbox namespace down with it.
    """
    bound: dict = {}
    for package in installed_packages():
        try:
            module = __import__(package.import_name)
        except Exception:  # noqa: BLE001 - a broken optional install is ignored
            continue
        bound[package.import_name] = module
    return bound


def prompt_block() -> str:
    """Return the system-prompt lines for what is installed, or ``''``.

    Empty when nothing is installed, so a lean install's prompt is byte-identical
    to one that never heard of these packages.
    """
    present = installed_packages()
    if not present:
        return ""

    lines = ["- Installed on this machine and importable from run_python:"]
    for package in present[:MAX_PROMPT_LINES]:
        stamp = version(package)
        label = f"{package.import_name} {stamp}".strip()
        lines.append(f"  {label} — {package.purpose}")
    if len(present) > MAX_PROMPT_LINES:
        lines.append(f"  ... and {len(present) - MAX_PROMPT_LINES} more (see skill())")

    lines.append(
        "  Know the interface, not the API: skill(\"torchgeo\") maps a package and "
        "skill(query=\"...\") searches it; python_help(\"torchgeo.models.ResNet18_Weights\") "
        "gives one symbol's signature. Weights: ai_fetch_model(...) downloads into "
        ".models, because the sandbox itself has no network."
    )
    if any(package.key in HEAVY_KEYS for package in present):
        lines.append(
            "  Heavy models: minutes per tile on CPU and RAM in the gigabytes — tile small, "
            "pass bounds, and keep one model loaded across cells."
        )
    return "\n".join(lines)


__all__ = [
    "EXTRA",
    "HEAVY_KEYS",
    "MAX_PROMPT_LINES",
    "PACKAGES",
    "SandboxPackage",
    "bindings",
    "import_error",
    "installed",
    "installed_packages",
    "optional_roots",
    "package_for",
    "prompt_block",
    "version",
]
