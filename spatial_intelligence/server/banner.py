"""The startup banner and this build's metadata, printed to the server's log.

:func:`~spatial_intelligence.server.run` prints it -- the entry point the Makefile,
``python -m spatial_intelligence.server`` and the packaged AppImage all share -- so
the first lines of a session say what is about to serve: the tool surface the model
gets, the model itself, and the libraries that will read the data.

Nothing here opens a session, starts a worker, or touches a workspace. The tool
table comes from a registry built against a path that is never created, because the
tool surface is a property of the build rather than of any one workspace.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from .. import __version__
from ..discovery import CAPABILITIES
from ..settings.env import api_key_env_for, app_root, list_workspaces, model_from_env
from ..tools.build import default_registry
from ..tools.registry import ToolRegistry
from ..tools.runtime import ToolRuntime
from ..workspace import Workspace

#: Column the values start at, so the block scans as a table.
LABEL_WIDTH = 13
#: Where a wrapped name list runs to.
WIDTH = 96

#: The wordmark, in block glyphs. Chosen at print time only when the stream can
#: encode them (a redirected stdout on Windows may be a single-byte code page).
BANNER_BLOCKS = """\
 ███  ████   ███  █████ █████  ███  █
█     █   █ █   █   █     █   █   █ █
 ███  ████  █████   █     █   █████ █
    █ █     █   █   █     █   █   █ █
 ███  █     █   █   █   █████ █   █ █████
█████ █   █ █████ █████ █     █     █████  ███  █████ █   █  ███  █████
  █   ██  █   █   █     █     █       █   █     █     ██  █ █   █ █
  █   █ █ █   █   ████  █     █       █   █ ███ ████  █ █ █ █     ████
  █   █  ██   █   █     █     █       █   █   █ █     █  ██ █   █ █
█████ █   █   █   █████ █████ █████ █████  ███  █████ █   █  ███  █████"""

#: The same wordmark in ASCII, for a stream that cannot take the block glyphs.
BANNER_ASCII = """\
 ###  ####   ###  ##### #####  ###  #
#     #   # #   #   #     #   #   # #
 ###  ####  #####   #     #   ##### #
    # #     #   #   #     #   #   # #
 ###  #     #   #   #   ##### #   # #####
##### #   # ##### ##### #     #     #####  ###  ##### #   #  ###  #####
  #   ##  #   #   #     #     #       #   #     #     ##  # #   # #
  #   # # #   #   ####  #     #       #   # ### ####  # # # #     ####
  #   #  ##   #   #     #     #       #   #   # #     #  ## #   # #
##### #   #   #   ##### ##### ##### #####  ###  ##### #   #  ###  #####"""

TAGLINE = "geo agent harness: workspace files, public catalogs, raster and vector analysis, live map"


def banner() -> str:
    """Return the wordmark, in block glyphs when the console can show them."""
    art = BANNER_BLOCKS if _encodable("\u2588") else BANNER_ASCII
    return f"{art}\n{TAGLINE}"


def _encodable(text: str) -> bool:
    """Whether the current stdout can encode ``text``."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _row(label: str, value: str) -> str:
    """One ``label   value`` row of the table block."""
    return f"{label:<{LABEL_WIDTH}}{value}"


def _wrap(prefix: str, items: list[str]) -> list[str]:
    """Return ``items`` comma-joined as lines that start with ``prefix``.

    A continuation line is indented to ``len(prefix)`` so the block stays a column,
    and wraps at :data:`WIDTH` so a long list does not run off a narrow terminal.
    """
    lines: list[str] = []
    current = prefix
    for index, item in enumerate(items):
        piece = item if index == 0 else f", {item}"
        if current != prefix and len(current) + len(piece) > WIDTH:
            lines.append(current)
            current = " " * len(prefix) + item
        else:
            current += piece
    lines.append(current)
    return lines


def tool_table(registry: ToolRegistry) -> list[str]:
    """Return the tool surface: totals, flags, routing, and one group per category.

    Category order is the registry's own: packs are added in the order the model
    should read them, so the table reads in that order too.
    """
    external = len(registry) - len(registry.implemented())
    flagged = [
        name for name in registry.names() if registry.requires_approval(name)
    ]
    lines = [
        _row(
            "tools",
            f"{len(registry)} registered: {len(registry.implemented())} with "
            f"implementations, {external} harness-declared",
        ),
        _row(
            "flags",
            f"{len(registry.core_names())} always visible, "
            f"{len(registry.interactive())} interactive, "
            f"{len(flagged)} needs approval ({', '.join(flagged) if flagged else 'none'}), "
            f"{len(registry.handoffs())} prose handoffs",
        ),
    ]
    lines.extend(
        _wrap(
            _row("capabilities", f"{len(CAPABILITIES)}: "),
            [cap.id for cap in CAPABILITIES],
        )
    )
    categories = registry.categories()
    lines.append(_row("by category", f"{len(categories)} groups"))
    for category, names in categories.items():
        lines.extend(_wrap(f"  {category} ({len(names)})  ", list(names)))
    return lines


def environment_table() -> list[str]:
    """Return what this build is: version, interpreter, model, data root, libraries.

    The model row reads the current environment: the server loads ``.env`` before
    printing (``validate_env``), so it names the configured model and the variable
    its key comes from, never the key itself.
    """
    model = model_from_env()
    key_var = api_key_env_for(model)
    key_state = "no key needed" if key_var is None else f"{key_var}"
    workspaces = list_workspaces()
    lines = [
        _row(
            "build",
            f"spatial-intelligence {__version__} "
            f"(python {platform.python_version()}, {sys.platform})",
        ),
        _row("model", f"{model} ({key_state})"),
        _row(
            "data root",
            f"{app_root() / 'workspaces'} "
            f"[{len(workspaces)}: {', '.join(workspaces) if workspaces else 'none yet'}]",
        ),
        _row(
            "libraries",
            f"geolibre {_version('geolibre')} | rasterio {_version('rasterio')} "
            f"(GDAL {_gdal_version()}) | geopandas {_version('geopandas')}",
        ),
    ]
    return lines


def _version(package: str) -> str:
    """Return an installed distribution's version, or a marker when it is absent."""
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _gdal_version() -> str:
    """Return the GDAL version rasterio was built against."""
    try:
        import rasterio
    except ImportError:
        return "unknown"
    return str(getattr(rasterio, "__gdal_version__", "unknown"))


def startup_report(registry: ToolRegistry | None = None) -> str:
    """Return the whole startup block: banner, environment, and the tool table.

    ``registry`` is supplied by tests; the server builds one against a workspace
    path that is never created, since the tool surface does not depend on a
    workspace's contents.
    """
    if registry is None:
        registry = default_registry(
            ToolRuntime(workspace=Workspace(app_root() / "workspaces" / "_introspection"))
        )
    blocks = [banner(), "", *environment_table(), "", *tool_table(registry)]
    return "\n".join(blocks)


def announce(report: str | None = None) -> str:
    """Print the startup block and return it, tolerating a stream it cannot encode.

    A console that cannot render a glyphs in the report (a workspace name with a
    non-ASCII character, say) degrades the character rather than failing on
    startup, which is what a redirected stdout on Windows would otherwise do.
    """
    text = startup_report() if report is None else report
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        sys.stdout.write(f"{text}\n")
    except UnicodeEncodeError:
        sys.stdout.write(text.encode(encoding, "replace").decode(encoding, "replace") + "\n")
    sys.stdout.flush()
    return text


__all__ = [
    "BANNER_ASCII",
    "BANNER_BLOCKS",
    "TAGLINE",
    "announce",
    "banner",
    "environment_table",
    "startup_report",
    "tool_table",
]
