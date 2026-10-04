"""The ``run_python`` guard: an AST check over imports and dangerous calls.

This is what keeps the sandbox in safe mode: a snippet may import the
geospatial stack plus basic stdlib (``os``, ``sys``, ``pathlib``, ``shutil``,
...), but not subprocess, network, dynamic execution, or raw command calls, so
the agent stays on the structured, workspace-confined path. Dangerous mode
(i.e. an approved runtime) bypasses the check entirely.

Optional heavy packages (see :mod:`~spatial_intelligence.pythonruntime.packages`)
join the allowlist only when they are installed, and a missing one gets its
install command instead of a flat refusal.
"""

from __future__ import annotations

import ast

from . import packages

#: Module roots the agent may import inside ``run_python`` in safe mode: the
#: geospatial stack plus basic stdlib (``os``, ``sys``, ``pathlib``, ``shutil``,
#: ...). Dangerous escape vectors (subprocess, socket, importlib, urllib,
#: requests, ctypes, ...) stay rejected.
ALLOWED_IMPORTS = frozenset({
    "numpy", "pandas", "geopandas", "rasterio", "rioxarray", "xarray",
    "shapely", "pyproj",
    # Image processing and scientific routines: skimage/scipy are what generated
    # snippets use for filtering, morphology, segmentation, and statistics.
    # Reading a raster over HTTP is already possible through rasterio, so these
    # add no new class of access.
    "skimage", "scipy", "tifffile", "PIL",
    # COG validation/creation, the tool GeoLibre's own error message names.
    "rio_cogeo",
    "os", "sys", "pathlib", "shutil", "time", "glob", "csv", "tempfile",
    "json", "math", "re", "collections", "datetime", "functools", "itertools",
    "statistics", "fractions", "decimal", "copy", "random", "string", "typing",
    "enum", "contextlib", "warnings", "uuid", "textwrap", "struct",
    "dataclasses", "xml", "bisect", "heapq", "operator", "numbers",
})

#: Builtins that can escape the sandbox; rejected as bare call names. Attribute
#: access (e.g. ``rasterio.open``) is unaffected.
BLOCKED_CALLS = frozenset({
    "open", "__import__", "eval", "exec", "compile", "input", "breakpoint",
})

#: Dotted attribute calls that spawn processes or run commands; rejected in safe
#: mode even though ``os`` is importable. Dangerous mode lifts this.
BLOCKED_ATTR_CALLS = frozenset({
    "os.system", "os.popen", "os.popen2", "os.popen3", "os.popen4",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.execv", "os.execl", "os.execve", "os.execle", "os.execvp",
    "os.execvpe", "os.execlp", "os.execlpe", "os.fork", "os.forkpty",
    "os.startfile",
})


def allowed_imports() -> frozenset[str]:
    """The core stack plus whatever optional packages this machine has."""
    return ALLOWED_IMPORTS | packages.optional_roots()


def dotted_name(node: ast.AST) -> str | None:
    """Reconstruct an attribute chain (e.g. ``os.system``) from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base is not None:
            return f"{base}.{node.attr}"
    return None


def _reject(root: str, *, from_import: bool = False) -> str:
    """Return the message for an import of ``root`` that safe mode refuses."""
    missing = packages.import_error(root)
    if missing:
        return missing
    verb = "import from" if from_import else "import of"
    return f"{verb} {root!r} is not allowed in run_python"


def guard(tree: ast.Module) -> str | None:
    """Return a violation message, or ``None`` if the code is allowed in safe mode.

    This is a cooperative guard, not an OS-level security boundary: it rejects
    the obvious escape vectors (subprocess, network, dynamic execution, raw
    command calls) so the agent stays on the structured, workspace-confined
    path. Dangerous mode bypasses this entirely. A hostile process cannot be
    contained by in-process ``exec``; that requires OS isolation.
    """
    permitted = allowed_imports()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in permitted:
                    return _reject(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.split(".")[0] not in permitted:
                return _reject(node.module, from_import=True)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
                return f"{func.id}() is not allowed in run_python"
            dotted = dotted_name(func)
            if dotted in BLOCKED_ATTR_CALLS:
                return f"{dotted}() is not allowed in run_python (use dangerous mode)"
    return None


__all__ = [
    "ALLOWED_IMPORTS",
    "BLOCKED_ATTR_CALLS",
    "BLOCKED_CALLS",
    "allowed_imports",
    "dotted_name",
    "guard",
]
