"""The ``run_python`` executor: parse, guard, run, and capture.

The executor owns the sandbox itself: it parses a snippet, refuses it in safe
mode when :func:`~spatial_intelligence.pythonruntime.sandbox.guard` finds an
escape vector, runs it from the workspace root against a namespace holding the
geospatial stack and the path-confined ``ws`` facade, and captures
stdout/stderr plus the last expression's ``repr``. A raised exception comes
back as a traceback string, not as a raised error: the model reads it and
adapts in the same run.

The snippet's working directory is the workspace root, so a relative path means
the same thing inside a snippet as it does in every other tool: ``data/x`` is
``<workspace>/data/x``. The geospatial stack resolves paths in C, where no
Python-level shim can reach, so binding the process cwd is the only mechanism
that makes ``gpd.read_file("data/x.geojson")`` agree with ``read_file``.

``approved`` is the dangerous-mode flag (the user granted approval), replacing
the module-global toggle the previous package kept. One executor lives per
session; the pack holds it through
:meth:`~spatial_intelligence.tools.runtime.ToolRuntime.service`, so the server
can run a notebook cell against the same namespace and read the full output.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import threading
import traceback

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rioxarray
import rio_cogeo
import scipy
import shapely
import skimage
import xarray

from ..workspace import Workspace
from .output import OutputStore
from .sandbox import guard

#: Wall-clock cap on one snippet. The snippet runs on a daemon thread, so an
#: abandoned snippet cannot keep the process alive.
TIMEOUT = 300.0

#: Serializes snippets while they hold the process working directory at their
#: workspace root. The process has one cwd, so two snippets from different
#: workspaces running at once would resolve each other's relative paths.
_CWD_LOCK = threading.Lock()


class _ConfinedWorkspace:
    """Path-confined facade over the active workspace (no raw ``Path``/map)."""

    def __init__(self, ws: Workspace):
        self._ws = ws

    def resolve(self, rel, *, must_exist: bool = False, write: bool = False):
        return self._ws.resolve(rel, must_exist=must_exist, write=write)

    def read_text(self, rel: str, encoding: str = "utf-8") -> str:
        return self._ws.resolve(rel, must_exist=True).read_text(
            encoding=encoding, errors="replace"
        )

    def write_text(self, rel: str, content: str, encoding: str = "utf-8") -> str:
        out = self._ws.resolve(rel, write=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding=encoding)
        rel_out = out.relative_to(self._ws.root).as_posix()
        self._ws.record_output(rel_out)
        return str(out)

    def list_files(self, subdir: str = "", pattern: str = "*") -> list[str]:
        return self._ws.list_files(subdir, pattern)

    @property
    def data(self) -> str:
        return str(self._ws.data)

    @property
    def results(self) -> str:
        return str(self._ws.results)

    @property
    def maps(self) -> str:
        return str(self._ws.maps)


class PythonExecutor:
    """Runs ``run_python`` snippets against one workspace.

    Attributes:
        workspace: The workspace ``ws`` is confined to.
        approved: Whether dangerous mode is on (every guard lifted).
        output: The store holding the most recent execution's full output.
    """

    def __init__(self, workspace: Workspace, *, approved: bool = False) -> None:
        self.workspace = workspace
        self.approved = bool(approved)
        self.output = OutputStore()

    def namespace(self) -> dict:
        """Return the globals a snippet runs against (fresh per execution)."""
        return {
            "ws": _ConfinedWorkspace(self.workspace),
            "rasterio": rasterio,
            "rioxarray": rioxarray,
            "gpd": gpd,
            "np": np,
            "pd": pd,
            "xr": xarray,
            "shapely": shapely,
            "pyproj": pyproj,
            "skimage": skimage,
            "scipy": scipy,
            "rio_cogeo": rio_cogeo,
            "json": json,
        }

    def run(self, code: str) -> str:
        """Execute ``code`` and return the truncated preview of its output.

        The full output is kept in :attr:`output` for ``inspect_output`` /
        ``query_output``. A syntax error, a guard violation, a timeout, and a
        raised exception all come back as output text rather than raising.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return self.output.store(f"SyntaxError: {exc}")

        if not self.approved:
            violation = guard(tree)
            if violation is not None:
                return self.output.store(
                    f"run_python blocked: {violation}. Use the structured tools, or "
                    "enable dangerous mode."
                )

        ns = self.namespace()
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        box: dict = {}

        def _target() -> None:
            try:
                body = tree.body
                last_expr = None
                if body and isinstance(body[-1], ast.Expr):
                    last_expr = ast.Expression(body.pop().value)
                with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                    if body:
                        module = ast.fix_missing_locations(
                            ast.Module(body=body, type_ignores=[])
                        )
                        exec(compile(module, "<run_python>", "exec"), ns)  # noqa: S102
                    if last_expr is not None:
                        expr = ast.fix_missing_locations(last_expr)
                        box["value"] = eval(compile(expr, "<run_python>", "eval"), ns)  # noqa: S307
                        box["has"] = True
            except Exception:
                err_buf.write(traceback.format_exc())

        thread = threading.Thread(target=_target, daemon=True)
        previous_cwd = os.getcwd()
        with _CWD_LOCK:
            os.chdir(self.workspace.root)
            try:
                thread.start()
                thread.join(TIMEOUT)
            finally:
                # A timed-out snippet keeps running on an abandoned daemon thread
                # that may still resolve relative paths, so the cwd stays at the
                # workspace root until it finishes by itself. Every later snippet
                # chdirs to its own root first, so a stale cwd cannot leak into a
                # different workspace.
                if not thread.is_alive():
                    os.chdir(previous_cwd)
        if thread.is_alive():
            return self.output.store(f"run_python timed out after {int(TIMEOUT)}s")

        stdout = out_buf.getvalue()
        parts: list[str] = []
        if stdout:
            parts.append(stdout.rstrip())
        if err_buf.getvalue():
            parts.append(err_buf.getvalue().rstrip())
        has_value = box.get("has", False)
        if has_value:
            parts.append("=> " + repr(box["value"]))
        return self.output.store(
            "\n".join(parts),
            value=box.get("value"),
            has_value=has_value,
            stdout=stdout,
        )


__all__ = ["PythonExecutor"]
