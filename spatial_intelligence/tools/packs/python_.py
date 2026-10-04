"""Python tools: the ``run_python`` escape hatch, its output paging, and API help.

The pack owns one executor for the whole session and hands it to the rest of
the app through ``ToolRuntime.service``: the agent's ``run_python`` call and the
server's notebook-adjacent path run against the same namespace and read the same
output store.
"""

from __future__ import annotations

from ...contracts.effects import Effect
from ...pythonruntime import help as api_help
from ...pythonruntime.executor import PythonExecutor
from ..runtime import ToolRuntime
from ..spec import pack, tool


@pack(category="python", effects=frozenset({Effect.READ}))
class PythonPack:
    """The sandbox: run snippets, page their output, and look up the API."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self.executor = runtime.service(
            "python.executor",
            lambda: PythonExecutor(runtime.workspace, approved=runtime.approved),
        )

    @tool(core=True, effects=frozenset({Effect.WORKSPACE_WRITE, Effect.PROCESS, Effect.NETWORK}))
    def run_python(self, code: str, timeout_seconds: int = 0) -> str:
        """Run a snippet of Python for math/processing and capture its output.

        In safe mode the sandbox exposes the geospatial stack (``numpy``, ``pandas``,
        ``geopandas``, ``rasterio``, ``rioxarray``, ``xarray``, ``shapely``, ``pyproj``),
        basic stdlib (``os``, ``sys``, ``pathlib``, ``shutil``, ...), a
        workspace-confined ``ws`` helper, and a read-only ``models`` view of the
        model cache; subprocess, network, dynamic execution, and raw command calls
        are rejected. Heavy packages installed as the ``geoai`` extra (torchgeo,
        terratorch, torch) are importable too — skill() documents their APIs.
        With dangerous mode enabled every guard is lifted.

        ``timeout_seconds`` raises the wall-clock cap for this call when a model
        needs minutes per tile (0 keeps the default). The snippet runs on a
        daemon thread that cannot be killed, so a call that times out keeps
        running in the background: size the work to finish inside the cap.

        The returned text is a bounded preview (first few lines). The full output is
        stored and can be paged with ``inspect_output`` or filtered with
        ``query_output`` when the result is JSON/XML.
        """
        # The session's approval is the single source of truth: the settings
        # toggle may have flipped since the shared executor was created.
        self.executor.approved = self._rt.approved
        return self.executor.run(code, timeout=timeout_seconds or None)

    @tool()
    def inspect_output(self, start: int = 0, count: int = 30) -> str:
        """Read a slice of the last ``run_python`` output by line index.

        ``start`` is 0-based and ``count`` is the maximum number of lines to return
        (capped at 500). Use this to page through a large output that ``run_python``
        truncated. Returns a header with the line range and total, then the lines.
        """
        return self.executor.output.inspect(start, count)

    @tool()
    def query_output(self, query: str = "") -> str:
        """Filter the last ``run_python`` result with a jq-like path or XPath-lite.

        JSON results support ``.`` (whole), ``.key.subkey``, ``[index]``,
        ``[start:end]``, and the bare filters ``keys`` and ``length``. XML results
        support tag paths relative to the root (``child``, ``child/grandchild``),
        descendant search (``.//tag``), a positional ``[n]``, plus ``/@attr`` and
        ``/text()``. With an empty query, returns a schema summary of the last
        structured result.
        """
        return self.executor.output.query(query)

    @tool()
    def python_help(self, name: str = "") -> dict:
        """Inspect a sandbox symbol's kind, signature, docstring, and members.

        Pass a dotted name reachable from the run_python namespace, e.g.
        ``"rasterio.warp.reproject"``, ``"rasterio.control.GroundControlPoint"``,
        ``"rasterio.warp.Resampling"``, ``"numpy.ndarray"``, or ``"ws"``. With no
        argument it lists the available top-level roots. Use this instead of
        writing ``dir()``/``__doc__``/``inspect.signature`` probes in run_python.
        """
        return api_help.describe(name)


__all__ = ["PythonPack"]
