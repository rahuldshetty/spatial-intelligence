"""Local model lifecycle: what is available, what is downloaded, what is loaded.

These tools are about the models themselves — the catalog, the download, the
memory they hold. The tasks that *use* a model live in their own packs
(:mod:`~spatial_intelligence.tools.packs.segmentation`,
:mod:`~spatial_intelligence.tools.packs.detection`), because segmentation and
detection are different problems with different arguments, different outputs,
and no reason to be edited in the same file.
"""

from __future__ import annotations

from typing import Any

from ...ai import store
from ...ai.catalog import find as find_model
from ...ai.manager import get_manager
from ...contracts.effects import Effect
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool


@pack(category="ai")
class AIPack:
    """The model cache and the memory it holds, for this workspace."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._manager = get_manager(runtime)

    @tool(core=True, effects=frozenset({Effect.READ}))
    def ai_models(self, task: str = "") -> list[dict]:
        """List the local AI models, what is downloaded, and what is in memory.

        Each entry has id, task ("segmentation" or "detection"), prompts it
        accepts, license, bytes, downloaded, loaded, provider, and the seconds it
        has been idle. Filter with task when the request names one. Call this
        before segment_image or detect_objects: it says whether a download is
        needed and what is already costing memory.
        """
        return self._manager.status(task=task.strip() or None)

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}), kind=ToolKind.REPORTING)
    def ai_pull_model(self, model_id: str, verify: bool = False) -> dict:
        """Download a model into the local cache, once, and verify it.

        Files land under ``<GEOAI_HOME>/.models`` and are checked against the
        sha256 recorded in the catalog, so a rerun skips what is already there;
        set verify=True to re-hash the files on disk. Sizes: slimsam-77 is about
        40 MB, yolos-tiny about 26 MB. The task tools download on their own when
        a model is missing, so this is only needed to fetch ahead of time or to
        repair a bad file.
        """
        spec = find_model(model_id)
        job = self._rt.reporter.job(
            "download", f"{spec.id} ({spec.total_bytes / 1e6:.0f} MB)", unit="bytes",
            total=float(spec.total_bytes),
        )
        with job:
            result = self._manager.pull(spec, verify_hashes=verify, progress=advance(job))
            job.done(artifact=str(store.spec_dir(spec)))
        result["models"] = self._manager.status()
        return result

    @tool(effects=frozenset({Effect.READ}))
    def ai_unload_model(self, model_id: str = "") -> dict:
        """Free the memory a loaded model holds; an empty model_id unloads all.

        Loaded models stay resident until they are idle past the TTL, so call
        this after a heavy run on a machine that needs the memory back. A model
        still in use is refused rather than pulled out from under a run.
        """
        return self._manager.unload(model_id)


def advance(job: Any):
    """Return a progress callback that reports downloaded bytes into ``job``.

    Shared with the task packs: they open a model-download job whenever the
    manager has to fetch a model on their behalf.
    """

    def report(done: float, total: float, label: str) -> None:
        job.progress(done, total=total, detail=label)

    return report


__all__ = ["AIPack", "advance"]
