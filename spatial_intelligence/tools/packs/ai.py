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

from ...ai import catalog, store
from ...ai.catalog import find as find_model
from ...ai.manager import get_manager
from ...contracts.effects import Effect
from ...contracts.errors import ToolInputError
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

    @tool(effects=frozenset({Effect.READ, Effect.NETWORK}), kind=ToolKind.REPORTING)
    def ai_fetch_model(
        self,
        repo_id: str,
        filenames: list[str] | None = None,
        revision: str = "",
        verify: bool = False,
    ) -> dict:
        """Download any Hugging Face repository into the local cache, verified.

        For weights that are not in the catalog — the ones a sandbox library
        loads, e.g. "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11".
        Nothing else in the app fetches on the sandbox's behalf: run_python has
        no network, so fetch here, then hand the returned path to the library.

        filenames picks the files to pull and defaults to every file in the
        revision; a large repo has many unrelated formats, so name what you
        need. revision defaults to the repository's current commit — pass one
        for a reproducible run. Sizes come from the Hub, and so does the hash
        each file is checked against on the way in — sha256 for an LFS object,
        the git blob id for everything else — while verify=True re-hashes what
        is already on disk.
        """
        revision, files = self._resolve_revision(repo_id, filenames, revision)
        job = self._rt.reporter.job(
            "download",
            f"{repo_id} ({sum(file.size for file in files) / 1e6:.0f} MB)",
            unit="bytes",
            total=float(sum(file.size for file in files)),
        )
        with job:
            result = store.fetch(
                repo_id,
                revision,
                files,
                verify_hashes=verify,
                progress=advance(job),
            )
            job.done(artifact=result["path"])
        result["revision"] = revision
        result["repo"] = repo_id
        return result

    @staticmethod
    def _resolve_revision(
        repo_id: str, filenames: list[str] | None, revision: str
    ) -> tuple[str, list[catalog.ModelFile]]:
        """Return the pinned revision and the catalog files the Hub reports."""
        import httpx

        commit = revision.strip()
        wanted = {name.strip() for name in filenames or [] if name.strip()}
        try:
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                meta = client.get(f"https://huggingface.co/api/models/{repo_id}")
                meta.raise_for_status()
                payload = meta.json()
                commit = commit or str(payload.get("sha", ""))
                tree = client.get(
                    f"https://huggingface.co/api/models/{repo_id}/tree/{commit}",
                    params={"recursive": "true"},
                )
                tree.raise_for_status()
                entries = tree.json()
        except Exception as exc:  # noqa: BLE001 - one message for any lookup failure
            raise ToolInputError(f"could not read {repo_id} from the Hub: {exc}") from exc

        if not commit:
            raise ToolInputError(f"{repo_id} reports no commit to pin")
        files = [
            catalog.ModelFile(
                path=entry["path"],
                sha256=entry.get("lfs", {}).get("oid", ""),
                size=int(entry.get("size", 0)),
                # The tree API reports the git blob id for every file and a
                # sha256 only for LFS objects; a plain file needs the former.
                sha1=entry.get("oid", ""),
            )
            for entry in entries
            if entry.get("type") == "file"
            and (not wanted or entry["path"] in wanted)
        ]
        if not files:
            raise ToolInputError(
                f"no files matched in {repo_id}; it has "
                + ", ".join(entry["path"] for entry in entries if entry.get("type") == "file")[:600]
            )
        missing = wanted - {file.path for file in files}
        if missing:
            raise ToolInputError(
                f"{repo_id} has no file(s) {sorted(missing)}; it has "
                + ", ".join(file.path for file in files)[:600]
            )
        return commit, files

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
