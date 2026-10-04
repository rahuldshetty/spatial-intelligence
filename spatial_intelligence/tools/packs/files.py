"""File tools: list, find, read, write, import, and download workspace data."""

from __future__ import annotations

from pathlib import Path

from ...contracts.effects import Effect
from ...contracts.errors import ToolInputError
from ...workspace import files as fileops
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool


@pack(category="files", effects=frozenset({Effect.READ}))
class FilesPack:
    """Workspace file tools; every path is confined to the active workspace."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime

    @tool(core=True)
    def list_files(self, subdir: str = "", pattern: str = "*") -> list[str]:
        """Recursively list files under a subdir as sorted relative paths."""
        return fileops.list_files(self._rt.workspace, subdir, pattern)

    @tool(core=True)
    def find_files(self, pattern: str) -> list[str]:
        """Recursively find files matching a glob under the workspace root."""
        return fileops.find_files(self._rt.workspace, pattern)

    @tool(
        core=True,
        summary="Read a text file, or a byte slice of a large one",
        effects=frozenset({Effect.READ}),
    )
    def read_file(
        self,
        path: str,
        max_bytes: int | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Read a UTF-8 text file from the workspace (errors replaced).

        ``max_bytes`` defaults to this deployment's read cap (a slice of the
        context window) and may only lower it. For a large file, read a byte
        range instead: ``offset`` is the 0-based byte, ``limit`` the max bytes.
        """
        cap = self._rt.max_read_bytes
        if max_bytes is not None and max_bytes > cap:
            raise ToolInputError(
                f"max_bytes {max_bytes:,} is above this deployment's read cap of "
                f"{cap:,} bytes; read a slice with offset/limit instead"
            )
        return fileops.read_text(
            self._rt.workspace,
            path,
            max_bytes=cap if max_bytes is None else max_bytes,
            offset=offset,
            limit=limit,
        )

    @tool(core=True, effects=frozenset({Effect.WORKSPACE_WRITE}))
    def write_file(self, path: str, content: str) -> str:
        """Write UTF-8 text to a workspace file; returns the absolute path."""
        out = fileops.write_text(self._rt.workspace, path, content)
        return self._rt.record_artifact(out)

    @tool(
        summary="Stream one URL into data/",
        effects=frozenset({Effect.WORKSPACE_WRITE, Effect.NETWORK}),
        kind=ToolKind.REPORTING,
    )
    def download(self, url: str, filename: str) -> str:
        """Stream a URL into ``data/`` and return its workspace-relative path.

        The UI shows one progress cell for this download. Tell the user when the
        download starts and where the file lives once it completes.
        """
        label = Path(filename).name or fileops.filename_from_url(url)
        job = self._rt.reporter.job("download", label, unit="bytes")
        with job:
            path = fileops.download_file(self._rt.workspace, url, filename, job=job)
            relative = self._rt.workspace.relative(path)
            job.done(artifact=relative)
        self._rt.record_artifact(path)
        return relative

    @tool(
        summary="Download several URLs into data/ concurrently",
        effects=frozenset({Effect.WORKSPACE_WRITE, Effect.NETWORK}),
        kind=ToolKind.REPORTING,
    )
    def download_files(self, files: list[dict[str, str]]) -> list[dict[str, str]]:
        """Download several URLs into ``data/`` concurrently.

        Each item must contain ``url`` and may contain ``filename``. Every file
        has its own progress event/cell, so a failed or completed download does
        not obscure the state of its siblings. The returned list preserves input
        order and contains one ``status``/``path`` or ``status``/``error`` result
        per job.
        """
        results = fileops.download_many(
            self._rt.workspace, files, reporter=self._rt.reporter
        )
        self._rt.events.notify_files()
        return results


__all__ = ["FilesPack"]
