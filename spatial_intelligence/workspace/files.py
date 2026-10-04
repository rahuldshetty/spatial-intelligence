"""Workspace file I/O: listing, reading, writing, importing, and downloading.

Services are plain functions over a :class:`~spatial_intelligence.workspace.Workspace`
(plus, for downloads, a progress :class:`~spatial_intelligence.contracts.progress.Job`).
They never read the ambient runtime, so they are directly testable and reusable
by the server outside an agent run.
"""

from __future__ import annotations

import shutil
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from ..contracts.errors import ToolInputError, WorkspaceError
from ..contracts.progress import Job, Reporter
from .store import Workspace

#: Hard cap on one download.
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
#: Hard cap on one batch download call.
MAX_BATCH_DOWNLOADS = 20
#: Worker threads used by :func:`download_many`.
MAX_DOWNLOAD_WORKERS = 6
#: Report progress at most every this many bytes...
PROGRESS_INTERVAL_BYTES = 4 * 1024 * 1024
#: ...or this many seconds, whichever comes first.
PROGRESS_INTERVAL_SECONDS = 0.25
#: Stream chunk size.
CHUNK_BYTES = 1024 * 1024
#: Default read cap for :func:`read_text`, and the ceiling for a larger window.
MAX_READ_BYTES = 1_000_000
#: Chars per token, matching the compaction harness's heuristic.
CHARS_PER_TOKEN = 4
#: Share of the window one read may spend. Compaction keeps the newest results,
#: so a few big reads must fit under the half-window target by themselves.
READ_WINDOW_SHARE = 0.125
#: Floor, so a small window still allows a useful read.
MIN_READ_BYTES = 16_000

_USER_AGENT = "spatial-intelligence/0.1"
_HTTPS_SCHEMES = ("http://", "https://")


def filename_from_url(url: str) -> str:
    """Derive a download filename from a URL path, with a safe fallback."""
    return Path(urlparse(url).path).name or "download"


def list_files(workspace: Workspace, subdir: str = "", pattern: str = "*") -> list[str]:
    """List files under ``subdir`` as sorted workspace-relative paths."""
    base = workspace.root if not subdir else workspace.resolve(subdir)
    return sorted(
        path.relative_to(workspace.root).as_posix()
        for path in base.glob(f"**/{pattern}")
        if path.is_file()
    )


def find_files(workspace: Workspace, pattern: str) -> list[str]:
    """Find files matching ``pattern`` anywhere under the workspace root."""
    return sorted(
        path.relative_to(workspace.root).as_posix()
        for path in workspace.root.rglob(pattern)
        if path.is_file()
    )


def read_limit_bytes(context_window: int) -> int:
    """Bytes one read may return for a model with this context window.

    Compaction cannot undo a result already in the history, so this cap is what
    keeps one read from spending the window. ``0`` means the window is unknown.
    """
    if context_window <= 0:
        return MAX_READ_BYTES
    scaled = int(context_window * READ_WINDOW_SHARE * CHARS_PER_TOKEN)
    return max(MIN_READ_BYTES, min(MAX_READ_BYTES, scaled))


def read_text(
    workspace: Workspace,
    path: str,
    *,
    max_bytes: int = MAX_READ_BYTES,
    offset: int = 0,
    limit: int | None = None,
) -> str:
    """Read a UTF-8 text file, optionally a byte slice of it.

    Reads the whole file by default, refusing anything larger than ``max_bytes``.
    Large files (a Sentinel-1 annotation XML, for instance) are read as a slice:
    ``offset`` is the 0-based byte to start at and ``limit`` the maximum number
    of bytes to return.
    """
    resolved = workspace.resolve(path, must_exist=True)
    if limit is None:
        size = resolved.stat().st_size
        if size > max_bytes:
            raise ToolInputError(
                f"file too large ({size} bytes > {max_bytes}): {path!r}; "
                "read a slice with offset/limit"
            )
        return resolved.read_text(encoding="utf-8", errors="replace")
    with resolved.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(min(limit, max_bytes))
    return data.decode("utf-8", errors="replace")


def write_text(workspace: Workspace, path: str, content: str) -> Path:
    """Write UTF-8 text under ``results/``, ``maps/``, or ``data/``.

    Written byte-for-byte (no newline translation), so a read-back returns
    exactly what the tool was given — scripts and data files the agent writes
    must reload identically on any platform.
    """
    out = workspace.resolve(path, write=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return out


def import_path(workspace: Workspace, source: str, dest_name: str | None = None) -> Path:
    """Copy an external file or folder into ``data/``.

    ``source`` is a path outside the workspace; ``dest_name`` names the target
    under ``data/`` (defaults to the source basename). Name collisions get a
    numeric suffix.
    """
    resolved_source = Path(source).expanduser()
    if not resolved_source.exists():
        raise WorkspaceError(f"import source does not exist: {source!r}")
    target = Path(dest_name) if dest_name else Path(resolved_source.name)
    dest = workspace.resolve_under(workspace.data, target)
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        index = 1
        while dest.exists():
            dest = workspace.resolve_under(workspace.data, f"{stem}_{index}{suffix}")
            index += 1
    if resolved_source.is_dir():
        shutil.copytree(resolved_source, dest)
    else:
        shutil.copy2(resolved_source, dest)
    return dest


def unique_names(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Give every pair a distinct destination name, so siblings cannot collide."""
    counts: dict[str, int] = {}
    result: list[tuple[str, str]] = []
    for url, filename in pairs:
        name = Path(filename).name or filename_from_url(url)
        stem, suffix = Path(name).stem, Path(name).suffix
        count = counts.get(name, 0)
        counts[name] = count + 1
        if count:
            name = f"{stem}_{count}{suffix}"
        result.append((url, name))
    return result


def prepare_downloads(requests: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Validate and de-duplicate a batch download request."""
    if not requests:
        raise ToolInputError("files must contain at least one download")
    if len(requests) > MAX_BATCH_DOWNLOADS:
        raise ToolInputError(
            f"a maximum of {MAX_BATCH_DOWNLOADS} downloads can be started at once"
        )
    prepared: list[tuple[str, str]] = []
    for item in requests:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            raise ToolInputError("each download must provide a url")
        url = item["url"].strip()
        if not url.startswith(_HTTPS_SCHEMES):
            raise ToolInputError("download URLs must use http:// or https://")
        prepared.append((url, item.get("filename") or filename_from_url(url)))
    return unique_names(prepared)


def download_file(
    workspace: Workspace,
    url: str,
    filename: str,
    *,
    job: Job | None = None,
    timeout: float = 60.0,
    opener=None,
) -> Path:
    """Stream ``url`` into ``data/`` and return the written path.

    Writes to a unique ``.part`` sibling and renames on success, so a partial
    download never appears as a usable file. ``job`` receives byte progress.
    ``opener`` defaults to :func:`urllib.request.urlopen`, resolved at call time
    so tests (and future transports) can substitute it.
    """
    if not url.startswith(_HTTPS_SCHEMES):
        raise ToolInputError("download URLs must use http:// or https://")
    if opener is None:
        opener = urllib.request.urlopen
    name = Path(filename).name or filename_from_url(url)
    out = workspace.resolve_under(workspace.data, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f"{out.name}.{uuid.uuid4().hex}.part")

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    downloaded = 0
    total: int | None = None
    try:
        with opener(request, timeout=timeout) as response, partial.open("wb") as handle:
            total = _content_length(response)
            if total is not None and total > MAX_DOWNLOAD_BYTES:
                raise ToolInputError(f"download exceeds the 2 GB cap: {url!r}")
            if job is not None:
                job.progress(0, total=total)
            last_reported = 0
            last_report_time = time.monotonic()
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_BYTES:
                    raise ToolInputError(f"download exceeds the 2 GB cap: {url!r}")
                handle.write(chunk)
                now = time.monotonic()
                if job is not None and (
                    downloaded - last_reported >= PROGRESS_INTERVAL_BYTES
                    or now - last_report_time >= PROGRESS_INTERVAL_SECONDS
                ):
                    job.progress(downloaded, total=total)
                    last_reported = downloaded
                    last_report_time = now
        partial.replace(out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return out


def download_many(
    workspace: Workspace, requests: list[dict[str, str]], *, reporter: Reporter
) -> list[dict[str, str]]:
    """Download several URLs concurrently, one progress job each.

    The returned list preserves input order and carries one ``status``/``path``
    or ``status``/``error`` result per request, so a failed sibling never hides
    the outcome of a successful one.
    """
    prepared = prepare_downloads(requests)
    results: list[dict[str, str] | None] = [None] * len(prepared)

    def run(url: str, filename: str) -> dict[str, str]:
        job = reporter.job("download", filename, unit="bytes")
        try:
            with job:
                path = download_file(workspace, url, filename, job=job)
                relative = workspace.relative(path)
                job.done(artifact=relative)
            return {"url": url, "filename": filename, "status": "done", "path": relative}
        except Exception as exc:  # keep sibling downloads reporting their own state
            return {
                "url": url,
                "filename": filename,
                "status": "error",
                "error": str(exc),
            }

    workers = min(MAX_DOWNLOAD_WORKERS, len(prepared))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run, url, filename): index
            for index, (url, filename) in enumerate(prepared)
        }
        for future, index in futures.items():
            results[index] = future.result()
    return [result for result in results if result is not None]


def _content_length(response) -> int | None:
    raw = response.headers.get("Content-Length")
    try:
        return int(raw) if raw else None
    except (TypeError, ValueError):
        return None
