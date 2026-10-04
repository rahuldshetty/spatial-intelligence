"""The per-run service bundle that tools reach through ``current_runtime()``.

Replaces the package-wide mutable context holder: a runtime is bound either
around an agent run (the runner's job) or around an out-of-band call (the
server's ``import_url``, a notebook-adjacent helper), so nothing global has to
be reset between sessions or tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..contracts.errors import RuntimeNotBoundError, ToolInputError
from ..contracts.progress import NULL_REPORTER, Reporter
from ..workspace import Workspace
from ..workspace.files import MAX_READ_BYTES


def _notify(hook: Callable[[], None] | None) -> None:
    """Call a UI notification hook, swallowing a broken subscriber."""
    if hook is None:
        return
    try:
        hook()
    except Exception:
        # Notifications carry no correctness contract: a dead SSE subscriber or
        # a busy session lock must never fail the tool that reported progress.
        pass


@dataclass(slots=True)
class RuntimeEvents:
    """Hooks a tool raises when it changed something the UI displays."""

    files_changed: Callable[[], None] | None = None
    map_changed: Callable[[], None] | None = None

    def notify_files(self) -> None:
        """Tell the session the workspace file list may have changed."""
        _notify(self.files_changed)

    def notify_map(self) -> None:
        """Tell the session the live map project may have changed."""
        _notify(self.map_changed)


@dataclass(slots=True)
class ToolRuntime:
    """Everything a tool is allowed to reach.

    Attributes:
        workspace: The active workspace (path confinement lives here).
        map: The shared live map, or ``None`` when no map exists.
        reporter: Progress factory already bound to the owning run.
        events: UI notification hooks.
        base_url: Origin under which the server serves workspace files.
        run_id: The run this runtime belongs to, if any.
        approved: Whether the user granted approval (dangerous mode).
    """

    workspace: Workspace
    map: Any | None = None
    reporter: Reporter = NULL_REPORTER
    events: RuntimeEvents = field(default_factory=RuntimeEvents)
    base_url: str = "http://127.0.0.1:8000/"
    run_id: str | None = None
    approved: bool = False
    services: dict[str, Any] = field(default_factory=dict)
    #: Bytes one read may hand the model; derived from the context window.
    max_read_bytes: int = MAX_READ_BYTES

    def service(self, key: str, factory: Callable[[], Any]) -> Any:
        """Return this session's instance of ``key``, creating it on first use.

        A few tools own state that outlives one call but must not outlive the
        session (the ``run_python`` output store is the only current user).
        Keeping it here means the agent path and the server's out-of-band calls
        share one instance without a module-level global.
        """
        if key not in self.services:
            self.services[key] = factory()
        return self.services[key]

    def file_url(self, rel: str) -> str:
        """Return the server URL under which workspace file ``rel`` is served.

        The map iframe is served from a different origin, so the server's
        ``/api/files/`` route answers this URL with permissive CORS headers.
        """
        normalized = rel.replace("\\", "/").lstrip("/")
        return f"{self.base_url.rstrip('/')}/api/files/{normalized}"

    def require_map(self) -> Any:
        """Return the live map, raising when the session has none."""
        if self.map is None:
            raise RuntimeNotBoundError("the live map is not available in this session")
        return self.map

    def record_artifact(self, path: Path) -> str:
        """Record a written output, notify the UI, and return its absolute path.

        The single implementation of what used to be three near-identical
        ``_record`` helpers in the tool modules.
        """
        self.workspace.record_output(self.workspace.relative(path))
        self.events.notify_files()
        return str(path)

    def require_approval(self, tool: str, *, hint: str = "") -> None:
        """Raise a model-visible failure unless the user granted approval."""
        if self.approved:
            return
        message = (
            f"{tool} requires approval: enable dangerous mode in File -> Settings "
            "and run again."
        )
        if hint:
            message = f"{message} {hint}"
        raise ToolInputError(message)

    def evolve(self, **overrides: Any) -> "ToolRuntime":
        """Return a copy with fields replaced (a new run, a new reporter)."""
        return replace(self, **overrides)


_runtime: ContextVar[ToolRuntime | None] = ContextVar(
    "spatial_intelligence_runtime", default=None
)


@contextmanager
def bind(runtime: ToolRuntime) -> Iterator[ToolRuntime]:
    """Bind ``runtime`` for the duration of the block (nestable)."""
    token = _runtime.set(runtime)
    try:
        yield runtime
    finally:
        _runtime.reset(token)


def current_runtime() -> ToolRuntime:
    """Return the bound runtime, raising when a tool runs outside a session."""
    runtime = _runtime.get()
    if runtime is None:
        raise RuntimeNotBoundError(
            "no ToolRuntime is bound; wrap the call in spatial_intelligence.bind(...) "
            "or invoke the tool through the agent"
        )
    return runtime


def maybe_runtime() -> ToolRuntime | None:
    """Return the bound runtime or ``None`` (for optional-notification paths)."""
    return _runtime.get()
