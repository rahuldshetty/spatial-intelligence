"""Per-session services: the tool runtime, the registry, and the agent.

One instance per server process. Binding builds the runtime, instantiates every
tool pack against it, and constructs the agent; unbinding tears all of it down
on a workspace switch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from pydantic_ai_harness.compaction import DEFAULT_CONTEXT_WINDOW

from ..agent.builder import BuiltAgent, build_agent
from ..agent.vision import VisionService
from ..ai.manager import MANAGER_SERVICE
from ..contracts.errors import RuntimeNotBoundError
from ..contracts.services import (
    CAPTURE_SERVICE,
    CONVERSATION_SERVICE,
    MAP_BRIDGE_SERVICE,
    VISION_SERVICE,
)
from ..pythonruntime.policy import PythonPolicy
from ..settings.env import server_base_url
from ..tools.build import default_registry
from ..tools.config import apply_tool_config, normalize_tool_config
from ..tools.packs.vision import VisionPack
from ..tools.registry import ToolRegistry
from ..tools.runtime import RuntimeEvents, ToolRuntime
from ..workspace import Workspace
from ..workspace.files import read_limit_bytes
from .bus import EventBus
from .capture import CaptureBroker
from .jobs import JobRegistry
from .map_bridge import MapBridgeBroker

#: Service-bag keys shared with the packs that provide or consume them.
REGISTRY_SERVICE = "tools.registry"
PYTHON_EXECUTOR_SERVICE = "python.executor"


class MapNotifier:
    """Rate-limits map-change notifications so a run cannot flood the browser.

    Tools persist the map on every mutation; projecting a large project and
    pushing it over SSE for each one is wasteful, so notifications are throttled
    and the session flushes the trailing one when the run ends.
    """

    def __init__(self, emit: Callable[[], None], min_interval: float = 0.2) -> None:
        self._emit = emit
        self._min_interval = min_interval
        self._last = 0.0
        self._pending = False

    def notify(self) -> None:
        """Emit now if the interval has elapsed, otherwise remember to flush."""
        now = time.monotonic()
        if now - self._last >= self._min_interval:
            self._last = now
            self._pending = False
            self._emit()
        else:
            self._pending = True

    def flush(self) -> None:
        """Emit a suppressed notification, if any."""
        if self._pending:
            self._last = time.monotonic()
            self._pending = False
            self._emit()

    def reset(self) -> None:
        """Drop a notification queued for the workspace being left."""
        self._pending = False
        self._last = 0.0


class SessionServices:
    """Builds and owns the runtime, registry, and agent for one open session."""

    def __init__(
        self,
        *,
        bus: EventBus,
        jobs: JobRegistry,
        settings: Callable[[], dict],
        files_provider: Callable[[], list[str]],
        map_state: Callable[[], dict],
        tool_config: Callable[[], dict] | None = None,
        bump_map: Callable[[], int] | None = None,
        trace_length: Callable[[str], int] | None = None,
        base_url: str | None = None,
        vision: VisionService | None = None,
        capture: CaptureBroker | None = None,
        map_bridge: MapBridgeBroker | None = None,
        conversation: Any | None = None,
        recall: Callable[[str | None, dict], None] | None = None,
    ) -> None:
        self._bus = bus
        self._jobs = jobs
        self._settings = settings
        self._files_provider = files_provider
        self._map_state = map_state
        self._tool_config = tool_config or (lambda: {})
        self._bump_map = bump_map or (lambda: 0)
        self._trace_length = trace_length
        self._base_url = base_url or server_base_url()
        self._vision = vision
        self._capture = capture
        self._map_bridge = map_bridge
        self._conversation = conversation
        self._recall = recall
        self._registry: ToolRegistry | None = None
        self._runtime: ToolRuntime | None = None
        self._built: BuiltAgent | None = None
        self.map_notifier = MapNotifier(self._publish_map)

    # -- accessors -------------------------------------------------------

    @property
    def bound(self) -> bool:
        """Whether a workspace is currently bound to services."""
        return self._runtime is not None

    @property
    def runtime(self) -> ToolRuntime | None:
        """The session runtime, or ``None`` before the first bind."""
        return self._runtime

    @property
    def registry(self) -> ToolRegistry:
        """The session's tool registry."""
        if self._registry is None:
            raise RuntimeNotBoundError("no workspace is open")
        return self._registry

    @property
    def agent(self) -> Any:
        """The session's agent."""
        if self._built is None:
            raise RuntimeNotBoundError("no agent is built; open a workspace first")
        return self._built.agent

    @property
    def plan_store(self) -> Any:
        """The session agent's plan store, or ``None`` when unbound."""
        return self._built.plan_store if self._built is not None else None

    # -- lifecycle -------------------------------------------------------

    def bind(self, workspace: Workspace, map_obj: Any) -> ToolRuntime:
        """Create the runtime, register every pack against it, and build the agent."""
        if self._runtime is not None:
            # A workspace switch rebinds; drop the old models, registry, and
            # agent first so nothing from the previous workspace survives.
            self.unbind()
        config = normalize_tool_config(self._tool_config())
        runtime = ToolRuntime(
            workspace=workspace,
            map=map_obj,
            reporter=self._jobs.reporter(),
            events=RuntimeEvents(
                files_changed=self._publish_files,
                map_changed=self._map_changed,
                recall=self._recall,
            ),
            base_url=self._base_url,
            approved=self._approval_granted(),
            # Compaction cannot undo a result already in the history, so the
            # read cap is derived from the same window it targets.
            max_read_bytes=read_limit_bytes(
                int(self._settings().get("context_window", 0) or 0) or DEFAULT_CONTEXT_WINDOW
            ),
            tool_config=config,
            python_policy=PythonPolicy.from_config(config),
        )
        if self._vision is not None:
            runtime.services[VISION_SERVICE] = self._vision
        if self._capture is not None:
            runtime.services[CAPTURE_SERVICE] = self._capture
        if self._map_bridge is not None:
            runtime.services[MAP_BRIDGE_SERVICE] = self._map_bridge
        if self._conversation is not None:
            runtime.services[CONVERSATION_SERVICE] = self._conversation
        registry = default_registry(runtime)
        apply_tool_config(registry, config)
        # The capability pack reads the registry back out of the service bag, so
        # discovery can annotate the capabilities it advertises with each tool's
        # registry summary without a second source of truth.
        runtime.services[REGISTRY_SERVICE] = registry
        self._registry = registry
        self._runtime = runtime
        self.rebuild_agent()
        # A probe may have completed between the pack check above and this
        # assignment; reconcile once the registry is visible to the callback.
        self.sync_vision()
        return runtime

    def unbind(self) -> None:
        """Forget the runtime, registry, and agent, and free loaded models."""
        runtime = self._runtime
        if runtime is not None:
            manager = runtime.services.get(MANAGER_SERVICE)
            if manager is not None:
                # A loaded model is tens of megabytes of session; dropping the
                # runtime without closing it would keep that alive until the
                # process ends, which is exactly the leak the manager exists to
                # prevent.
                manager.close()
        self._registry = None
        self._runtime = None
        self._built = None

    def rebind(self) -> None:
        """Rebuild the registry, runtime, and agent for the open workspace.

        Tool Configuration changed: a filtered registry and a runtime carrying
        the new run_python policy are both needed. Reusing the workspace and
        map avoids disturbing the open notebook, the map iframe, or the files.
        """
        runtime = self._runtime
        if runtime is not None:
            self.bind(runtime.workspace, runtime.map)

    def rebuild_agent(self) -> None:
        """(Re)build the agent from the current settings and registry."""
        if self._registry is None or self._runtime is None:
            return
        config = normalize_tool_config(self._tool_config())
        absent = [name for name in config["disabled_tools"] if name not in self._registry]
        self._built = build_agent(
            self._registry,
            lambda: self._settings().get("model", ""),
            context_window=int(self._settings().get("context_window", 0) or 0),
            python_policy=self._runtime.python_policy,
            disabled_tools=absent,
        )

    def sync_vision(self) -> None:
        """Add or remove the image tool after a probe, then rebuild the agent.

        The registry is authoritative for discovery too, so adding the pack here
        is the only step needed to make ``analyze_image`` visible and callable.
        Tool Configuration can disable the tool independently of the model, so
        it wins over the probe.
        """
        registry = self._registry
        if registry is None or self._runtime is None:
            return
        config = normalize_tool_config(self._tool_config())
        available = bool(self._vision is not None and self._vision.supported)
        if "analyze_image" in config["disabled_tools"]:
            available = False
        registered = "analyze_image" in registry
        if available == registered:
            return
        if available:
            registry.add_pack(VisionPack, self._runtime)
        else:
            registry.remove("analyze_image")
        self.rebuild_agent()

    @contextmanager
    def run_scope(self, cell_id: str) -> Iterator[ToolRuntime]:
        """Point the session runtime at one run for the duration of the block.

        Packs hold a single runtime reference, so run-scoped fields (the owning
        run, its reporter, and the approval the run started with) are set on
        that shared runtime. The run worker is the only runner, which is what
        makes this safe — and it is what keeps every progress job a tool opens
        attached to the cell that opened it.

        The reporter also carries the run's anchor: the number of steps the cell
        has published, read as each job opens, which is where the browser draws
        that job's card.
        """
        runtime = self._runtime
        if runtime is None:
            raise RuntimeNotBoundError("no workspace is open")
        runtime.run_id = cell_id
        runtime.reporter = self._jobs.reporter(
            parent_id=cell_id, anchor=self._anchor_for(cell_id)
        )
        runtime.approved = self._approval_granted()
        try:
            yield runtime
        finally:
            runtime.run_id = None
            runtime.reporter = self._jobs.reporter()
            runtime.approved = self._approval_granted()

    def _anchor_for(self, cell_id: str) -> Callable[[], int | None]:
        """Return a callable reading the cell's published step count."""
        trace_length = self._trace_length
        if trace_length is None:
            return lambda: None
        return lambda: trace_length(cell_id)

    def python_executor(self, runtime: ToolRuntime) -> Any:
        """Return the session's ``run_python`` executor (shared with the pack).

        The executor is per session so its output store survives across cells
        (``inspect_output`` pages the previous run), but approval is a per-run
        question, so it is refreshed on every access.
        """
        from ..pythonruntime.executor import PythonExecutor

        executor = runtime.service(
            PYTHON_EXECUTOR_SERVICE,
            lambda: PythonExecutor(
                runtime.workspace,
                approved=runtime.approved,
                services=runtime.services,
            ),
        )
        executor.approved = runtime.approved
        return executor

    def flush(self) -> None:
        """Emit any notification the notifier suppressed."""
        self.map_notifier.flush()

    # -- internals -------------------------------------------------------

    def _approval_granted(self) -> bool:
        return bool(self._settings().get("dangerous_mode", False))

    def _publish_files(self) -> None:
        self._bus.publish("files", {"files": self._files_provider()})

    def _map_changed(self) -> None:
        """Record an authoritative map change before the throttled push."""
        self._bump_map()
        self.map_notifier.notify()

    def _publish_map(self) -> None:
        self._bus.publish("map", self._map_state())
