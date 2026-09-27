"""The session facade: notebook, workspace, run worker, jobs, and settings.

Composed of the collaborators built in this package rather than doing their
work: the notebook owns cells, the run queue owns scheduling, the job registry
owns progress, the services object owns the runtime and agent. This class wires
them together, guards mutation with one short-lived lock, and drains the run
queue on a single worker thread.
"""

from __future__ import annotations

import threading
import traceback

from ..agent.model import validate_model
from ..agent.runner import PromptRunner, RunnerHooks
from ..contracts.errors import ToolInputError
from ..map.bridge import reset_bridge
from ..settings import prefs
from ..settings.env import DEFAULT_MODEL, list_workspaces
from ..tools.runtime import ToolRuntime, bind
from .bus import EventBus
from .jobs import JobRegistry
from .notebook_session import RUNNABLE_KINDS, NotebookSession
from .runs import RunQueue
from .services import SessionServices
from .workspace_session import WorkspaceSession

#: Settings a PUT may patch.
PATCHABLE_SETTINGS = frozenset(
    {"model", "theme", "dangerous_mode", "max_retries", "record_agent_steps"}
)


class AppState:
    """Single-owner state for the running server."""

    def __init__(self, *, settings: dict | None = None, worker: bool = True) -> None:
        self.lock = threading.RLock()
        self.settings = settings or prefs.load_settings()
        self.bus = EventBus()
        self.jobs = JobRegistry(self.bus)
        self.runs = RunQueue()
        self.notebook = NotebookSession()
        self.workspaces = WorkspaceSession()
        self.services = SessionServices(
            bus=self.bus,
            jobs=self.jobs,
            settings=lambda: self.settings,
            files_provider=lambda: self.list_files(),
            map_provider=lambda: self.workspaces.project(),
            trace_length=self._trace_length,
        )
        self._worker: threading.Thread | None = None
        if worker:
            self._start_worker()

    # -- snapshot --------------------------------------------------------

    def snapshot(self) -> dict:
        """Everything the browser needs to render the app from scratch."""
        with self.lock:
            return {
                "active_workspace": self.workspaces.name,
                "workspaces": list_workspaces(),
                "cells": self.notebook.cells(),
                "map_project": self.workspaces.project(),
                "map_app_url": self.workspaces.map._app_url,
                "files": self.list_files(),
                "jobs": self.jobs.snapshot(),
                "settings": dict(self.settings),
            }

    def subscribe(self):
        """Register an SSE subscriber."""
        return self.bus.subscribe()

    def unsubscribe(self, subscriber) -> None:
        """Deregister an SSE subscriber."""
        self.bus.unsubscribe(subscriber)

    # -- settings --------------------------------------------------------

    def update_settings(self, patch: dict) -> dict:
        """Merge ``patch`` into settings, persist, and rebuild the agent on change.

        A model change requires a rebuilt agent; theme, dangerous mode, retry
        cap, and recording do not. A failed rebuild rolls the model back, so an
        invalid model string cannot leave the session unusable.
        """
        with self.lock:
            previous_model = self.settings.get("model")
            model_changed = False
            if patch.get("model"):
                candidate = str(patch["model"]).strip()
                if candidate and candidate != previous_model:
                    self.settings["model"] = candidate
                    model_changed = True
            if patch.get("theme") in ("light", "dark"):
                self.settings["theme"] = patch["theme"]
            if patch.get("dangerous_mode") is not None:
                self.settings["dangerous_mode"] = bool(patch["dangerous_mode"])
            if patch.get("max_retries") is not None:
                try:
                    self.settings["max_retries"] = max(1, int(patch["max_retries"]))
                except (TypeError, ValueError):
                    self.settings["max_retries"] = prefs.load_settings()["max_retries"]
            if patch.get("record_agent_steps") is not None:
                self.settings["record_agent_steps"] = bool(patch["record_agent_steps"])
            try:
                if model_changed:
                    validate_model(self.settings["model"])
                self.settings = prefs.save_settings(self.settings)
                if model_changed and self.services.bound:
                    self.services.rebuild_agent()
            except Exception as exc:  # roll back and surface
                self.settings["model"] = previous_model
                self.settings = prefs.save_settings(self.settings)
                raise ToolInputError(f"could not apply settings: {exc}") from exc
            settings = dict(self.settings)
        self.bus.publish("settings", {"settings": settings})
        return settings

    # -- workspace lifecycle ---------------------------------------------

    def open_workspace(self, name: str) -> dict:
        """Open a workspace, binding a fresh runtime and agent to it."""
        self.runs.wait_idle()
        # The map iframe is replaced when the workspace name changes, so the
        # recorded handshake belongs to an iframe that is gone until the new one
        # posts its own. Re-opening the workspace that is already open leaves the
        # iframe in place, and clearing the record there would report the live
        # app as disconnected for the rest of the session.
        if name != self.workspaces.name:
            reset_bridge()
        with self.lock:
            workspace = self.workspaces.open(name)
            self.notebook.load(workspace)
            self.jobs.clear()
            runtime = self.services.bind(workspace, self.workspaces.map)
            from ..map.layers import repoint_local_rasters

            repoint_local_rasters(self.workspaces.map, workspace, runtime.file_url)
            return self.snapshot()

    def new_workspace(self, name: str) -> dict:
        """Create a workspace directory and open it."""
        with self.lock:
            self.workspaces.new(name)
        return self.open_workspace(name)

    def close_workspace(self) -> dict:
        """Close the open workspace and reset the map and notebook."""
        self.runs.wait_idle()
        reset_bridge()
        with self.lock:
            self.workspaces.close()
            self.notebook.close()
            self.jobs.clear()
            self.services.unbind()
            return self.snapshot()

    def save_workspace(self) -> dict:
        """Persist the notebook and the live map."""
        with self.lock:
            self.notebook.save()
            self.workspaces.save_map()
            return {"ok": True}

    def set_map_project(self, project: dict) -> dict:
        """Adopt a project pushed by the browser."""
        with self.lock:
            self.workspaces.set_map_project(project)
            return {"ok": True}

    # -- cells -----------------------------------------------------------

    def add_cell(self, kind: str, source: str = "", index: int | None = None) -> dict:
        """Add a cell and return the new snapshot."""
        with self.lock:
            self.notebook.add(kind, source, index)
            return self.snapshot()

    def update_cell(self, cell_id: str, source: str) -> dict:
        """Replace a cell's source and return the new snapshot."""
        with self.lock:
            self.notebook.update_source(cell_id, source)
            return self.snapshot()

    def delete_cell(self, cell_id: str) -> dict:
        """Delete a cell, its provenance, its traces, and its progress jobs."""
        with self.lock:
            self.notebook.delete(cell_id)
            self.jobs.clear_parent(cell_id)
            return self.snapshot()

    def move_cell(self, cell_id: str, index: int) -> dict:
        """Reorder a cell and return the new snapshot."""
        with self.lock:
            self.notebook.move(cell_id, index)
            return self.snapshot()

    # -- running ---------------------------------------------------------

    def run_cell(self, cell_id: str) -> None:
        """Reset a cell and queue it for execution."""
        with self.lock:
            # The whole reset cell goes out, not a patch: the browser repaints
            # this cell from the event, so the previous run's output and trace
            # cannot survive into the new one.
            cell = self.notebook.begin_run(cell_id)
            self.jobs.clear_parent(cell_id)
            self.bus.publish("cell", cell)
        self.runs.submit(cell_id)

    def run_all(self) -> None:
        """Queue every runnable cell, in order."""
        with self.lock:
            runnable = [
                cell["id"] for cell in self.notebook.cells() if cell["kind"] in RUNNABLE_KINDS
            ]
        for cell_id in runnable:
            self.run_cell(cell_id)

    def stop_cell(self, cell_id: str) -> bool:
        """Cancel a running or queued cell."""
        stopped = self.runs.stop(cell_id)
        if stopped:
            self.jobs.clear_parent(cell_id)
        return stopped

    # -- interactions ----------------------------------------------------

    def respond_interaction(self, cell_id: str, interaction_id: str, answers: dict) -> None:
        """Resume a paused prompt with the user's answers."""
        with self.lock:
            trace_path = self.notebook.trace_path(cell_id)
            from ..workspace import traces

            messages = traces.read_messages(trace_path) if trace_path is not None else []
            cell, payload = self.notebook.prepare_resume(
                cell_id, interaction_id, answers, messages=messages
            )
            self.jobs.clear_parent(cell_id)
            self.bus.publish("cell", cell)
        self.runs.submit(cell_id, resume=payload)

    def cancel_interaction(self, cell_id: str, interaction_id: str) -> None:
        """Stop a prompt that is paused for input."""
        with self.lock:
            cell = self.notebook.cancel_interaction(cell_id, interaction_id)
            self.bus.publish("cell", cell)

    # -- imports ---------------------------------------------------------

    def import_local(self, uploaded: list[tuple[str, bytes]]) -> dict:
        """Copy uploaded files into the workspace ``data/`` folder."""
        with self.lock:
            workspace = self._require_workspace()
            imported: list[str] = []
            for filename, data in uploaded:
                if not filename:
                    raise ToolInputError("empty filename")
                dest = workspace.resolve_under(workspace.data, filename)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                imported.append(workspace.relative(dest))
            self.bus.publish("files", {"files": self.list_files()})
            return {"imported": imported}

    def import_url(self, url: str, filename: str | None = None) -> dict:
        """Download a URL into the workspace ``data/`` folder."""
        from ..workspace.files import download_file, filename_from_url

        with self.lock:
            workspace = self._require_workspace()
            name = filename or filename_from_url(url)
            job = self.jobs.reporter().job("download", name, unit="bytes")
            with job:
                path = download_file(workspace, url, name, job=job)
                relative = workspace.relative(path)
                job.done(artifact=relative)
            self.bus.publish("files", {"files": self.list_files()})
            return {"path": relative}

    def list_files(self) -> list[str]:
        """List workspace files."""
        with self.lock:
            return self.workspaces.list_files()

    # -- run worker ------------------------------------------------------

    def _start_worker(self) -> None:
        self._worker = threading.Thread(
            target=self._worker_loop, name="spatial-intelligence-run-worker", daemon=True
        )
        self._worker.start()

    def stop_worker(self, timeout: float | None = None) -> bool:
        """Stop the run worker and wait for it to leave, if one is running.

        The server does not need this while it serves, but every other owner
        does: a stopped worker is what makes it safe to delete or replace the
        workspace tree, and it stops a process from accumulating one idle thread
        per opened session. Idempotent, so a caller may stop an already-stopped
        state.

        ``timeout`` bounds the wait for the queue to drain and for the thread to
        leave, and returning ``False`` says it was still busy. The server passes
        a bound because it calls this from the lifespan shutdown, which uvicorn
        awaits without one: an unbounded wait would make the first Ctrl+C hang
        until a queued batch finished. The sentinel is queued either way, so a
        worker that outlasts the bound still leaves once its cell ends.
        """
        worker = self._worker
        if worker is None:
            return True
        self.runs.wait_idle(timeout)
        self.runs.shutdown()
        worker.join(timeout)
        if worker.is_alive():
            return False
        self._worker = None
        return True

    def _worker_loop(self) -> None:
        while True:
            cell_id = self.runs.take()
            if cell_id is None:  # the queue was shut down
                return
            try:
                token = self.runs.start(cell_id)
                if token is None:
                    continue  # cancelled before it started
                resume = self.runs.resume_payload(cell_id)
                self._execute(cell_id, token, resume)
            except Exception:  # a run must never kill the worker
                traceback.print_exc()
            finally:
                self.runs.finish(cell_id)
                self.runs.task_done()

    def _execute(self, cell_id: str, token, resume: dict | None) -> None:
        with self.lock:
            cell = self.notebook.find_or_none(cell_id)
            if cell is None:
                return
            kind = cell["kind"]
            source = cell["source"]
            # Every execution attempt counts, including a resumed one.
            cell["execution_count"] = (cell.get("execution_count") or 0) + 1
        try:
            with self.services.run_scope(cell_id) as runtime:
                if kind == "python":
                    self._execute_python(cell_id, source, runtime)
                else:
                    self._execute_prompt(cell_id, source, token, resume, runtime)
        finally:
            with self.lock:
                cell = self.notebook.find_or_none(cell_id)
                if cell is not None:
                    self.notebook.save()
                    self.bus.publish("cell", cell)
            self.services.flush()
            self.bus.publish("map", {"project": self.workspaces.project()})
            self.bus.publish("files", {"files": self.list_files()})

    def _execute_python(self, cell_id: str, source: str, runtime: ToolRuntime) -> None:
        # The cell shows the FULL output (the tool's return value is only the
        # bounded preview the model sees); `inspect_output` pages it later.
        executor = self.services.python_executor(runtime)
        failed = False
        try:
            with bind(runtime):
                executor.run(source)
            output = executor.output.text
        except Exception as exc:  # surface a crash in the cell output
            output = f"ERROR: {exc}"
            failed = True
        with self.lock:
            cell = self.notebook.find_or_none(cell_id)
            if cell is not None:
                self.notebook.apply_python_result(cell_id, output, failed=failed)

    def _execute_prompt(
        self,
        cell_id: str,
        source: str,
        token,
        resume: dict | None,
        runtime: ToolRuntime,
    ) -> None:
        with self.lock:
            cell = self.notebook.find_or_none(cell_id)
            if cell is None:
                return
            # The runner streams into the cell's own list. The snapshot serves
            # these very cell dicts, so a browser that reloads mid-run rebuilds
            # the whole trace instead of only the steps published after it
            # reconnected.
            trace_steps = list(cell.get("trace", [])) if resume else []
            cell["trace"] = trace_steps
        runner = PromptRunner(self._runner_hooks())
        with bind(runtime):
            outcome = runner.run(cell_id, source, trace_steps, token=token, resume=resume)
        with self.lock:
            cell = self.notebook.find_or_none(cell_id)
            if cell is None:
                return
            cell["trace"] = trace_steps
            self.notebook.apply_outcome(cell_id, outcome)

    def _runner_hooks(self) -> RunnerHooks:
        return RunnerHooks(
            registry=self.services.registry,
            agent=lambda: self.services.agent,
            plan_store=lambda: self.services.plan_store,
            model=lambda: self.settings.get("model", DEFAULT_MODEL),
            settings=lambda: self.settings,
            augment_prompt=self._augment_prompt,
            trace_path=self.notebook.trace_path,
            publish_trace=lambda cell_id, step: self.bus.publish(
                "trace", {"id": cell_id, "step": step}
            ),
            find_recorded_tool=self.notebook.find_recorded_tool,
            record_tool_call=self.notebook.record_tool_call,
            record_tool_result=self.notebook.record_tool_result,
            record_agent_response=self.notebook.record_agent_response,
            mark_tool_waiting=self.notebook.mark_tool_waiting,
        )

    # -- helpers ---------------------------------------------------------

    def _trace_length(self, cell_id: str) -> int:
        """How many steps ``cell_id`` has published (where its next job starts)."""
        cell = self.notebook.find_or_none(cell_id)
        return len(cell.get("trace") or []) if cell is not None else 0

    def _augment_prompt(self, source: str) -> str:
        """Prepend the ``data/`` listing to the current user turn."""
        workspace = self.workspaces.workspace
        if workspace is None:
            return source
        files = workspace.list_files("data")
        if not files:
            return source
        listing = "\n".join(f"- {name}" for name in files)
        context = (
            "Files currently available in the workspace data/ folder "
            "(imported inputs the user may refer to):\n" + listing
        )
        return context + "\n\n" + source

    def _require_workspace(self):
        workspace = self.workspaces.workspace
        if workspace is None:
            raise ToolInputError("no workspace open")
        return workspace
