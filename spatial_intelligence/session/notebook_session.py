"""The open notebook: visible cells, recorded provenance, and persistence.

Ported from the cell half of the old ``AppState``. It owns no locks and no
threads: the session facade guards mutations, and the runner reports outcomes
here instead of mutating cells itself.
"""

from __future__ import annotations

from pathlib import Path
from pprint import pformat

from ..agent.interactions import answers_markdown, pending_interaction, validate_answers
from ..agent.runner import RunOutcome
from ..contracts.errors import ToolInputError
from ..workspace import notebook, traces
from ..workspace.store import Workspace

NOTEBOOK_NAME = "notebook.ipynb"

#: Cell kinds a run can execute.
RUNNABLE_KINDS = frozenset({"python", "prompt"})


def stream_output(text: str) -> dict:
    """Build a normalized stream output (stdout-like cell output)."""
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text,
        "ename": None,
        "evalue": None,
        "traceback": None,
    }


def error_output(ename: str, text: str) -> dict:
    """Build a normalized error output for a failed cell."""
    return {
        "output_type": "error",
        "name": None,
        "text": None,
        "ename": ename,
        "evalue": text,
        "traceback": [text],
    }


def is_generated_cell(cell: dict) -> bool:
    """Whether a cell is harness-recorded provenance rather than a user cell."""
    geoai = (cell.get("metadata") or {}).get("geoai") or {}
    return bool(geoai.get("generated"))


def ordered_notebook_cells(cells: list[dict], provenance: list[dict]) -> list[dict]:
    """Place recorded provenance directly after its parent in the saved notebook."""
    by_parent: dict[str, list[dict]] = {}
    orphaned: list[dict] = []
    visible_ids = {cell["id"] for cell in cells}
    for recorded in provenance:
        geoai = (recorded.get("metadata") or {}).get("geoai") or {}
        parent_id = geoai.get("parent_cell_id")
        if parent_id in visible_ids:
            by_parent.setdefault(parent_id, []).append(recorded)
        else:
            orphaned.append(recorded)
    ordered: list[dict] = []
    for cell in cells:
        ordered.append(cell)
        ordered.extend(by_parent.get(cell["id"], []))
    ordered.extend(orphaned)
    return ordered


class NotebookSession:
    """The open notebook's cells and its ``notebook.ipynb`` document."""

    def __init__(self) -> None:
        self._workspace: Workspace | None = None
        self._cells: list[dict] = []
        self._provenance: list[dict] = []

    # -- lifecycle -------------------------------------------------------

    @property
    def workspace(self) -> Workspace | None:
        return self._workspace

    def load(self, workspace: Workspace) -> None:
        """Open ``workspace``'s notebook, splitting user and recorded cells."""
        self._workspace = workspace
        loaded = notebook.read_nb(workspace.root / NOTEBOOK_NAME)
        self._cells = [
            cell
            for cell in loaded
            if not is_generated_cell(cell) and cell.get("kind") != "interaction"
        ]
        self._provenance = [cell for cell in loaded if is_generated_cell(cell)]
        self.rehydrate_traces()

    def close(self) -> None:
        """Forget the open notebook (the document stays on disk)."""
        self._workspace = None
        self._cells = []
        self._provenance = []

    def save(self) -> None:
        """Write the notebook document, interleaving recorded provenance."""
        if self._workspace is None:
            return
        notebook.write_nb(
            self._workspace.root / NOTEBOOK_NAME,
            ordered_notebook_cells(self._cells, self._provenance),
        )

    def rehydrate_traces(self) -> None:
        """Restore per-cell trace steps and token usage from ``traces/*.jsonl``."""
        if self._workspace is None:
            return
        for cell in self._cells:
            if cell.get("kind") != "prompt":
                continue
            loaded = traces.read_trace(
                self._workspace.traces / f"{cell['id']}.jsonl", include_messages=False
            )
            if loaded["steps"]:
                cell["trace"] = loaded["steps"]
            if loaded["usage"] is not None:
                cell["usage"] = loaded["usage"]
            if loaded["run_id"]:
                cell["run_id"] = loaded["run_id"]
            if loaded["conversation_id"]:
                cell["conversation_id"] = loaded["conversation_id"]
            if loaded["status"] == "done" and loaded["output"] is not None:
                cell["status"] = "done"
                cell["outputs"] = [stream_output(str(loaded["output"]))]
            elif loaded["status"] == "error" and loaded["error"] is not None:
                cell["status"] = "error"
                cell["outputs"] = [error_output("AgentError", f"ERROR: {loaded['error']}")]
            elif loaded["status"] == "stopped":
                cell["status"] = "stopped"
                cell["outputs"] = [stream_output("Stopped.")]

    # -- cell access -----------------------------------------------------

    def cells(self) -> list[dict]:
        """The visible cells, in order (a live list)."""
        return self._cells

    def provenance(self) -> list[dict]:
        """The recorded provenance cells, in order (a live list)."""
        return self._provenance

    def find(self, cell_id: str) -> dict:
        """Return one visible cell, raising ``KeyError`` when it is gone."""
        for cell in self._cells:
            if cell["id"] == cell_id:
                return cell
        raise KeyError(cell_id)

    def find_or_none(self, cell_id: str) -> dict | None:
        """Return one visible cell, or ``None`` when it was deleted mid-run."""
        for cell in self._cells:
            if cell["id"] == cell_id:
                return cell
        return None

    def trace_path(self, cell_id: str) -> Path | None:
        """Where the cell's trace file lives, if a workspace is open."""
        if self._workspace is None:
            return None
        return self._workspace.traces / f"{cell_id}.jsonl"

    # -- cell mutations --------------------------------------------------

    def add(
        self,
        kind: str,
        source: str = "",
        index: int | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Create a cell and place it in the notebook."""
        cell = notebook.new_cell(kind, source, metadata=metadata)
        if index is None or index < 0:
            self._cells.append(cell)
        else:
            self._cells.insert(min(index, len(self._cells)), cell)
        self.save()
        return cell

    def update_source(self, cell_id: str, source: str) -> dict:
        """Replace a cell's source text."""
        cell = self.find(cell_id)
        cell["source"] = source
        self.save()
        return cell

    def delete(self, cell_id: str) -> None:
        """Remove a cell, its recorded provenance, and its trace file."""
        cell = self.find(cell_id)
        self._cells.remove(cell)
        self._provenance = [
            recorded
            for recorded in self._provenance
            if ((recorded.get("metadata") or {}).get("geoai") or {}).get("parent_cell_id")
            != cell_id
        ]
        path = self.trace_path(cell_id)
        if path is not None:
            path.unlink(missing_ok=True)
        self.save()

    def move(self, cell_id: str, index: int) -> None:
        """Move a cell to ``index``."""
        cell = self.find(cell_id)
        self._cells.remove(cell)
        self._cells.insert(min(max(index, 0), len(self._cells)), cell)
        self.save()

    # -- run lifecycle ---------------------------------------------------

    def begin_run(self, cell_id: str) -> dict:
        """Reset a cell for a fresh run and drop its stale provenance."""
        cell = self.find(cell_id)
        if cell["kind"] not in RUNNABLE_KINDS:
            raise ToolInputError("markdown cells are not runnable")
        cell["status"] = "running"
        cell["outputs"] = []
        cell["trace"] = []
        cell["usage"] = None
        cell["run_id"] = None
        cell["conversation_id"] = None
        # A fresh run supersedes any interaction left pending by a prior run.
        cell["interaction"] = None
        cell["interaction_history"] = []
        cell.setdefault("metadata", {}).setdefault("geoai", {}).pop("interaction", None)
        self.purge_provenance(cell_id)
        return cell

    def purge_provenance(self, cell_id: str) -> None:
        """Drop recorded cells belonging to ``cell_id``."""
        self._provenance = [
            recorded
            for recorded in self._provenance
            if ((recorded.get("metadata") or {}).get("geoai") or {}).get("parent_cell_id")
            != cell_id
        ]

    def apply_python_result(self, cell_id: str, output: str, *, failed: bool) -> dict:
        """Record the result of a ``python`` cell."""
        cell = self.find(cell_id)
        cell["trace"] = []
        if failed:
            cell["status"] = "error"
            cell["outputs"] = [error_output("PythonError", output)]
        else:
            cell["status"] = "done"
            cell["outputs"] = [stream_output(output)]
        return cell

    def apply_outcome(self, cell_id: str, outcome: RunOutcome) -> dict:
        """Record what a prompt run did with the cell."""
        cell = self.find(cell_id)
        cell["usage"] = outcome.usage
        cell["run_id"] = outcome.run_id
        cell["conversation_id"] = outcome.conversation_id
        if outcome.status == "waiting_for_input":
            cell["status"] = "waiting_for_input"
            cell["interaction"] = outcome.interaction
            cell.setdefault("metadata", {}).setdefault("geoai", {})["interaction"] = (
                outcome.interaction
            )
            cell["outputs"] = []
        elif outcome.status == "stopped":
            cell["status"] = "stopped"
            cell["interaction"] = None
            cell["outputs"] = [stream_output("Stopped.")]
        elif outcome.status == "error":
            cell["status"] = "error"
            cell["outputs"] = [
                error_output("AgentError", f"ERROR: {outcome.error}")
            ]
        else:
            cell["status"] = "done"
            cell["outputs"] = [stream_output(outcome.output or "")]
        return cell

    # -- interactions ----------------------------------------------------

    def pending_interaction(self, cell_id: str) -> dict:
        """Return the interaction a cell is waiting on, raising when stale."""
        cell = self.find(cell_id)
        interaction = pending_interaction(cell)
        if interaction is None:
            raise ToolInputError("interaction is no longer pending")
        return interaction

    def prepare_resume(
        self, cell_id: str, interaction_id: str, answers: dict, *, messages: list
    ) -> tuple[dict, dict]:
        """Validate answers, record them, and return ``(cell, resume_payload)``.

        Raises:
            ToolInputError: When the interaction is stale, the answers do not
                match the form, or the conversation history is unavailable.
        """
        from ..agent.runner import resume_payload

        cell = self.find(cell_id)
        interaction = self.pending_interaction(cell_id)
        if interaction.get("id") != interaction_id:
            raise ToolInputError("interaction is no longer pending")
        validate_answers(interaction, answers)
        if not messages:
            raise ToolInputError(
                "cannot resume because conversation history is unavailable"
            )
        call_id = interaction["tool_call_id"]
        self.record_interaction_response(cell_id, interaction, answers)
        cell["status"] = "running"
        cell["interaction"] = None
        completed = dict(interaction)
        completed["answers"] = answers
        completed["submitted"] = True
        cell.setdefault("interaction_history", []).append(completed)
        cell.setdefault("metadata", {}).setdefault("geoai", {}).pop("interaction", None)
        payload = resume_payload(
            messages=messages,
            interaction=interaction,
            answers=answers,
            conversation_id=cell.get("conversation_id"),
            call_id=call_id,
        )
        self.save()
        return cell, payload

    def cancel_interaction(self, cell_id: str, interaction_id: str) -> dict:
        """Stop a cell that is paused for browser input."""
        cell = self.find(cell_id)
        interaction = self.pending_interaction(cell_id)
        if interaction.get("id") != interaction_id:
            raise ToolInputError("interaction is no longer pending")
        cell["status"] = "stopped"
        cell["interaction"] = None
        cell["outputs"] = [stream_output("Stopped while waiting for input.")]
        cell.setdefault("metadata", {}).setdefault("geoai", {}).pop("interaction", None)
        self.save()
        return cell

    def record_interaction_response(
        self, parent_id: str, interaction: dict, answers: dict
    ) -> None:
        """Complete the deferred tool record and save the user's choices."""
        call_id = interaction.get("tool_call_id")
        for recorded in reversed(self._provenance):
            geoai = (recorded.get("metadata") or {}).get("geoai") or {}
            if geoai.get("tool_call_id") == call_id:
                self.record_tool_result(recorded, {"content": {"user_response": answers}})
                break
        cell = notebook.new_cell(
            "markdown",
            answers_markdown(interaction, answers),
            metadata={
                "geoai": {
                    "kind": "interaction_response",
                    "role": "user",
                    "parent_cell_id": parent_id,
                    "interaction_id": interaction.get("id"),
                    "generated": True,
                }
            },
        )
        cell["status"] = "done"
        self._append_provenance(cell)

    # -- provenance recording --------------------------------------------

    def find_recorded_tool(self, tool_call_id: str) -> dict | None:
        """Find a provenance cell for a tool call already recorded before a resume."""
        for cell in reversed(self._provenance):
            geoai = (cell.get("metadata") or {}).get("geoai") or {}
            if geoai.get("tool_call_id") == tool_call_id:
                return cell
        return None

    def record_tool_call(self, parent_id: str, step: dict) -> dict:
        """Create a read-only code cell for a structured agent tool call."""
        name = step.get("name") or "unknown_tool"
        args = step.get("args")
        if (
            name == "run_python"
            and isinstance(args, dict)
            and isinstance(args.get("code"), str)
        ):
            source = args["code"]
        else:
            rendered = pformat(args if isinstance(args, dict) else {}, sort_dicts=False)
            source = (
                f"# Spatial Intelligence structured tool call: {name}\n"
                f"# Replayed by the harness with the active workspace context.\n"
                f"{name}(**{rendered})"
            )
        cell = notebook.new_cell(
            "tool",
            source,
            metadata={
                "geoai": {
                    "kind": "tool",
                    "role": "assistant",
                    "parent_cell_id": parent_id,
                    "tool_name": name,
                    "tool_call_id": step.get("tool_call_id"),
                    "args": args,
                    "generated": True,
                }
            },
        )
        cell["status"] = "running"
        self._append_provenance(cell)
        return cell

    def record_tool_result(self, cell: dict, step: dict) -> None:
        """Attach a tool result to its previously recorded code cell."""
        import json

        content = step.get("content")
        text = content if isinstance(content, str) else json.dumps(content, indent=2, ensure_ascii=False)
        cell["outputs"] = [stream_output(text)]
        cell["execution_count"] = 1
        cell["status"] = "done"
        self.save()

    def record_agent_response(self, parent_id: str, output: str) -> None:
        """Append the agent's final response as a standard Markdown cell."""
        cell = notebook.new_cell(
            "markdown",
            output,
            metadata={
                "geoai": {
                    "kind": "response",
                    "role": "assistant",
                    "parent_cell_id": parent_id,
                    "generated": True,
                }
            },
        )
        cell["status"] = "done"
        self._append_provenance(cell)

    def mark_tool_waiting(self, tool_call_id: str) -> None:
        """Mark the recorded tool cell that is paused for user input."""
        recorded = self.find_recorded_tool(tool_call_id)
        if recorded is not None:
            recorded["status"] = "waiting_for_input"
            self.save()

    def _append_provenance(self, cell: dict) -> None:
        """Persist a recorded cell without adding it to the interactive list."""
        self._provenance.append(cell)
        self.save()
