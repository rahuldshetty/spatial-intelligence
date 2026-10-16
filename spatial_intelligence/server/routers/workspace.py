"""Workspace lifecycle, cells, and running them."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...contracts.errors import ToolInputError
from ..deps import app_state

router = APIRouter(tags=["workspace"])

#: Cell kinds a user may add from the UI.
VALID_KINDS = frozenset({"markdown", "python", "prompt"})


class WorkspaceName(BaseModel):
    name: str


class NewWorkspace(BaseModel):
    """A workspace to create, optionally seeded from a workflow's answers."""

    name: str
    workflow: str = ""
    params: dict = Field(default_factory=dict)


class AddCell(BaseModel):
    kind: str
    source: str = ""
    index: int | None = None


class UpdateCell(BaseModel):
    source: str


class MoveCell(BaseModel):
    index: int


class InteractionResponse(BaseModel):
    interaction_id: str
    answers: dict


@router.post("/api/workspace/new")
def api_workspace_new(body: NewWorkspace) -> dict:
    """Create a workspace and open it, seeding a workflow when one is named."""
    return app_state().new_workspace(
        body.name.strip(), body.workflow.strip(), body.params
    )


@router.post("/api/workspace/open")
def api_workspace_open(body: WorkspaceName) -> dict:
    """Open an existing workspace."""
    return app_state().open_workspace(body.name)


@router.post("/api/workspace/save")
def api_workspace_save() -> dict:
    """Persist the notebook and the live map."""
    return app_state().save_workspace()


@router.post("/api/workspace/close")
def api_workspace_close() -> dict:
    """Close the open workspace."""
    return app_state().close_workspace()


@router.post("/api/cells")
def api_add_cell(body: AddCell) -> dict:
    """Append a cell to the notebook."""
    if body.kind not in VALID_KINDS:
        raise ToolInputError(f"invalid cell kind: {body.kind!r}")
    return app_state().add_cell(body.kind, body.source, body.index)


@router.put("/api/cells/{cell_id}")
def api_update_cell(cell_id: str, body: UpdateCell) -> dict:
    """Replace a cell's source."""
    return app_state().update_cell(cell_id, body.source)


@router.delete("/api/cells/{cell_id}")
def api_delete_cell(cell_id: str) -> dict:
    """Delete a cell."""
    return app_state().delete_cell(cell_id)


@router.post("/api/cells/{cell_id}/move")
def api_move_cell(cell_id: str, body: MoveCell) -> dict:
    """Reorder a cell."""
    return app_state().move_cell(cell_id, body.index)


@router.post("/api/cells/{cell_id}/run")
def api_run_cell(cell_id: str) -> dict:
    """Queue one cell."""
    app_state().run_cell(cell_id)
    return {"accepted": True}


@router.post("/api/cells/{cell_id}/stop")
def api_stop_cell(cell_id: str) -> dict:
    """Cancel a running or queued cell."""
    return {"stopped": app_state().stop_cell(cell_id)}


@router.post("/api/run-all")
def api_run_all() -> dict:
    """Queue every runnable cell."""
    app_state().run_all()
    return {"accepted": True}


@router.post("/api/cells/{cell_id}/interaction")
def api_respond_interaction(cell_id: str, body: InteractionResponse) -> dict:
    """Resume a prompt paused for structured input."""
    app_state().respond_interaction(cell_id, body.interaction_id, body.answers)
    return {"accepted": True}


@router.delete("/api/cells/{cell_id}/interaction/{interaction_id}")
def api_cancel_interaction(cell_id: str, interaction_id: str) -> dict:
    """Stop a prompt paused for structured input."""
    app_state().cancel_interaction(cell_id, interaction_id)
    return {"cancelled": True}
