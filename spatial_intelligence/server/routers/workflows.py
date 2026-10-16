"""The workflow catalog and the model lookups its wizard needs.

Everything the browser asks for before a workspace exists: what workflows there
are, what a place name's bounding box is, what checkpoints a keyword finds on the
Hub, and whether a chosen checkpoint can do the workflow's task at all. The agent
reaches the same ground through its ``geocode`` tool and the prompts it is handed.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ... import workflows
from ...ai import hub, profiles
from ...contracts.errors import ToolInputError
from ...geo import places
from ..deps import app_state

router = APIRouter(tags=["workflows"])


class ModelQuery(BaseModel):
    """A checkpoint to describe, and the task it is being considered for."""

    model: str
    task: str = ""


@router.get("/api/workflows")
def api_workflows() -> dict:
    """List the workflows the wizard may create, with their fields."""
    return {
        "workflows": [
            {
                "id": spec.id,
                "label": spec.label,
                "summary": spec.summary,
                "task": next(
                    (field.task for field in spec.params if field.type == "model"), None
                ),
                "params": [param.model_dump(mode="json") for param in spec.params],
            }
            for spec in workflows.WORKFLOWS
        ]
    }


@router.get("/api/geocode")
def api_geocode(q: str = Query(min_length=1)) -> dict:
    """Resolve a place name to ``{label, bbox, center}`` for an AOI field."""
    return places.geocode(q)


@router.get("/api/models/search")
def api_model_search(
    q: str = Query(min_length=1),
    task: str = "",
    limit: int = 8,
) -> dict:
    """Search the Hub for checkpoints, with what is known about each of them."""
    results = hub.search_models(q, limit)
    for row in results:
        profile = profiles.guess_from_search(row, task)
        answer = profile.as_dict()
        answer["supported"] = profile.supports(task)
        row["profile"] = answer
    return {"results": results}


@router.post("/api/models/profile")
def api_model_profile(body: ModelQuery) -> dict:
    """Decide whether one checkpoint can do this workflow's task, up front."""
    if not body.model.strip():
        raise ToolInputError("choose a model first")
    return app_state().model_profile(body.model.strip(), body.task.strip())
