"""Tool Configuration: catalog, scopes, and persistence."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..deps import app_state

router = APIRouter(tags=["tools"])


class UpdateToolConfig(BaseModel):
    """A configuration to save at one scope."""

    scope: Literal["global", "workspace"]
    config: dict = Field(default_factory=dict)


@router.get("/api/tools/config")
def api_get_tool_config() -> dict:
    """Return both scopes, the effective config, and the tool/python catalog."""
    return app_state().tool_config_payload()


@router.put("/api/tools/config")
def api_update_tool_config(body: UpdateToolConfig) -> dict:
    """Persist the selected scope and rebuild the open session's agent."""
    return app_state().update_tool_config(body.scope, body.config)
