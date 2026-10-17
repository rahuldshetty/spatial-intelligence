"""Map capture and JSON-query callbacks from the browser's bridge."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..deps import app_state

router = APIRouter(tags=["map"])


class CaptureResult(BaseModel):
    token: str
    image: str | None = None
    view: dict | None = None
    error: str | None = None


class BridgeResult(BaseModel):
    token: str
    result: Any | None = None
    error: str | None = None


@router.post("/api/map/capture/{capture_id}")
def api_map_capture(capture_id: str, body: CaptureResult) -> dict:
    """Resolve a pending capture request posted by the live map iframe."""
    accepted = app_state().complete_capture(
        capture_id,
        body.token,
        image=body.image,
        view=body.view,
        error=body.error,
    )
    return {"accepted": accepted}


@router.post("/api/map/bridge/{request_id}")
def api_map_bridge(request_id: str, body: BridgeResult) -> dict:
    """Resolve a pending JSON map query posted by the live map iframe."""
    accepted = app_state().complete_map_bridge(
        request_id,
        body.token,
        result=body.result,
        error=body.error,
    )
    return {"accepted": accepted}


__all__ = ["router"]
