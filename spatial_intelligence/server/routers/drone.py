"""Drone imagery detection, import, and cancellation routes.

Detection is read-only; the import route writes the photo GeoJSON and adds
layers to the live map. Upload, URL import, the workflow wizard, and the file
tree all offer the same flow.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..deps import app_state

router = APIRouter(tags=["drone"])


class DronePaths(BaseModel):
    """Workspace files or folders to inspect/import."""

    paths: list[str] = Field(min_length=1)


class DroneImport(DronePaths):
    """A drone import, optionally building an orthomosaic first."""

    orthorectify: bool = False


class DroneCancel(BaseModel):
    """The orthorectification job to stop."""

    job_id: str = Field(min_length=1)


@router.post("/api/drone/detect")
def api_detect_drone(body: DronePaths) -> dict:
    """List the drone photos and georeferenced rasters in a selection."""
    if app_state().workspaces.workspace is None:
        raise HTTPException(status_code=409, detail="no workspace open")
    return app_state().detect_drone_files(body.paths)


@router.post("/api/drone/import")
def api_import_drone(body: DroneImport) -> dict:
    """Add detected drone files to the live GeoLibre map."""
    if app_state().workspaces.workspace is None:
        raise HTTPException(status_code=409, detail="no workspace open")
    return app_state().import_drone_files(body.paths, body.orthorectify)


@router.post("/api/drone/cancel")
def api_cancel_drone(body: DroneCancel) -> dict:
    """Ask a running orthorectification job to stop between frames."""
    return app_state().cancel_drone_import(body.job_id)
