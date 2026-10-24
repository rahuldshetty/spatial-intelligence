"""Drone imagery detection and import-to-map routes.

The browser calls these two routes at two points: after a local/URL import, and
from the file tree's context menu. Detection is read-only; the import route is
what writes the photo GeoJSON and adds layers to the live map.
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
