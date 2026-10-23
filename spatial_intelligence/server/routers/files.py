"""Workspace file serving and imports.

The ``/api/files`` route is what the embedded map fetches: the GeoLibre iframe
is served from its own loopback origin, so a workspace raster is a cross-origin
read. This server owns the file, so it answers with a permissive origin while
the path stays confined to the active workspace, and ``FileResponse`` streams
with HTTP Range support for the in-browser GeoTIFF reader.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...contracts.errors import WorkspaceError
from ...workspace import files as file_ops
from ..deps import app_state

router = APIRouter(tags=["files"])

FILE_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Range",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
}


class ImportUrl(BaseModel):
    url: str
    filename: str | None = None


class CheckUrl(BaseModel):
    url: str


class DeleteFile(BaseModel):
    path: str


class RenameFile(BaseModel):
    path: str
    name: str


@router.post("/api/import/check")
def api_check_url(body: CheckUrl) -> dict:
    """Check a URL from the server before a workspace is created."""
    return file_ops.check_url(body.url)


@router.api_route("/api/files/{rel_path:path}", methods=["GET", "HEAD"])
def api_workspace_file(rel_path: str) -> FileResponse:
    """Serve one workspace file to the map iframe."""
    workspace = app_state().workspaces.workspace
    if workspace is None:
        raise HTTPException(status_code=409, detail="no workspace open")
    try:
        path = workspace.resolve(rel_path, must_exist=True)
    except WorkspaceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not a file")
    return FileResponse(path, headers=FILE_CORS)


@router.api_route("/api/files/{rel_path:path}", methods=["OPTIONS"])
def api_workspace_file_preflight(rel_path: str) -> Response:
    """Answer the CORS preflight for a workspace file."""
    return Response(status_code=200, headers=FILE_CORS)


@router.post("/api/files/delete")
def api_delete_file(body: DeleteFile) -> dict:
    """Delete one user file from the open workspace."""
    return app_state().delete_file(body.path)


@router.post("/api/files/rename")
def api_rename_file(body: RenameFile) -> dict:
    """Rename one user file within the open workspace."""
    return app_state().rename_file(body.path, body.name)


@router.post("/api/import/local")
async def api_import_local(files: list[UploadFile] = File(...)) -> dict:
    """Copy uploaded files (or a folder's files) into ``data/``."""
    if app_state().workspaces.workspace is None:
        raise HTTPException(status_code=409, detail="no workspace open")
    uploaded: list[tuple[str, bytes]] = []
    for upload in files:
        uploaded.append((upload.filename or "", await upload.read()))
    return app_state().import_local(uploaded)


@router.post("/api/import/url")
def api_import_url(body: ImportUrl) -> dict:
    """Download a URL into ``data/``."""
    if app_state().workspaces.workspace is None:
        raise HTTPException(status_code=409, detail="no workspace open")
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="url must be http(s)")
    return app_state().import_url(body.url, body.filename)


@router.post("/api/map/project")
def api_set_map_project(body: dict) -> dict:
    """Adopt a map project pushed by the browser and persist it."""
    if app_state().workspaces.workspace is None:
        raise HTTPException(status_code=409, detail="no workspace open")
    project = body.get("project")
    if not isinstance(project, dict):
        raise HTTPException(status_code=400, detail="project must be an object")
    revision = body.get("revision")
    if revision is not None:
        try:
            revision = int(revision)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="revision must be an integer") from exc
    workspace = body.get("workspace")
    if workspace is not None and not isinstance(workspace, str):
        raise HTTPException(status_code=400, detail="workspace must be a string")
    return app_state().set_map_project(project, revision=revision, workspace=workspace)
