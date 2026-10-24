"""The live map document: create it, snapshot it, reload it.

One constant names the workspace snapshot the live map is persisted to, so the
three places that used to spell ``"current.geolibre.json"`` independently (the
view, the map tools, and the server state) now agree by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geolibre import Map
from geolibre.project import build_empty_project

from ..workspace import Workspace
from . import locked

#: File name of the workspace snapshot the live map is persisted to.
SNAPSHOT_NAME = "current.geolibre.json"

__all__ = [
    "SNAPSHOT_NAME",
    "create_map",
    "load_project",
    "persist_map",
    "snapshot_path",
]


def snapshot_path(workspace: Workspace) -> Path:
    """Return the path of the workspace's live-map snapshot."""
    return workspace.maps / SNAPSHOT_NAME


def create_map(workspace: Workspace, *, height: str = "100%", theme: str = "light") -> Map:
    """Create the shared map, restoring the last saved snapshot if present."""
    m = Map(height=height, layout="embed", theme=theme)
    snapshot = snapshot_path(workspace)
    if snapshot.exists():
        m.load_project(snapshot)
    else:
        m.load_project(build_empty_project(center=(0, 0), zoom=2))
    return m


@locked
def persist_map(map_obj: Map, workspace: Workspace) -> None:
    """Save the current project to the workspace snapshot and bump it."""
    map_obj.save_project(str(snapshot_path(workspace)))
    workspace.bump()


def load_project(map_obj: Map, project: dict[str, Any]) -> None:
    """Replace the live project with ``project`` (a project dict)."""
    map_obj.load_project(project)
