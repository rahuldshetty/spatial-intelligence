"""The workspace currently open in this process, and its live map."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from geolibre import Map
from geolibre import project as geolibre_project

from ..contracts.errors import ToolInputError
from ..map import document
from ..settings.env import workspace_root
from ..workspace import Workspace

EMPTY_VIEW = {"center": (0, 0), "zoom": 2}


class WorkspaceSession:
    """Owns the active workspace and the single shared live map.

    The map object is created once and reused across workspaces (the browser
    iframe keeps pointing at it), exactly as the previous implementation did.
    """

    def __init__(self, root_for: Callable[[str], Path] = workspace_root) -> None:
        self._root_for = root_for
        self.name: str | None = None
        self.workspace: Workspace | None = None
        self.map = Map(
            center=EMPTY_VIEW["center"],
            zoom=EMPTY_VIEW["zoom"],
            height="100%",
            layout="embed",
            theme="light",
        )

    # -- lifecycle -------------------------------------------------------

    def open(self, name: str) -> Workspace:
        """Open (creating if needed) a workspace and load its map snapshot."""
        workspace = Workspace(self._root_for(name)).create()
        self.name = name
        self.workspace = workspace
        snapshot = document.snapshot_path(workspace)
        if snapshot.exists():
            self.map.load_project(snapshot)
        else:
            self.map.load_project(
                geolibre_project.build_empty_project(
                    center=EMPTY_VIEW["center"], zoom=EMPTY_VIEW["zoom"]
                )
            )
        return workspace

    def new(self, name: str) -> Workspace:
        """Create a workspace directory and open it, refusing an existing one.

        Creating over an existing directory would open its old notebook and map
        instead — "new" has to mean new, so the name is checked first.
        """
        root = self._root_for(name)
        if root.exists():
            raise ToolInputError(
                f"a workspace named {name!r} already exists — open it, or pick another name"
            )
        return self.open(name)

    def close(self) -> None:
        """Forget the open workspace and reset the map to an empty project."""
        self.name = None
        self.workspace = None
        self.map.load_project(
            geolibre_project.build_empty_project(
                center=EMPTY_VIEW["center"], zoom=EMPTY_VIEW["zoom"]
            )
        )

    # -- map -------------------------------------------------------------

    def save_map(self) -> None:
        """Persist the live map into the workspace snapshot."""
        if self.workspace is not None:
            document.persist_map(self.map, self.workspace)

    def set_map_project(self, project: dict) -> None:
        """Adopt a project pushed by the browser and persist it."""
        if self.workspace is None:
            raise ValueError("no workspace open")
        self.map.load_project(project)
        self.map.save_project(str(document.snapshot_path(self.workspace)))
        self.workspace.bump()

    def project(self) -> dict:
        """The live project, as sent to the browser."""
        return self.map.to_project()

    # -- files -----------------------------------------------------------

    def list_files(self, subdir: str = "") -> list[str]:
        """List workspace files, or nothing when no workspace is open."""
        return self.workspace.list_files(subdir) if self.workspace else []
