"""The workspace tree: one notebook's inputs, outputs, maps, and traces."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from ..contracts.errors import WorkspaceError
from . import paths as _paths

__all__ = ["Workspace", "WorkspaceError", "now_iso"]


def now_iso() -> str:
    """Return the current UTC time as a second-resolution ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Workspace:
    """A workspace rooted at ``<app_root>/workspaces/<name>/``.

    Layout: ``data/`` (inputs), ``results/`` (outputs), ``maps/`` (saved
    ``.geolibre.json`` projects), ``traces/`` (per-run agent logs), plus a
    ``workspace.json`` manifest recording outputs and a monotonically increasing
    ``version`` counter.
    """

    def __init__(self, root: str | Path):
        # Resolved once so every confinement check and relative path compares
        # like with like (services resolve candidate paths before comparing).
        self.root = Path(root).resolve()
        self._manifest_path = self.root / "workspace.json"
        self._manifest_lock = threading.RLock()

    # -- layout ----------------------------------------------------------

    @property
    def data(self) -> Path:
        return self.root / _paths.DATA

    @property
    def results(self) -> Path:
        return self.root / _paths.RESULTS

    @property
    def maps(self) -> Path:
        return self.root / _paths.MAPS

    @property
    def traces(self) -> Path:
        return self.root / _paths.TRACES

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    # -- lifecycle -------------------------------------------------------

    def create(self) -> "Workspace":
        """Create the directory tree and manifest; idempotent."""
        for directory in (self.data, self.results, self.maps, self.traces):
            directory.mkdir(parents=True, exist_ok=True)
        if not self._manifest_path.exists():
            self._write_manifest(
                {"name": self.root.name, "created_at": now_iso(), "outputs": [], "version": 0}
            )
        return self

    # -- path confinement ------------------------------------------------

    def resolve(self, rel: str, *, must_exist: bool = False, write: bool = False) -> Path:
        """Resolve ``rel`` against the workspace root with confinement enforced."""
        return _paths.resolve_workspace_path(
            self.root, rel, must_exist=must_exist, write=write
        )

    def resolve_under(self, base: Path, rel: str) -> Path:
        """Resolve ``rel`` confined to ``base`` (for tools pinned to a subdir)."""
        return _paths.resolve_within(base, rel)

    # -- manifest --------------------------------------------------------

    def _read_manifest(self) -> dict:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"name": self.root.name, "created_at": now_iso(), "outputs": [], "version": 0}

    def _write_manifest(self, manifest: dict) -> None:
        manifest.setdefault("created_at", now_iso())
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def bump(self) -> None:
        """Increment the manifest ``version`` counter."""
        with self._manifest_lock:
            manifest = self._read_manifest()
            manifest["version"] = int(manifest.get("version", 0)) + 1
            self._write_manifest(manifest)

    def record_output(self, rel: str) -> None:
        """Append ``rel`` to the manifest outputs and bump the version."""
        with self._manifest_lock:
            manifest = self._read_manifest()
            outputs = manifest.setdefault("outputs", [])
            if rel not in outputs:
                outputs.append(rel)
            manifest["version"] = int(manifest.get("version", 0)) + 1
            self._write_manifest(manifest)

    def forget_output(self, rel: str) -> None:
        """Drop ``rel`` from the manifest outputs and bump the version."""
        with self._manifest_lock:
            manifest = self._read_manifest()
            outputs = [item for item in manifest.get("outputs", []) if item != rel]
            manifest["outputs"] = outputs
            manifest["version"] = int(manifest.get("version", 0)) + 1
            self._write_manifest(manifest)

    def move_output(self, old_rel: str, new_rel: str) -> None:
        """Replace an output path after a rename and bump the version."""
        with self._manifest_lock:
            manifest = self._read_manifest()
            outputs = [
                new_rel if item == old_rel else item
                for item in manifest.get("outputs", [])
            ]
            manifest["outputs"] = outputs
            manifest["version"] = int(manifest.get("version", 0)) + 1
            self._write_manifest(manifest)

    # -- convenience -----------------------------------------------------

    def list_files(self, subdir: str = "", pattern: str = "*") -> list[str]:
        """Recursively list files under ``subdir`` as sorted relative POSIX paths."""
        base = self.root if not subdir else self.resolve(subdir)
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in base.glob(f"**/{pattern}")
            if path.is_file()
        )

    def relative(self, path: Path) -> str:
        """Return ``path`` as a workspace-relative POSIX string."""
        return path.relative_to(self.root).as_posix()
