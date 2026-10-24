"""Tool Configuration persistence in ``tools.json``.

Two files share one name and one schema:

* ``<data root>/workspaces/tools.json`` — the global configuration.
* ``<data root>/workspaces/<name>/tools.json`` — a workspace override.

``settings.json`` and ``workspace.json`` keep their existing responsibilities;
tool configuration is deliberately separate. Configurations written by the
first implementation lived in those two files, so reads fall back to them once
and :func:`migrate_legacy_tool_configs` moves them into ``tools.json`` at
session start.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..settings import env, prefs
from .config import default_tool_config, normalize_tool_config

#: Filename used at both scopes.
FILENAME = "tools.json"


def global_tools_path() -> Path:
    """Return the global ``tools.json`` under the workspaces root."""
    return env.app_root() / "workspaces" / FILENAME


def _workspace_root(workspace: object) -> Path:
    """Accept a ``Workspace`` or a plain path without tripping on ``Path.root``."""
    if isinstance(workspace, (str, Path)):
        return Path(workspace)
    return Path(workspace.root)


def workspace_tools_path(workspace: object) -> Path:
    """Return one workspace's ``tools.json`` (inside its context)."""
    return _workspace_root(workspace) / FILENAME


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _read_config(path: Path) -> dict | None:
    data = _load_json(path)
    return normalize_tool_config(data) if data is not None else None


def load_global_tool_config() -> dict:
    """Return the global configuration, or the empty default."""
    return _read_config(global_tools_path()) or default_tool_config()


def load_workspace_tool_config(workspace: object) -> dict | None:
    """Return a workspace override, or ``None`` when it follows global."""
    return _read_config(workspace_tools_path(workspace))


def save_global_tool_config(config: dict) -> dict:
    """Write the global ``tools.json`` and drop the legacy settings key."""
    normalized = normalize_tool_config(config)
    _write_json(global_tools_path(), normalized)
    _drop_settings_key(prefs.settings_path())
    return normalized


def save_workspace_tool_config(workspace: object, config: dict) -> dict:
    """Write ``<workspace>/tools.json`` and drop the legacy manifest key."""
    normalized = normalize_tool_config(config)
    _write_json(workspace_tools_path(workspace), normalized)
    clear_legacy_manifest_key(workspace)
    return normalized


def _drop_settings_key(path: Path) -> None:
    """Remove ``tool_config`` from ``settings.json`` without touching the rest."""
    data = _load_json(path)
    if data is None or "tool_config" not in data:
        return
    data.pop("tool_config", None)
    _write_json(path, data)


def clear_legacy_manifest_key(workspace: object) -> None:
    """Remove ``tool_config`` from a workspace manifest, if present."""
    manifest_path = _workspace_root(workspace) / "workspace.json"
    data = _load_json(manifest_path)
    if data is None or "tool_config" not in data:
        return
    data.pop("tool_config", None)
    _write_json(manifest_path, data)


def migrate_legacy_tool_configs() -> list[Path]:
    """Move configurations from the first implementation into ``tools.json``.

    Idempotent and local: it runs at session start, writes a new file only when
    the old key is present, and removes the key afterwards so settings and
    manifests stop mixing tool policy with their own state. A file that cannot
    be read or written is left alone; the app keeps working with the next
    readable scope.
    """
    migrated: list[Path] = []

    settings_path = prefs.settings_path()
    settings_data = _load_json(settings_path)
    if settings_data is not None and isinstance(settings_data.get("tool_config"), dict):
        target = global_tools_path()
        if _read_config(target) is None:
            try:
                _write_json(target, normalize_tool_config(settings_data["tool_config"]))
                migrated.append(target)
            except OSError:
                pass
        if _read_config(target) is not None:
            try:
                _drop_settings_key(settings_path)
            except OSError:
                pass

    base = env.app_root() / "workspaces"
    if base.is_dir():
        for workspace_root in sorted(path for path in base.iterdir() if path.is_dir()):
            manifest_path = workspace_root / "workspace.json"
            manifest = _load_json(manifest_path)
            if manifest is None or not isinstance(manifest.get("tool_config"), dict):
                continue
            target = workspace_root / FILENAME
            if _read_config(target) is None:
                try:
                    _write_json(
                        target, normalize_tool_config(manifest["tool_config"])
                    )
                    migrated.append(target)
                except OSError:
                    continue
            if _read_config(target) is None:
                continue
            try:
                clear_legacy_manifest_key(workspace_root)
            except OSError:
                pass
    return migrated


__all__ = [
    "FILENAME",
    "clear_legacy_manifest_key",
    "global_tools_path",
    "load_global_tool_config",
    "load_workspace_tool_config",
    "migrate_legacy_tool_configs",
    "save_global_tool_config",
    "save_workspace_tool_config",
    "workspace_tools_path",
]
