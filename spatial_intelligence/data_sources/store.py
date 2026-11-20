"""Persistence for the data-source registry configuration.

Same two scopes as Tool Configuration:

* ``<data root>/workspaces/data_sources.json`` — global.
* ``<workspace>/data_sources.json`` — a workspace override.

Only non-secret settings live here.  A source that needs a key names an
environment variable instead of writing the key into this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..settings import env

FILENAME = "data_sources.json"


def default_data_source_config() -> dict:
    """Return the empty configuration (every source enabled)."""
    return {"disabled_sources": [], "options": {}}


def normalize_data_source_config(config: Any) -> dict:
    """Return a well-formed configuration, preserving unknown options."""
    data = config if isinstance(config, dict) else {}
    disabled = data.get("disabled_sources")
    if isinstance(disabled, str):
        disabled = [disabled]
    names: list[str] = []
    seen: set[str] = set()
    for item in disabled or []:
        name = str(item).strip() if item is not None else ""
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    options = data.get("options")
    return {
        "disabled_sources": sorted(names),
        "options": dict(options) if isinstance(options, dict) else {},
    }


def global_data_sources_path() -> Path:
    """Return the global ``data_sources.json`` under the workspaces root."""
    return env.app_root() / "workspaces" / FILENAME


def _workspace_root(workspace: object) -> Path:
    if isinstance(workspace, (str, Path)):
        return Path(workspace)
    return Path(workspace.root)


def workspace_data_sources_path(workspace: object) -> Path:
    """Return one workspace's ``data_sources.json``."""
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


def load_global_data_source_config() -> dict:
    """Return the global configuration, or the empty default."""
    data = _load_json(global_data_sources_path())
    return normalize_data_source_config(data) if data is not None else default_data_source_config()


def load_workspace_data_source_config(workspace: object) -> dict | None:
    """Return a workspace override, or ``None`` when it follows global."""
    data = _load_json(workspace_data_sources_path(workspace))
    return normalize_data_source_config(data) if data is not None else None


def save_global_data_source_config(config: dict) -> dict:
    """Write the global configuration and return the normalized value."""
    normalized = normalize_data_source_config(config)
    _write_json(global_data_sources_path(), normalized)
    return normalized


def save_workspace_data_source_config(workspace: object, config: dict) -> dict:
    """Write a workspace override and return the normalized value."""
    normalized = normalize_data_source_config(config)
    _write_json(workspace_data_sources_path(workspace), normalized)
    return normalized


def effective_data_source_config(workspace: object | None) -> dict:
    """Return the workspace override when present, otherwise the global config."""
    if workspace is not None:
        stored = load_workspace_data_source_config(workspace)
        if stored is not None:
            return stored
    return load_global_data_source_config()


__all__ = [
    "FILENAME",
    "default_data_source_config",
    "effective_data_source_config",
    "global_data_sources_path",
    "load_global_data_source_config",
    "load_workspace_data_source_config",
    "normalize_data_source_config",
    "save_global_data_source_config",
    "save_workspace_data_source_config",
    "workspace_data_sources_path",
]
