"""App-level user settings persisted to ``<app_root>/settings.json``.

Settings are runtime-editable (the Settings dialog): model and UI theme.
Environment variables seed the first-run defaults; once a value is
written to the file it takes precedence.

Ported from ``geoai/settings.py``; the keys and normalization rules are
unchanged so an existing ``settings.json`` keeps working.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import env

_THEMES = ("light", "dark")


def settings_path() -> Path:
    """Return the settings file location.

    Resolved at call time so a ``GEOAI_HOME`` loaded from ``.env`` is honored.
    """
    return env.app_root() / "settings.json"


def _defaults() -> dict:
    return {
        "model": env.model_from_env() or env.DEFAULT_MODEL,
        "theme": "light",
        "dangerous_mode": False,
        "max_retries": env.max_retries(),
        "max_requests": env.max_requests(),
        "context_window": env.context_window(),
        "record_agent_steps": True,
    }

def load_settings() -> dict:
    """Return the merged settings (env defaults overlaid by the file)."""
    settings = _defaults()
    path = settings_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
        if isinstance(data, dict):
            for key in settings:
                value = data.get(key)
                if value not in (None, ""):
                    settings[key] = value
    return _normalize(settings)


def save_settings(settings: dict) -> dict:
    """Normalize and persist ``settings``; return the normalized dict."""
    normalized = _normalize(settings)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return normalized


def _normalize(settings: dict) -> dict:
    defaults = _defaults()
    model = (settings.get("model") or "").strip() or defaults["model"]
    theme = settings.get("theme") if settings.get("theme") in _THEMES else "light"
    try:
        retries = int(settings.get("max_retries", defaults["max_retries"]))
    except (TypeError, ValueError):
        retries = defaults["max_retries"]
    try:
        requests = int(settings.get("max_requests", defaults["max_requests"]))
    except (TypeError, ValueError):
        requests = defaults["max_requests"]
    try:
        window = int(settings.get("context_window", defaults["context_window"]))
    except (TypeError, ValueError):
        window = defaults["context_window"]
    return {
        "model": model,
        "theme": theme,
        "dangerous_mode": bool(settings.get("dangerous_mode", False)),
        "max_retries": max(1, retries),
        "max_requests": max(1, requests),
        "context_window": max(0, window),
        "record_agent_steps": bool(settings.get("record_agent_steps", True)),
    }
