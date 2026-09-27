"""Runtime description of the GeoLibre iframe's versioned embed bridge."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_info: dict = {
    "connected": False,
    "version": None,
    "protocol_version": 1,
    "methods": [],
}


def update_bridge(version: str | None, methods: list[str] | None = None) -> dict:
    """Record capabilities reported by the currently connected iframe."""
    with _lock:
        _info.update(
            {
                "connected": True,
                "version": version,
                "protocol_version": 1,
                "methods": sorted(set(methods or [])),
            }
        )
        return dict(_info)


def reset_bridge() -> None:
    """Forget the last handshake: no iframe is connected to this session.

    The record describes *the iframe currently on screen*, so it has to be
    cleared when that iframe is replaced: switching workspace rebuilds it (the app
    recreates the iframe when the workspace name changes) and so does closing one,
    and the resulting iframe posts its own handshake. Re-opening the workspace that
    is already open leaves the iframe alone, so its opener does not reset this.
    """
    with _lock:
        _info.update(
            {
                "connected": False,
                "version": None,
                "protocol_version": 1,
                "methods": [],
            }
        )


def bridge_info() -> dict:
    """Return the last capability handshake from the GeoLibre iframe."""
    with _lock:
        return dict(_info)
