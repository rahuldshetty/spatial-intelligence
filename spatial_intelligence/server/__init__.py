"""Spatial Intelligence server: FastAPI backend plus the web UI it serves."""

from __future__ import annotations

import os

DEFAULT_PORT = 8000


def _open_browser(url: str) -> None:
    """Open the UI once the server has had a moment to come up.

    Suppressed with ``GEOAI_NO_BROWSER=1`` (used by packaging and tests).
    """
    if os.getenv("GEOAI_NO_BROWSER", "").strip().lower() in {"1", "true", "yes"}:
        return
    import threading
    import webbrowser

    threading.Timer(1.0, webbrowser.open, (url,)).start()


def _port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def run() -> None:
    """Validate configuration, ensure assets, and serve the app."""
    from ..settings.env import validate_env
    from .assets import ensure_frontend_assets

    validate_env()
    port = int(os.getenv("GEOAI_PORT", str(DEFAULT_PORT)))
    if _port_in_use(port):
        raise SystemExit(
            f"Spatial Intelligence: port {port} is already in use; "
            "set GEOAI_PORT to another port and start again."
        )
    ensure_frontend_assets()
    # Before the browser opens, so the first thing an operator sees is what is
    # about to serve: banner first, then the metadata blocks.
    from .banner import announce

    announce()
    _open_browser(f"http://127.0.0.1:{port}/")
    import uvicorn

    # An SSE client holds a long-lived connection that never completes on its
    # own, so a single Ctrl+C would otherwise wait forever for it to close.
    # Bound the grace period so shutdown force-cancels the stream and exits.
    uvicorn.run(
        "spatial_intelligence.server.app:app",
        host="127.0.0.1",
        port=port,
        reload=False,
        timeout_graceful_shutdown=3,
    )


__all__ = ["run"]
