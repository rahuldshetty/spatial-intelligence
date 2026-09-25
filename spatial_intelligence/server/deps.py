"""The process-wide session, created on first use.

Kept behind a function so importing a router (or the app) never constructs the
live map, opens a workspace, or starts the run worker — tests and tooling can
import the server package without side effects.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from ..session.app_state import AppState
from ..map.bridge import reset_bridge
from ..settings.env import load_env

#: Seconds to wait for a running cell before stopping the worker anyway.
#: Uvicorn awaits the lifespan shutdown with no timeout of its own, so this has
#: to be short: the first Ctrl+C must end the process, not wait out a queued
#: batch. A worker that outlasts it still leaves once its cell finishes.
SHUTDOWN_TIMEOUT = 5.0


@lru_cache(maxsize=1)
def app_state() -> AppState:
    """Return the process-wide :class:`AppState`, creating it once."""
    load_env()
    return AppState()


def close_app_state() -> None:
    """Shut the process-wide session down: stop its worker, then drop it.

    Called on server shutdown so the run worker does not outlive the process's
    serving loop, and by tests that replace the singleton between cases —
    dropping a cached ``AppState`` without stopping it leaves its worker thread
    reading a workspace that is about to be deleted.
    """
    if app_state.cache_info().currsize:
        if not app_state().stop_worker(SHUTDOWN_TIMEOUT):
            print(
                "spatial-intelligence: a cell was still running when the server "
                f"stopped; waited {SHUTDOWN_TIMEOUT:g}s for it before leaving",
                file=sys.stderr,
            )
    app_state.cache_clear()
    reset_bridge()
