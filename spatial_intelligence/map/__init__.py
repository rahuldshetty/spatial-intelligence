"""Map domain: the live GeoLibre document, its persistence, and the iframe bridge."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any, TypeVar

#: Serializes every mutation and paired read of the shared live map.
#:
#: Map operations are read-copy-reassign, so a browser push and a tool that run
#: concurrently can silently drop each other's layer. The lock lives with the
#: map, not with the callers: mutations acquire it in this package, and readers
#: that need a consistent project/revision pair acquire it once.
MAP_LOCK = threading.RLock()

_T = TypeVar("_T")


def locked(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Run ``fn`` while holding :data:`MAP_LOCK` (re-entrant)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        with MAP_LOCK:
            return fn(*args, **kwargs)

    return wrapper
