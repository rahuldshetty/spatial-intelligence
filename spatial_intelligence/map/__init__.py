"""Map domain: the live GeoLibre document, its persistence, and the iframe bridge."""

from __future__ import annotations

import threading

#: Serializes every mutation of the shared live map.
#:
#: Tool calls from one model response run in parallel, and the browser pushes
#: its own project back on every ``geolibre:state``. Both eventually do a
#: read-copy-reassign of ``Map.project``; without one lock the later writer can
#: silently drop a layer the other just added. ``Map`` is process-wide, so the
#: lock must be too.
MAP_LOCK = threading.RLock()

__all__ = ["MAP_LOCK"]
