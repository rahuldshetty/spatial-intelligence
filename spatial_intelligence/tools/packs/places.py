"""Place-name lookup, separate from the data-source registry."""

from __future__ import annotations

from ...contracts.effects import Effect
from ...geo import places
from ..runtime import ToolRuntime
from ..spec import pack, tool


@pack(category="geo", effects=frozenset({Effect.READ, Effect.NETWORK}))
class PlacesPack:
    """Resolve a place name to an area of interest."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime

    @tool()
    def geocode(self, place: str) -> dict:
        """Resolve a place name to the bounding box of an area of interest.

        Use this when the user names a place instead of giving coordinates. The
        result carries ``bbox`` as ``[west, south, east, north]``, which is what
        the data-source, raster, and vector tools take.
        """
        return places.geocode(place)


__all__ = ["PlacesPack"]
