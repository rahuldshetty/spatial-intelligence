"""One image-analysis path shared by the tool and ``run_python``.

Both surfaces need the same things: resolve the live map or a workspace image,
turn it into model-ready JPEG bytes, and ask the configured vision model. The
tool returns the answer as text; ``vision.ask`` returns the same text to a
snippet. Keeping the source resolution here is what stops the two from drifting.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts.errors import ToolInputError
from ..contracts.services import CAPTURE_SERVICE, VISION_SERVICE
from ..workspace import Workspace
from .imaging import encode_capture, load_image

DEFAULT_MAX_SIDE = 1400
MIN_MAX_SIDE = 256
MAX_MAX_SIDE = 2048


def analyze_image_source(
    question: str,
    *,
    source: str = "map",
    workspace: Workspace,
    services: Mapping[str, Any],
    max_side: int = DEFAULT_MAX_SIDE,
    bands: list[int] | None = None,
    thinking: bool = False,
    max_tokens: int = 2048,
) -> str:
    """Ask the vision model ``question`` about ``source`` and return its text.

    ``source="map"`` captures the current GeoLibre viewport. A path is resolved
    inside ``workspace``. ``bands`` selects 1-based raster channels in R, G, B
    order and is rejected for the live map. The image is sent as JPEG (the same
    bounded encoding used by both callers); no image part is added to the main
    conversation.
    """
    service = services.get(VISION_SERVICE)
    if service is None:
        raise ToolInputError(
            "image input is not configured for this session; reopen the "
            "workspace after enabling a vision-capable model"
        )
    if not service.supported:
        raise ToolInputError(
            "image input is not available for the configured model; "
            "choose a vision-capable OpenAI-compatible model"
        )

    question = (question or "").strip()
    if not question:
        raise ToolInputError("question is required")
    limit = max(MIN_MAX_SIDE, min(int(max_side), MAX_MAX_SIDE))
    source = (source or "map").strip()

    if source == "map":
        if bands:
            raise ToolInputError(
                "bands only applies to a raster image, not the live map capture"
            )
        broker = services.get(CAPTURE_SERVICE)
        if broker is None:
            raise ToolInputError("the map capture service is not available")
        capture = broker.request(workspace=workspace.root.name)
        data, _, _ = encode_capture(capture.image, limit)
    else:
        path = workspace.resolve(source, must_exist=True)
        data, _, _, _, _, _ = load_image(path, limit, bands)

    return service.ask(
        question,
        data,
        "image/jpeg",
        thinking=thinking,
        max_tokens=max_tokens,
    )


__all__ = [
    "DEFAULT_MAX_SIDE",
    "MAX_MAX_SIDE",
    "MIN_MAX_SIDE",
    "analyze_image_source",
]
