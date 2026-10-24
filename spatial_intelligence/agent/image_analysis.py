"""The image-analysis path shared by the ``analyze_image`` tool and ``vision.ask``.

Both resolve the live map or a workspace image, encode it as a bounded JPEG,
and ask the configured vision model. The tool returns the answer plus capture
context; a Python snippet gets the answer text alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts.errors import ToolInputError
from ..contracts.services import CAPTURE_SERVICE, VISION_SERVICE
from ..workspace import Workspace
from .imaging import encode_capture, load_image

DEFAULT_MAX_SIDE = 1400
MIN_MAX_SIDE = 256
MAX_MAX_SIDE = 2048

_RASTER_SUFFIXES = frozenset({".tif", ".tiff", ".img", ".vrt", ".jp2"})


@dataclass(frozen=True, slots=True)
class ImageAnswer:
    """The model's answer and what the model was looking at."""

    text: str
    context: str = ""


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
) -> ImageAnswer:
    """Ask the vision model ``question`` about ``source``.

    ``source="map"`` captures the current GeoLibre viewport; a path is resolved
    inside ``workspace``. ``bands`` selects 1-based raster channels in R, G, B
    order and is rejected for the live map.
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
        data, width, height = encode_capture(capture.image, limit)
        context = _map_context(capture, width, height)
    else:
        path = workspace.resolve(source, must_exist=True)
        data, width, height, original_width, original_height, selected = load_image(
            path, limit, bands
        )
        context = _image_context(
            workspace,
            path,
            width,
            height,
            original_width,
            original_height,
            selected,
        )

    text = service.ask(
        question,
        data,
        "image/jpeg",
        thinking=thinking,
        max_tokens=max_tokens,
    )
    return ImageAnswer(text=text, context=context)


def _map_context(capture: Any, width: int, height: int) -> str:
    """Describe the captured viewport and rendered image for the model."""
    view = getattr(capture, "view", None) or {}
    parts = []
    bbox = view.get("bbox")
    parts.append(f"current map view, bbox {bbox}" if bbox else "current map view")
    if view.get("zoom") is not None:
        parts.append(f"zoom {float(view['zoom']):.2f}")
    if view.get("pitch"):
        parts.append(f"pitch {float(view['pitch']):.1f}°")
    if view.get("bearing"):
        parts.append(f"bearing {float(view['bearing']):.1f}°")
    return (
        f"Capture: {', '.join(parts)}. Image {width}x{height} "
        f"(canvas {capture.width}x{capture.height}). The capture is the map "
        "canvas only; no UI, legend, or scale bar is included."
    )


def _image_context(
    workspace: Workspace,
    path: Any,
    width: int,
    height: int,
    original_width: int,
    original_height: int,
    selected: list[int] | None,
) -> str:
    """Describe the workspace image and the conversion for the model."""
    if path.suffix.lower() in _RASTER_SUFFIXES:
        conversion = (
            f"converted to an RGB JPEG from bands {selected} "
            "(2nd–98th percentile per band)"
        )
    else:
        conversion = "converted to an RGB JPEG"
    context = (
        f"Image: {workspace.relative(path)} ({original_width}x{original_height}), "
        f"{conversion}, shown at {width}x{height}."
    )
    if min(original_width, original_height) < 512:
        context += (
            " The source is small; individual small objects may span only a few "
            "pixels."
        )
    return context


__all__ = [
    "DEFAULT_MAX_SIDE",
    "MAX_MAX_SIDE",
    "MIN_MAX_SIDE",
    "ImageAnswer",
    "analyze_image_source",
]
