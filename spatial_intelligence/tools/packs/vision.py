"""Image input: show the live map view or a workspace image to the model."""

from __future__ import annotations

from pydantic_ai import BinaryContent, ToolReturn

from ...agent.imaging import encode_capture, load_image
from ...contracts.effects import Effect
from ...contracts.errors import ToolInputError
from ...contracts.services import CAPTURE_SERVICE, VISION_SERVICE
from ..runtime import ToolRuntime
from ..spec import pack, tool


@pack(category="vision", effects=frozenset({Effect.READ, Effect.NETWORK}))
class VisionPack:
    """One generic image tool, gated by the endpoint's image-support probe."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._vision = runtime.services.get(VISION_SERVICE)

    def _require(self) -> None:
        if self._vision is None or not self._vision.supported:
            raise ToolInputError(
                "image input is not available for the configured model; "
                "choose a vision-capable OpenAI-compatible model"
            )

    @tool(core=True)
    def analyze_image(
        self,
        question: str,
        source: str = "map",
        max_side: int = 1400,
        bands: list[int] | None = None,
    ) -> ToolReturn:
        """Look at the live map or a workspace image and answer a question about it.

        ``source="map"`` captures exactly what the GeoLibre map canvas is showing
        now — the current viewport, without toolbar, legend, or scale bar — and
        passes it with the view bounds. ``source`` may instead be a workspace
        path to a PNG/JPEG/WebP or a GeoTIFF; a raster is downsampled,
        percentile-stretched, and converted to an RGB JPEG. ``bands`` selects the
        raster channels in R, G, B order (1-based); it defaults to the first
        three bands. Pass it when the raster is not true-colour, e.g.
        ``bands=[4, 3, 2]``. A one-band raster is rendered as grayscale.
        ``question`` is what you want read from the image.

        Use this for "what do you see", layer/layout checks, or approximate
        visual boxes. For precise measurements or geometry, use the raster and
        vector tools instead; the image is only the visible viewport, not the
        whole dataset.
        """
        self._require()
        question = (question or "").strip()
        if not question:
            raise ToolInputError("question is required")
        limit = max(256, min(int(max_side), 2048))
        source = (source or "map").strip()

        if source == "map":
            if bands:
                raise ToolInputError(
                    "bands only applies to a raster source; the map capture is already RGB"
                )
            broker = self._rt.services.get(CAPTURE_SERVICE)
            if broker is None:
                raise ToolInputError("the map capture service is not available")
            capture = broker.request(workspace=self._rt.workspace.root.name)
            data, width, height = encode_capture(capture.image, limit)
            view = capture.view or {}
            bounds = view.get("bbox")
            location = f"current map view, bbox {bounds}" if bounds else "current map view"
            if view.get("zoom") is not None:
                location += f", zoom {float(view['zoom']):.2f}"
            if view.get("pitch"):
                location += f", pitch {float(view['pitch']):.1f}°"
            if view.get("bearing"):
                location += f", bearing {float(view['bearing']):.1f}°"
            detail = (
                f"Attached: {location}. Image {width}x{height} "
                f"(canvas {capture.width}x{capture.height}). The capture is the map "
                "canvas only; no UI, legend, or scale bar is included."
            )
        else:
            path = self._rt.workspace.resolve(source, must_exist=True)
            data, width, height, original_width, original_height, selected = load_image(
                path, limit, bands
            )
            if path.suffix.lower() in {".tif", ".tiff", ".img", ".vrt", ".jp2"}:
                conversion = (
                    f"Converted to an RGB JPEG from bands {selected} "
                    "(2nd–98th percentile per band)"
                )
            else:
                conversion = "Converted to an RGB JPEG"
            detail = (
                f"Attached: {self._rt.workspace.relative(path)} "
                f"({original_width}x{original_height}), {conversion}, "
                f"shown at {width}x{height}."
            )
            if min(original_width, original_height) < 512:
                detail += (
                    " The source is small; individual small objects may span only "
                    "a few pixels."
                )

        return ToolReturn(
            detail,
            content=[
                BinaryContent(
                    data=data,
                    media_type="image/jpeg",
                    vendor_metadata={"detail": "auto"},
                )
            ],
        )


__all__ = ["VisionPack"]
