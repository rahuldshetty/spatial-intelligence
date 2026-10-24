"""Image analysis: ask the configured vision model about the map or an image."""

from __future__ import annotations

from ...agent.image_analysis import analyze_image_source
from ...contracts.effects import Effect
from ..runtime import ToolRuntime
from ..spec import pack, tool


@pack(category="vision", effects=frozenset({Effect.READ, Effect.NETWORK}))
class VisionPack:
    """One text-returning vision tool, gated by the endpoint's image-support probe."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime

    @tool(core=True)
    def analyze_image(
        self,
        question: str,
        source: str = "map",
        max_side: int = 1400,
        bands: list[int] | None = None,
    ) -> str:
        """Ask the configured vision model a question about the map or an image.

        ``source="map"`` captures exactly what the GeoLibre map canvas is showing
        now — the current viewport, without toolbar, legend, or scale bar.
        ``source`` may instead be a workspace path to a PNG/JPEG/WebP or a
        GeoTIFF; a raster is downsampled, percentile-stretched, and converted to
        an RGB JPEG. ``bands`` selects the raster channels in R, G, B order
        (1-based); it defaults to the first three bands. Pass it when the raster
        is not true-colour, e.g. ``bands=[4, 3, 2]``. A one-band raster is
        rendered as grayscale.

        The model's text answer is returned with the capture's bbox/zoom (or
        the image's path and scale), so normalized boxes can be georeferenced.
        No image part is added to this conversation. ``run_python`` exposes the
        same implementation as ``vision.ask(...)`` and adds ``thinking`` and
        ``max_tokens`` overrides.
        """
        answer = analyze_image_source(
            question,
            source=source,
            workspace=self._rt.workspace,
            services=self._rt.services,
            max_side=max_side,
            bands=bands,
        )
        return f"{answer.text}\n\n{answer.context}" if answer.context else answer.text


__all__ = ["VisionPack"]
