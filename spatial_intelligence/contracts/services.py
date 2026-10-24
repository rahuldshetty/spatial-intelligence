"""Service-bag keys shared across layers without importing those layers."""

#: The configured model's image-input profile.
VISION_SERVICE = "vision.service"

#: The request/response bridge to the live map canvas.
CAPTURE_SERVICE = "map.capture"

#: One-shot map queries (currently the user's drawn sketches).
MAP_BRIDGE_SERVICE = "map.bridge"

#: The open workspace's searchable conversation index.
CONVERSATION_SERVICE = "conversation.store"

__all__ = [
    "CAPTURE_SERVICE",
    "CONVERSATION_SERVICE",
    "MAP_BRIDGE_SERVICE",
    "VISION_SERVICE",
]
