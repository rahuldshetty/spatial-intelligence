"""Ask the browser for the current map view, and wait for the PNG.

Tools run on the server, but only the live GeoLibre iframe can capture what the
user is looking at. The broker publishes one request over SSE; the app's message
bridge captures the canvas and posts the result back. Nothing here holds the
session lock while waiting: the browser needs the event stream and the callback
route to stay responsive.
"""

from __future__ import annotations

import base64
import io
import secrets
import threading
import uuid
from dataclasses import dataclass

from PIL import Image

from ..contracts.errors import ToolInputError
from ..contracts.services import CAPTURE_SERVICE
from .bus import EventBus

#: How long a tool waits for the browser to answer one capture.
CAPTURE_TIMEOUT = 20.0

#: A data URL larger than this is not a map screenshot; reject it early.
MAX_CAPTURE_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MapCapture:
    """One captured view: the image plus the metadata needed to place it."""

    image: bytes
    media_type: str
    width: int
    height: int
    view: dict


class _Pending:
    def __init__(self, token: str) -> None:
        self.token = token
        self.event = threading.Event()
        self.capture: MapCapture | None = None
        self.error: str | None = None


class CaptureBroker:
    """Thread-safe request/response bridge to the browser's map canvas."""

    def __init__(self, bus: EventBus, *, timeout: float = CAPTURE_TIMEOUT) -> None:
        self._bus = bus
        self._timeout = timeout
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    def request(self, *, workspace: str | None = None, timeout: float | None = None) -> MapCapture:
        """Ask the browser for the current view and block until it answers."""
        if self._bus.subscriber_count() == 0:
            raise ToolInputError(
                "the live map is not available: open the app and keep the map visible "
                "before asking the agent to look at it"
            )
        capture_id = uuid.uuid4().hex
        pending = _Pending(secrets.token_urlsafe(18))
        with self._lock:
            self._pending[capture_id] = pending
        self._bus.publish(
            "map_capture",
            {"id": capture_id, "token": pending.token, "workspace": workspace},
        )
        wait = self._timeout if timeout is None else max(1.0, float(timeout))
        if not pending.event.wait(wait):
            with self._lock:
                self._pending.pop(capture_id, None)
            raise ToolInputError(
                "the map did not answer the capture request; make sure the app is "
                "open and the map has finished loading"
            )
        if pending.error:
            raise ToolInputError(pending.error)
        if pending.capture is None:
            raise ToolInputError("the map returned an empty capture")
        return pending.capture

    def complete(
        self,
        capture_id: str,
        token: str,
        *,
        image: str | None = None,
        view: dict | None = None,
        error: str | None = None,
    ) -> bool:
        """Resolve one pending request; return whether it was accepted."""
        with self._lock:
            pending = self._pending.get(capture_id)
        if pending is None or not secrets.compare_digest(pending.token, token):
            return False
        if not error:
            try:
                capture = _decode_capture(image or "", view or {})
            except ToolInputError as exc:
                error = str(exc)
        with self._lock:
            # Another tab may have answered between the two lock sections; the
            # first valid answer wins and the rest are no-ops.
            pending = self._pending.pop(capture_id, None)
        if pending is None:
            return False
        if error:
            pending.error = error
        else:
            pending.capture = capture
        pending.event.set()
        return True

    def cancel_all(self, reason: str = "the map was closed") -> None:
        """Fail every waiting request (workspace switch or shutdown)."""
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.error = reason
            item.event.set()


def _decode_capture(image: str, view: dict) -> MapCapture:
    """Decode a data URL (or raw base64) and read its real dimensions."""
    payload = image.strip()
    media_type = "image/png"
    if payload.startswith("data:"):
        header, separator, body = payload.partition(",")
        if not separator or ";base64" not in header:
            raise ToolInputError("the map returned an unsupported image payload")
        media_type = header[5:].split(";", 1)[0] or media_type
        payload = body
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001 - malformed base64 from the browser
        raise ToolInputError("the map returned invalid image data") from exc
    if not data:
        raise ToolInputError("the map returned an empty image")
    if len(data) > MAX_CAPTURE_BYTES:
        raise ToolInputError("the map capture is too large")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            format_name = (probe.format or "").upper()
    except Exception as exc:  # noqa: BLE001
        raise ToolInputError("the map returned a file that is not a readable image") from exc
    if format_name == "PNG":
        media_type = "image/png"
    elif format_name in {"JPEG", "JPG"}:
        media_type = "image/jpeg"
    return MapCapture(
        image=data,
        media_type=media_type,
        width=width,
        height=height,
        view=dict(view),
    )


__all__ = ["CAPTURE_SERVICE", "CaptureBroker", "MapCapture"]
