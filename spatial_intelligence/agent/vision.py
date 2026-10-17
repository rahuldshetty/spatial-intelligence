"""Image-input support for the configured OpenAI-compatible endpoint.

The check is deliberately a real request with a tiny image: model listings do
not reliably say whether a model accepts vision, and self-hosted servers that
claim an OpenAI API often differ. The result gates the ``analyze_image`` tool,
so a text-only endpoint never advertises an image tool it cannot serve.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from collections.abc import Callable

import httpx

from ..contracts.errors import ToolInputError
from ..contracts.services import VISION_SERVICE
from .model import OpenAIEndpoint, openai_endpoint

#: How long one probe may take before it is reported as unknown.
PROBE_TIMEOUT = 12.0
#: How long one image question may take.
ASK_TIMEOUT = 120.0
#: Re-probe an unchanged model after this many seconds.
PROBE_TTL = 600.0

_IMAGE_TERMS = ("image", "vision", "multimodal", "media type", "content type")


def _probe_png() -> bytes:
    """Return a small opaque image; 1x1 is rejected by some processors."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 40, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _data_uri(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _headers(endpoint: OpenAIEndpoint) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    return headers


def _payload(endpoint: OpenAIEndpoint, question: str, data_uri: str, *, max_tokens: int) -> dict:
    if endpoint.api == "responses":
        return {
            "model": endpoint.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question},
                        {"type": "input_image", "image_url": data_uri},
                    ],
                }
            ],
            "max_output_tokens": max_tokens,
        }
    return {
        "model": endpoint.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }


def _is_deepseek(endpoint: OpenAIEndpoint) -> bool:
    """Whether the endpoint is DeepSeek's OpenAI-compatible API."""
    return "deepseek" in endpoint.base_url.lower() or endpoint.model.lower().startswith(
        "deepseek"
    )


def _thinking_payload(endpoint: OpenAIEndpoint, thinking: bool) -> dict:
    """DeepSeek's direct-answer switch; other providers do not know the field."""
    if not _is_deepseek(endpoint):
        return {}
    return {"thinking": {"type": "enabled" if thinking else "disabled"}}


def _endpoint_url(endpoint: OpenAIEndpoint) -> str:
    route = "/responses" if endpoint.api == "responses" else "/chat/completions"
    return endpoint.base_url.rstrip("/") + route


def _extract_text(data: dict) -> str:
    """Return assistant text from either OpenAI response shape."""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
    text = data.get("output_text")
    if isinstance(text, str):
        return text
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "".join(parts)


class VisionService:
    """Probe and use image input on the configured model.

    ``on_change`` is called after the profile changes, from the probe thread.
    The callback owns re-registering the tool; this class only reports state.
    """

    def __init__(
        self,
        model: str,
        *,
        on_change: Callable[[], None] | None = None,
        enabled: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._model = model
        self._on_change = on_change
        self._enabled = bool(enabled)
        self._state = "disabled" if not enabled else "unknown"
        self._detail = "" if enabled else "Image-support probe is disabled."
        self._checked_at = 0.0
        self._generation = 0
        self._thread: threading.Thread | None = None

    # -- state -----------------------------------------------------------

    @property
    def supported(self) -> bool:
        with self._lock:
            return self._state == "supported"

    def snapshot(self) -> dict:
        """Return the status the browser renders (no secrets)."""
        with self._lock:
            return {
                "available": self._state == "supported",
                "state": self._state,
                "detail": self._detail,
                "model": self._model,
            }

    def _set(self, state: str, detail: str = "") -> None:
        with self._lock:
            self._state = state
            self._detail = detail
            self._checked_at = time.monotonic()
        self._notify()

    def _notify(self) -> None:
        callback = self._on_change
        if callback is None:
            return
        try:
            callback()
        except Exception:
            # A broken callback must not kill the probe thread.
            pass

    # -- probing ---------------------------------------------------------

    def start(self, *, force: bool = False) -> None:
        """Probe once, unless disabled or already running.

        ``force`` starts a new probe even when the previous thread is still
        alive; its generation is stale and its result will be discarded.
        """
        with self._lock:
            if not self._enabled:
                return
            if not force and self._thread is not None and self._thread.is_alive():
                return
            if (
                not force
                and self._state == "supported"
                and (time.monotonic() - self._checked_at) < PROBE_TTL
            ):
                return
            self._generation += 1
            generation = self._generation
            model = self._model
            self._state = "checking"
            self._detail = "Checking whether this model accepts images…"
            thread = threading.Thread(
                target=self._run_probe,
                args=(generation, model),
                name="vision-probe",
                daemon=True,
            )
            self._thread = thread
        self._notify()
        thread.start()

    def reconfigure(self, model: str) -> None:
        """Probe a new model; the old probe's result is discarded."""
        with self._lock:
            if model == self._model and self._state in {"supported", "checking"}:
                self._model = model
                return
            self._model = model
            self._checked_at = 0.0
        self.start(force=True)

    def _run_probe(self, generation: int, model: str) -> None:
        try:
            state, detail = _probe(model)
        except Exception as exc:  # noqa: BLE001 - a failed probe must not hang "checking"
            state, detail = "error", f"Image probe failed: {exc}"
        with self._lock:
            if generation != self._generation:
                return
            self._state = state
            self._detail = detail
            self._checked_at = time.monotonic()
        self._notify()

    # -- use -------------------------------------------------------------

    def ask(
        self,
        question: str,
        data: bytes,
        media_type: str = "image/jpeg",
        *,
        timeout: float = ASK_TIMEOUT,
        thinking: bool = False,
        max_tokens: int = 2048,
    ) -> str:
        """Ask the configured model one question about one image.

        ``thinking`` is DeepSeek's reasoning switch and is ignored for other
        providers. It defaults off so a bounding-box or JSON answer is not lost
        when the model spends its entire output budget on reasoning.
        """
        endpoint = openai_endpoint(self._model)
        if endpoint is None:
            raise ToolInputError(
                "analyze_image needs an OpenAI-compatible endpoint with image input."
            )
        if not self.supported:
            with self._lock:
                detail = self._detail or self._state
            raise ToolInputError(
                "image input is not available for this model: " + detail
            )
        payload = _payload(
            endpoint,
            question,
            _data_uri(data, media_type),
            max_tokens=max(1, int(max_tokens)),
        )
        payload.update(_thinking_payload(endpoint, thinking))
        try:
            response = httpx.post(
                _endpoint_url(endpoint),
                headers=_headers(endpoint),
                json=payload,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a tool failure
            raise ToolInputError(f"vision request failed: {exc}") from exc
        if not 200 <= response.status_code < 300:
            raise ToolInputError(
                f"vision request failed (HTTP {response.status_code}): "
                f"{response.text[:300]}"
            )
        try:
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ToolInputError(f"vision response was not JSON: {exc}") from exc
        text = _extract_text(data)
        if text.strip():
            return text
        choice = (data.get("choices") or [{}])[0]
        finish = choice.get("finish_reason") or "unknown"
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content") or ""
        if finish == "length" and reasoning:
            raise ToolInputError(
                "the vision model spent its whole output budget on reasoning and "
                "returned no answer; disable thinking for direct box/JSON answers "
                "or raise max_tokens"
            )
        raise ToolInputError(
            f"the vision model returned no text (finish_reason={finish})"
        )


def _probe(model: str) -> tuple[str, str]:
    """Return ``(state, detail)`` for one model, with a tiny image request."""
    endpoint = openai_endpoint(model)
    if endpoint is None:
        return "unsupported", "Only OpenAI-compatible models can be probed."
    data_uri = _data_uri(_probe_png(), "image/png")
    payload = _payload(endpoint, "Reply with OK.", data_uri, max_tokens=16)
    payload.update(_thinking_payload(endpoint, False))
    try:
        response = httpx.post(
            _endpoint_url(endpoint),
            headers=_headers(endpoint),
            json=payload,
            timeout=PROBE_TIMEOUT,
        )
    except httpx.TimeoutException:
        return "error", "The endpoint did not answer the image probe in time."
    except Exception as exc:  # noqa: BLE001 - connection refused, DNS, TLS, ...
        return "error", f"The endpoint is unreachable: {exc}"

    status = response.status_code
    text = response.text.lower()
    if 200 <= status < 300:
        return "supported", ""
    if status in (401, 403):
        return "error", f"Image probe rejected the credentials (HTTP {status})."
    if status == 404:
        return "error", "Image probe could not find the model or endpoint (HTTP 404)."
    if status in (400, 415, 422) and any(term in text for term in _IMAGE_TERMS):
        return "unsupported", "The endpoint rejected image content."
    if status == 429:
        return "error", "Image probe was rate-limited; it will be retried later."
    return "error", f"Image probe failed (HTTP {status}): {response.text[:200]}"


def probe_now(model: str) -> dict:
    """Probe synchronously; useful for tests and a manual recheck."""
    state, detail = _probe(model)
    return {"available": state == "supported", "state": state, "detail": detail, "model": model}


__all__ = ["VISION_SERVICE", "VisionService", "probe_now"]
