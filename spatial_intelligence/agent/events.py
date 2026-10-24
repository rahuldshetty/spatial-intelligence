"""Turning pydantic-ai stream events into the trace steps the UI renders.

The step vocabulary is the wire contract between the runner and the browser:
every step is JSON-safe and is appended to the cell's trace (and its trace
file) exactly once.
"""

from __future__ import annotations

from typing import Any


def json_safe(value: object) -> object:
    """Coerce a tool argument or result into a JSON-serializable structure.

    Preserves dict/list shape (so the UI can pretty-print JSON and detect
    code-bearing tools) while falling back to ``str`` for anything exotic.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def step_from_event(event: object) -> dict[str, Any] | None:
    """Map one pydantic-ai stream event to a trace step, or ``None``.

    Text deltas are emitted as ``text_delta`` so the browser can append them
    without re-parsing the whole message.
    """
    kind = getattr(event, "event_kind", None)
    if kind == "function_tool_call":
        part = event.part
        return {
            "type": "tool_call",
            "name": part.tool_name,
            "args": json_safe(part.args),
            "tool_call_id": part.tool_call_id,
        }
    if kind == "function_tool_result":
        part = event.part
        return {
            "type": "tool_result",
            "name": getattr(part, "tool_name", None),
            "content": json_safe(getattr(part, "content", None)),
            "outcome": getattr(part, "outcome", None),
            "tool_call_id": event.tool_call_id,
        }
    if kind == "part_start":
        part = event.part
        part_kind = getattr(part, "part_kind", None)
        if part_kind in ("text", "thinking"):
            content = getattr(part, "content", "") or ""
            if content:
                step = {"type": "text", "content": content}
                if part_kind == "thinking":
                    # The UI renders both as text, but the conversation index
                    # must not store model reasoning. Keep the distinction.
                    step["reasoning"] = True
                return step
    elif kind == "part_delta":
        delta = event.delta
        delta_kind = getattr(delta, "part_delta_kind", None)
        if delta_kind in ("text", "thinking"):
            content = getattr(delta, "content_delta", "") or ""
            if content:
                step = {"type": "text_delta", "content": content}
                if delta_kind == "thinking":
                    step["reasoning"] = True
                return step
    return None
