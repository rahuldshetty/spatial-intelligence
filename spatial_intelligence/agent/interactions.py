"""The structured user-interaction protocol.

A tool raises ``CallDeferred`` with an ``interaction`` form; pydantic-ai ends
the run with ``DeferredToolRequests``; the browser submits answers; the run
resumes with ``DeferredToolResults``. These three functions own the shape of
that exchange, so the runner, the session, and the notebook recorder agree.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import DeferredToolRequests

from ..contracts.errors import ToolInputError
from ..contracts.ids import new_id


def interaction_from_deferred(requests: DeferredToolRequests) -> dict:
    """Normalize one deferred request into the browser interaction schema."""
    if not requests.calls:
        raise RuntimeError("agent requested approval without an external interaction")
    call = requests.calls[0]
    metadata = requests.metadata.get(call.tool_call_id, {})
    form = metadata.get("interaction")
    if not isinstance(form, dict):
        # A deferred call that carried no form (an approval request) still has
        # to render, so fall back to the call's own arguments.
        args = call.args if isinstance(call.args, dict) else {}
        form = {
            "title": args.get("title", "Input required"),
            "prompt": args.get("prompt", ""),
            "fields": args.get("fields", []),
            "submit_label": args.get("submit_label", "Continue"),
            "allow_cancel": args.get("allow_cancel", True),
        }
    return {"id": new_id(), "tool_call_id": call.tool_call_id, **form}


def validate_answers(interaction: dict, answers: dict) -> None:
    """Raise when submitted answers do not match the pending form.

    Checks unknown fields, missing required fields, and values outside a
    radio/multi-select option set, so a stale or hand-crafted request from the
    browser cannot resume a run with garbage.
    """
    field_ids = {field.get("id") for field in interaction.get("fields", [])}
    unknown = set(answers) - field_ids
    if unknown:
        raise ToolInputError(f"unknown interaction fields: {sorted(unknown)}")

    missing = {
        field.get("id")
        for field in interaction.get("fields", [])
        if field.get("required", True)
        and field.get("id") not in answers
        and field.get("default") is None
    }
    if missing:
        raise ToolInputError(f"missing required fields: {sorted(missing)}")

    for field in interaction.get("fields", []):
        if field.get("type") not in {"radio", "multi_select"}:
            continue
        allowed = {option.get("value") for option in field.get("options", [])}
        answer = answers.get(field.get("id"), field.get("default"))
        selected = answer if isinstance(answer, list) else [answer]
        invalid = {
            value for value in selected if value is not None and value not in allowed
        }
        if invalid:
            raise ToolInputError(
                f"invalid value for {field.get('label', field.get('id'))}: "
                f"{sorted(invalid)}"
            )


_AOI_MODES = {
    "view": "Map view",
    "drawn": "Drawn shape",
    "bbox": "Bounding box",
    "place": "Place",
}


def _aoi_text(value: object) -> str:
    """Render the composite AOI answer without dumping a whole geometry."""
    if not isinstance(value, dict):
        return str(value)
    mode = str(value.get("mode") or "")
    parts = [_AOI_MODES.get(mode, mode or "Area of interest")]
    label = value.get("label")
    if label:
        parts.append(str(label))
    bbox = value.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            parts.append("bbox [" + ", ".join(f"{float(item):g}" for item in bbox) + "]")
        except (TypeError, ValueError):
            parts.append("bbox " + str(list(bbox)))
    elif isinstance(value.get("geojson"), dict):
        features = (value["geojson"] or {}).get("features")
        if isinstance(features, list):
            parts.append(f"{len(features)} feature(s)")
    return " — ".join(parts)


def _answer_item(field: dict, item: object, labels: dict) -> str:
    if isinstance(item, (str, int, float, bool)):
        return str(labels.get(item, item))
    return json.dumps(item, ensure_ascii=False, default=str)


def _answer_text(field: dict, value: object) -> str:
    if value is None:
        return "none"
    if field.get("type") == "aoi":
        return _aoi_text(value)
    labels = (
        {
            option.get("value"): option.get("label", option.get("value"))
            for option in field.get("options", [])
        }
        if field.get("type") in {"radio", "multi_select"}
        else {}
    )
    values = value if isinstance(value, list) else [value]
    return ", ".join(_answer_item(field, item, labels) for item in values)


def answers_markdown(interaction: dict, answers: dict) -> str:
    """Render submitted answers as the Markdown cell recorded in the notebook."""
    fields = {field.get("id"): field for field in interaction.get("fields", [])}
    lines = ["### Input provided"]
    for key, value in answers.items():
        field = fields.get(key) or {}
        rendered = _answer_text(field, value)
        lines.append(f"- **{field.get('label', key)}:** {rendered}")
    return "\n".join(lines)


def pending_interaction(cell: dict) -> dict | None:
    """Return the interaction a cell is currently waiting on, if any."""
    interaction: Any = cell.get("interaction")
    if cell.get("status") != "waiting_for_input" or not isinstance(interaction, dict):
        return None
    return interaction
