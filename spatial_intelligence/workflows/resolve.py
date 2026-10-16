"""Wizard answers -> workspace files and one prompt.

The wizard collects strings; the agent needs a prompt and, for an AOI, a file it
can clip to. This module is that boundary: it validates the answers, writes
``data/aoi.geojson``, renders the workflow's template, and returns the cells the
notebook should hold.
"""

from __future__ import annotations

import json

from ..ai import profiles
from ..contracts.errors import ToolInputError
from ..contracts.forms import InteractionField
from ..geo import places
from ..workspace.store import Workspace, now_iso
from ..workspace import files as fileops
from . import library
from .spec import WorkflowSpec

#: Where a resolved area of interest lands, so tools and the agent can read it.
AOI_PATH = "data/aoi.geojson"
#: What an empty optional answer renders as in the prompt.
EMPTY = "none"
#: AOI modes the wizard may send.
AOI_MODES = ("view", "drawn", "bbox", "place")


def parse_bbox(value: object) -> list[float]:
    """Parse a bounding box from a list or a ``w,s,e,n`` string."""
    numbers: list[float]
    if isinstance(value, str):
        cleaned = value.replace(",", " ").split()
        if len(cleaned) != 4:
            raise ToolInputError(
                "bounding box must be four numbers: west, south, east, north"
            )
        try:
            numbers = [float(part) for part in cleaned]
        except ValueError as exc:
            raise ToolInputError(f"bounding box is not numeric: {value!r}") from exc
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            numbers = [float(part) for part in value]
        except (TypeError, ValueError) as exc:
            raise ToolInputError(f"bounding box is not numeric: {value!r}") from exc
    else:
        raise ToolInputError("bounding box must be four numbers: west, south, east, north")
    west, south, east, north = numbers
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ToolInputError("bounding box longitudes must be between -180 and 180")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ToolInputError("bounding box latitudes must be between -90 and 90")
    if west > east or south > north:
        raise ToolInputError(
            f"bounding box is inverted: expected west <= east and south <= north, got "
            f"[{west:g}, {south:g}, {east:g}, {north:g}]"
        )
    return numbers


def _geometry_nodes(geojson: dict):
    """Yield every geometry object inside a GeoJSON value."""
    kind = geojson.get("type")
    if kind == "FeatureCollection":
        for feature in geojson.get("features") or []:
            if isinstance(feature, dict):
                geometry = feature.get("geometry")
                if isinstance(geometry, dict):
                    yield geometry
    elif kind == "Feature":
        geometry = geojson.get("geometry")
        if isinstance(geometry, dict):
            yield geometry
    elif kind:
        yield geojson


def _positions(node: object):
    """Yield every ``[lon, lat]`` position found in nested coordinates."""
    if isinstance(node, (list, tuple)):
        if len(node) >= 2 and all(isinstance(value, (int, float)) for value in node[:2]):
            yield float(node[0]), float(node[1])
        else:
            for item in node:
                yield from _positions(item)


def bbox_of_geojson(geojson: dict) -> list[float]:
    """The bounding box of a GeoJSON value, as ``[west, south, east, north]``."""
    lons: list[float] = []
    lats: list[float] = []
    for geometry in _geometry_nodes(geojson):
        for lon, lat in _positions(geometry.get("coordinates")):
            lons.append(lon)
            lats.append(lat)
    if not lons:
        raise ToolInputError("the drawn area of interest has no coordinates")
    return [min(lons), min(lats), max(lons), max(lats)]


def aoi_answer(value: object) -> dict:
    """Normalize the wizard's AOI value into ``{mode, bbox, geojson, label}``.

    A place name is resolved here (the server owns geocoding, so the wizard and
    the agent get the same answer), and a bare string is accepted as a bounding
    box when it parses as one.
    """
    if isinstance(value, str) and value.strip():
        try:
            value = {"mode": "bbox", "bbox": parse_bbox(value)}
        except ToolInputError:
            value = {"mode": "place", "label": value.strip()}
    if not isinstance(value, dict):
        raise ToolInputError("area of interest must be an object or a bounding box")

    mode = str(value.get("mode") or "").strip() or "bbox"
    if mode not in AOI_MODES:
        raise ToolInputError(f"unknown area-of-interest mode {mode!r}")
    label = value.get("label")
    label = str(label).strip() if label else None
    geojson = value.get("geojson")
    if geojson is not None and not isinstance(geojson, dict):
        raise ToolInputError("area-of-interest geometry must be a GeoJSON object")

    bbox = value.get("bbox")
    if bbox in (None, "", []):
        if mode == "place":
            if not label:
                raise ToolInputError("a place name is required for that area of interest")
            found = places.geocode(label)
            bbox = found["bbox"]
            label = found.get("label") or label
        elif isinstance(geojson, dict):
            # A drawn shape is the answer; the box is derived from it rather than
            # asked for twice.
            bbox = bbox_of_geojson(geojson)
        else:
            raise ToolInputError(
                f"area of interest in {mode!r} mode needs a bounding box"
            )
    return {
        "mode": mode,
        "bbox": parse_bbox(bbox),
        "geojson": geojson,
        "label": label,
    }


def aoi_geometry(answer: dict) -> dict:
    """The AOI as a FeatureCollection: what was drawn, or its bounding rectangle."""
    geojson = answer.get("geojson")
    if isinstance(geojson, dict):
        kind = geojson.get("type")
        if kind == "FeatureCollection":
            return geojson
        if kind == "Feature":
            return {"type": "FeatureCollection", "features": [geojson]}
        if kind:
            return {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": geojson}],
            }
        raise ToolInputError("area-of-interest geometry has no GeoJSON type")
    west, south, east, north = answer["bbox"]
    ring = [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"source": answer.get("mode") or "bbox"},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }


def option_label(field: InteractionField, value: str) -> str:
    """Render one chosen option as the model should read it."""
    for option in field.options:
        if option.value == value:
            repo = option.metadata.get("repo")
            suffix = f" (repo {repo})" if repo else ""
            return f"{option.label} [{value}]{suffix}"
    return value


#: The analysis tool that runs a shipped model, by task.
TASK_TOOLS = {"segmentation": "segment_image", "detection": "detect_objects"}

#: How a checkpoint gets loaded, per route. The workflow's prompt says what the
#: analysis is; this line says how this particular model runs, because only the
#: model knows whether it is a built-in runner, a sandbox library, or bare
#: weights the agent has to wire up itself.
ROUTE_RUNNERS = {
    "onnx": 'run {tool}(model_id="{ref}").',
    "torchgeo": "fetch it with ai_fetch_model, then load it in run_python with "
    "torchgeo (skill(\"torchgeo\")) and write the inference code.",
    "terratorch": "fetch it with ai_fetch_model, then load it in run_python with "
    "terratorch (skill(\"terratorch\")) and write the inference code.",
    "transformers": "fetch it with ai_fetch_model, then load it in run_python with "
    "transformers (skill(\"transformers\")) and write the inference code.",
    "research": "fetch it with ai_fetch_model, then load it in run_python with torch "
    "and write the inference code.",
}


def runner_line(profile, task: str | None, ref: str) -> str:
    """Say how this checkpoint is run, in the terms the agent can act on."""
    template = ROUTE_RUNNERS.get(profile.route)
    if not template:
        return ""
    tool = TASK_TOOLS.get(task or "", "the analysis tool")
    return "How to run it: " + template.format(tool=tool, ref=ref)


def model_label(field: InteractionField, ref: str) -> str:
    """The picker's name for a chosen model, falling back to the reference."""
    for option in field.options:
        if option.value == ref or option.metadata.get("ref") == ref:
            return option.label
    return ref


def model_value(field: InteractionField, value: object, verdict=None) -> str:
    """Resolve a chosen model into the block a prompt carries.

    The decision this makes is the point of the feature: a model that cannot do
    this workflow's task — or cannot run on this machine — is refused here, with
    the profile's own sentence as the error, so the wizard can say it before the
    run starts.
    """
    raw = value if isinstance(value, dict) else {"ref": value}
    ref = str(raw.get("ref") or "").strip()
    profile = profiles.resolve(ref, field.task or "", verdict=verdict, hint=library.option_hint(ref))
    if not profile.supports(field.task or ""):
        raise ToolInputError(
            f"{profile.repo or ref} cannot do {field.task}: {profile.reason}"
        )
    if not profile.runnable:
        raise ToolInputError(
            f"{profile.repo or ref} cannot run here yet: {profile.reason}"
        )
    name = model_label(field, ref)
    if name == ref and profile.repo:
        # A pasted link reads as the repository it points at, not the URL.
        name = profile.repo
    head = f"{name} [{profile.ref or ref}]"
    details = [part for part in (f"repo {profile.repo}" if profile.repo else "", f"route {profile.route}") if part]
    lines = [f"{head} ({'; '.join(details)})"]

    labels = [str(label).strip() for label in (raw.get("labels") or []) if str(label).strip()]
    target = " ".join(str(raw.get("target") or "").split())
    if labels:
        lines.append("Labels to report: " + ", ".join(labels))
    elif target:
        lines.append("What to find: " + target)
    elif profile.promptable:
        lines.append("What to find: everything the model can segment on its own")
    else:
        lines.append("What to find: every class this model predicts")
    runner = runner_line(profile, field.task, ref)
    if runner:
        lines.append(runner)
    return "\n".join(lines)


def render_value(field: InteractionField, value: object, verdict=None) -> str:
    """Render one answer as the text a prompt interpolates."""
    if field.type == "model":
        return model_value(field, value, verdict)
    if field.type == "aoi":
        answer = aoi_answer(value)
        west, south, east, north = answer["bbox"]
        text = (
            f"bbox [{west:g}, {south:g}, {east:g}, {north:g}] "
            f"(west, south, east, north in degrees); outline at {AOI_PATH}"
        )
        if answer["label"]:
            text += f"; user's label: {answer['label']}"
        if answer["mode"] == "drawn":
            text += "; drawn by the user, use it as given"
        return text
    if field.type == "confirmation":
        return "yes" if value else "no"
    if field.type == "multi_select":
        values = value if isinstance(value, list) else ([value] if value else [])
        labels = [option_label(field, str(item)) for item in values if item not in (None, "")]
        return ", ".join(labels) or EMPTY
    if field.type == "radio":
        chosen = "" if value in (None, "") else str(value)
        return option_label(field, chosen) if chosen else EMPTY
    text = " ".join(str(value).split()) if value not in (None, "") else ""
    return text or EMPTY


def answers_for(spec: WorkflowSpec, params: dict) -> dict[str, object]:
    """Validate the submitted answers and fill in every declared field."""
    submitted = params if isinstance(params, dict) else {}
    answers: dict[str, object] = {}
    for field in spec.params:
        value = submitted.get(field.id)
        if field.required and value in (None, "", [], {}):
            raise ToolInputError(f"{field.label} is required")
        answers[field.id] = value
    return answers


def prompt_text(spec: WorkflowSpec, params: dict, verdict=None) -> str:
    """Render the workflow's prompt from the submitted answers."""
    if not spec.prompt:
        raise ToolInputError(f"workflow {spec.id!r} has no prompt to run")
    answers = answers_for(spec, params)
    rendered = {
        field.id: render_value(field, answers.get(field.id), verdict)
        for field in spec.params
    }
    return spec.prompt.format_map(rendered).strip()


def _cell_line(text: object) -> str:
    """One table cell: no newlines, no column separators."""
    collapsed = " ".join(str(text).split())
    return collapsed.replace("|", "\\|") or EMPTY


def markdown_record(spec: WorkflowSpec, params: dict, verdict=None) -> str:
    """The human-readable record of what this workspace was set up to do."""
    answers = answers_for(spec, params)
    lines = [f"### {spec.label}", "", spec.summary, "", "| Field | Value |", "| --- | --- |"]
    for field in spec.params:
        rendered = render_value(field, answers.get(field.id), verdict).replace("\n", " — ")
        lines.append(
            f"| {_cell_line(field.label)} | {_cell_line(rendered)} |"
        )
    lines += [
        "",
        "The prompt cell below runs this workflow. Edit it or re-run it as often as you like; "
        "the workflow's answers stay in this cell.",
    ]
    return "\n".join(lines)


def workflow_metadata(spec: WorkflowSpec, params: dict, aoi: dict | None) -> dict:
    """The provenance block stored on the seeded cells."""
    answers = answers_for(spec, params)
    stored = {
        field.id: (
            None
            if field.type == "aoi"
            else render_value(field, answers.get(field.id)).replace("\n", "; ")
        )
        for field in spec.params
    }
    return {
        "id": spec.id,
        "label": spec.label,
        "summary": spec.summary,
        "answers": {key: value for key, value in stored.items() if value is not None},
        "aoi": (
            {key: value for key, value in aoi.items() if key != "geojson"} if aoi else None
        ),
        "created_at": now_iso(),
    }


def seed(
    workspace: Workspace, notebook, spec: WorkflowSpec, params: dict, verdict=None
) -> list[dict]:
    """Create the workflow's cells in ``notebook`` and return them.

    The AOI is written as a file before the cells exist, so the prompt can point
    at a path that is already there when the agent starts. ``verdict`` is the
    session's model classifier, used only when a checkpoint's metadata leaves
    its task open.
    """
    answers = answers_for(spec, params)
    aoi = None
    for field in spec.params:
        if field.type == "aoi" and answers.get(field.id) not in (None, "", {}):
            aoi = aoi_answer(answers[field.id])
            fileops.write_text(
                workspace, AOI_PATH, json.dumps(aoi_geometry(aoi), indent=1)
            )
            answers[field.id] = aoi
            break
    metadata = {"geoai": {"workflow": workflow_metadata(spec, dict(answers), aoi)}}
    cells = [
        notebook.add(
            "markdown",
            markdown_record(spec, dict(answers), verdict),
            metadata=metadata,
        )
    ]
    cells.append(
        notebook.add(
            "prompt", prompt_text(spec, dict(answers), verdict), metadata=metadata
        )
    )
    return cells


__all__ = [
    "AOI_PATH",
    "aoi_answer",
    "aoi_geometry",
    "markdown_record",
    "parse_bbox",
    "prompt_text",
    "render_value",
    "seed",
]
