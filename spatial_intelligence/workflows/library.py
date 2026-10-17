"""The shipped workflows, in one place: a wizard's fields plus its prompt.

Adding a workflow is writing one :class:`WorkflowSpec` and appending it to
:data:`WORKFLOWS` — the wizard, the API, the seeding, and the prompting are
generic. Nothing here executes anything.

Model choices are built from what this machine can actually run: the built-in
ONNX catalog first, then the sandbox routes (torchgeo/terratorch) only when the
optional extra is installed, so the wizard never offers a route that would fail
on a lean install.
"""

from __future__ import annotations

from ..ai import catalog, profiles
from ..contracts.errors import ToolInputError
from ..contracts.forms import ChoiceOption, InteractionField
from ..pythonruntime import packages
from .spec import WorkflowSpec

SEGMENTATION_PROMPT = """\
Segment what the model below can find in this workspace and show the result on the map.

Model: {model}
Area of interest: {aoi}
Input data: {source}

Work with the tools you have: if no input file is given, search the catalogs for a
scene over the AOI, download it into data/ (add_catalog_scene), and clip it to the
AOI. Then run the model above the way its route describes, add_geojson the result
and fit_bounds the AOI. Keep outputs under results/ and report their paths.

Analysis goal: {notes}
"""

DETECTION_PROMPT = """\
Detect what the model below is trained to detect in this workspace and show the results on the map.

Model: {model}
Area of interest: {aoi}
Input data: {source}

Work with the tools you have: if no input file is given, find and download a scene
over the AOI into data/. A detector needs about a metre per pixel, so prefer
aerial or drone imagery (Vantor, OpenAerialMap) over 10 m satellite pixels: clip to
the AOI, run the model above the way its route describes, then add_geojson the
result and fit_bounds the AOI. Keep outputs under results/ and report their paths.

Analysis goal: {notes}
"""

CHANGE_PROMPT = """\
Measure what changed over the area of interest and show it on the map.

Area of interest: {aoi}
Analysis goal: {notes}

Work with the tools you have: choose two comparable dates over the AOI. When the
goal names dates, sensors, an index, or a threshold, use them; otherwise choose the
most recent usable pair for the area. Search the catalogs, download both scenes
into data/, clip both to the AOI, and compute the change with run_python (band math
over the aligned rasters) when no tool covers it. Polygonize the changed area, then
add_geojson the result and fit_bounds the AOI. Keep outputs under results/ and
report their paths, the two dates, and the index or threshold you used.
"""


def _installed(*keys: str) -> bool:
    """Whether every named optional sandbox package is installed here."""
    present = {package.key for package in packages.installed_packages()}
    return all(key in present for key in keys)


def _catalog_option(model_id: str) -> ChoiceOption:
    """A shipped model as a picker option, carrying its profile."""
    profile = profiles.from_catalog(model_id)
    assert profile is not None  # ids come from the catalog itself
    return ChoiceOption(
        value=profile.ref,
        label=f"{model_id} — built in",
        description=profile.reason,
        recommended=True,
        metadata=profile.as_dict(),
    )


def _sandbox_option(
    *, ref: str, label: str, repo: str, route: str, task: str, description: str
) -> ChoiceOption:
    """A curated checkpoint the sandbox runs, with what we know about it.

    The task is declared here rather than guessed from the Hub: these are the
    checkpoints this build has actually been run with, and a foundation model
    whose repo declares no task must not be reported as unable to do one.
    """
    return ChoiceOption(
        value=ref,
        label=label,
        description=description,
        metadata={
            "ref": ref,
            "repo": repo,
            "revision": "",
            "route": route,
            "tasks": [task],
            "labels": [],
            "promptable": False,
            "runnable": _installed(route),
            "source": "curated",
            "reason": description,
        },
    )


def _catalog_options(task: str) -> list[ChoiceOption]:
    """The shipped models that do a task, as picker options."""
    return [_catalog_option(model.id) for model in catalog.iter_specs(task)]


def _segmentation_models() -> list[ChoiceOption]:
    """Shipped segmentation models, then the sandbox routes when they can run."""
    options = _catalog_options("segmentation")
    if _installed("torchgeo"):
        options.append(
            _sandbox_option(
                ref="hf:torchgeo/ftw",
                label="Fields of the World — torchgeo",
                repo="torchgeo/ftw",
                route="torchgeo",
                task="segmentation",
                description=(
                    "4-band Sentinel-2 U-Net for agricultural field boundaries "
                    "(2- or 3-class), ~53 MB. The agent fetches the weights and "
                    "writes the inference code."
                ),
            )
        )
    if _installed("terratorch", "torch"):
        options.append(
            _sandbox_option(
                ref="hf:ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
                label="Prithvi-EO-2.0 — terratorch",
                repo="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
                route="terratorch",
                task="segmentation",
                description=(
                    "IBM/NASA multispectral foundation model (300M). Needs a task "
                    "head or fine-tuned weights; the agent loads it with "
                    "skill(\"terratorch\") and writes the inference code."
                ),
            )
        )
    return options


def _detection_models() -> list[ChoiceOption]:
    """Shipped detection models. No sandbox route: the ONNX runner is the one."""
    return _catalog_options("detection")


def _model_field(task: str, options: list[ChoiceOption], default: str = "") -> InteractionField:
    """The picker a workflow uses to choose what runs its analysis."""
    return InteractionField(
        id="model",
        label="Model",
        type="model",
        task=task,
        options=options,
        default=default or (options[0].value if options else ""),
        description=(
            "Pick a model this build ships with, or search Hugging Face for a "
            "checkpoint. You will be told up front whether it can do this task."
        ),
    )


def option_hint(ref: str) -> dict | None:
    """The curated metadata for a picker option, if one declares it.

    The profile endpoint uses this so a curated checkpoint keeps the task this
    build knows it does, instead of whatever the Hub's metadata implies. A
    pasted link is matched by its repository, so
    ``https://huggingface.co/torchgeo/ftw`` finds the Fields of the World entry.
    """
    wanted = str(ref or "").strip()
    if not wanted:
        return None
    repo = ""
    try:
        _, repo, _ = profiles.parse_ref(wanted)
    except ToolInputError:
        repo = ""
    for spec in WORKFLOWS:
        for field in spec.params:
            if field.type != "model":
                continue
            for option in field.options:
                metadata = option.metadata
                if option.value == wanted or metadata.get("ref") == wanted:
                    return dict(metadata)
                if repo and metadata.get("repo") == repo:
                    return dict(metadata)
    return None


AOI_FIELD = InteractionField(
    id="aoi",
    label="Area of interest",
    type="aoi",
    description=(
        "Draw it with the map's GeoEditor, take the current map view, type a place "
        "name, or paste a bounding box as west, south, east, north."
    ),
)

NOTES_FIELD = InteractionField(
    id="notes",
    label="Analysis goal",
    type="textarea",
    required=False,
    description=(
        "Describe what you want to learn or produce. Dates, sensors, an index, a "
        "threshold, or a custom prompt all belong here; the agent follows it. Optional."
    ),
    placeholder=(
        "e.g. Map new construction between June 2024 and June 2025 using "
        "Sentinel-2, then measure the changed area inside the AOI."
    ),
)


WORKFLOWS: tuple[WorkflowSpec, ...] = (
    WorkflowSpec(
        id="blank",
        label="Blank",
        summary="An empty workspace: write your own prompt in the Cells tab.",
    ),
    WorkflowSpec(
        id="segmentation",
        label="Segmentation",
        summary="Segment features in imagery with a chosen model and AOI.",
        params=(
            _model_field("segmentation", _segmentation_models()),
            AOI_FIELD,
            InteractionField(
                id="source",
                label="Input data",
                type="text",
                required=False,
                description=(
                    "Accepted formats: a workspace path (data/…), an https URL, or a "
                    "STAC scene key. Leave empty to search the catalogs for the AOI."
                ),
                placeholder="path in data/, a URL, or a scene key — leave empty to search the catalogs",
            ),
            NOTES_FIELD,
        ),
        prompt=SEGMENTATION_PROMPT,
        requires=(
            "segment_image",
            "add_catalog_scene",
            "clip",
            "add_geojson",
            "fit_bounds",
            "search_stac_scenes",
        ),
    ),
    WorkflowSpec(
        id="detection",
        label="Detection",
        summary="Detect known objects (cars, boats, buildings) in high-resolution imagery.",
        params=(
            _model_field("detection", _detection_models()),
            AOI_FIELD,
            InteractionField(
                id="source",
                label="Input data",
                type="text",
                required=False,
                description=(
                    "Accepted formats: a workspace path (data/…), an https URL, or a "
                    "STAC scene key. Leave empty to search the catalogs for the AOI."
                ),
                placeholder="path in data/, a URL, or a scene key — leave empty to search the catalogs",
            ),
            NOTES_FIELD,
        ),
        prompt=DETECTION_PROMPT,
        requires=(
            "detect_objects",
            "search_vantor_imagery",
            "search_openaerialmap",
            "add_catalog_scene",
            "clip",
            "add_geojson",
            "fit_bounds",
        ),
    ),
    WorkflowSpec(
        id="change-detection",
        label="Change detection",
        summary="Compare two dates over an AOI and map what changed.",
        params=(AOI_FIELD, NOTES_FIELD),
        prompt=CHANGE_PROMPT,
        requires=(
            "search_stac_scenes",
            "add_catalog_scene",
            "clip",
            "polygonize",
            "add_geojson",
            "fit_bounds",
        ),
    ),
)

#: Lookup by id, built once from the manifest.
BY_ID: dict[str, WorkflowSpec] = {spec.id: spec for spec in WORKFLOWS}


def get(workflow_id: str) -> WorkflowSpec:
    """Return the workflow with this id, naming the known ids otherwise."""
    spec = BY_ID.get(workflow_id or "")
    if spec is None:
        known = ", ".join(BY_ID)
        raise KeyError(f"unknown workflow {workflow_id!r}; known: {known}")
    return spec


__all__ = ["BY_ID", "WORKFLOWS", "get"]
