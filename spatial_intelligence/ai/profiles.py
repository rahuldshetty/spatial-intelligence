"""What a model can actually do here, decided before the run starts.

The wizard has to answer three questions about a checkpoint the user picked:
can this machine run it at all, does it do the task this workflow performs, and
does it already know the labels — or does it need them typed in (a promptable or
zero-shot model)?

Everything is decided from metadata: the shipped catalog, then the Hub's repo
card and file list. When that evidence runs out, the session's own model is asked
for a verdict (injected as ``verdict``, so this module stays below the agent
layer), which is what keeps the wizard instant for known models and still honest
for obscure ones.
"""

from __future__ import annotations

import threading
import urllib.parse
from dataclasses import dataclass, replace

from ..contracts.errors import ToolInputError
from ..pythonruntime import packages
from . import catalog, hub

#: Hub pipeline tags -> the task vocabulary workflows use.
PIPELINE_TASKS = {
    "image-segmentation": "segmentation",
    "mask-generation": "segmentation",
    "object-detection": "detection",
    "zero-shot-object-detection": "detection",
    "image-classification": "classification",
}

#: Tasks a promptable (zero-shot) model can be pointed at.
PROMPT_TASKS = frozenset({"segmentation", "detection"})

#: Weight kinds we can load, with what loading them takes.
ROUTES = ("onnx", "torchgeo", "terratorch", "transformers", "research", "unusable")

_HUB_PREFIX = "hf:"
_lock = threading.Lock()
_cache: dict[tuple[str, str], "Profile"] = {}


@dataclass(frozen=True, slots=True)
class Profile:
    """What we know about one candidate model."""

    ref: str
    repo: str = ""
    revision: str = ""
    route: str = "unusable"
    tasks: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    promptable: bool = False
    runnable: bool = False
    source: str = "heuristic"
    reason: str = ""

    def supports(self, task: str) -> bool:
        """Whether this model can carry out ``task`` (empty task: always)."""
        if not task:
            return True
        if task in self.tasks:
            return True
        return self.promptable and task in PROMPT_TASKS

    def as_dict(self) -> dict:
        """The shape the wizard and the prompt renderer read."""
        return {
            "ref": self.ref,
            "repo": self.repo,
            "revision": self.revision,
            "route": self.route,
            "tasks": list(self.tasks),
            "labels": list(self.labels),
            "promptable": self.promptable,
            "runnable": self.runnable,
            "source": self.source,
            "reason": self.reason,
        }


#: Hosts a repository link may point at.
HUB_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})
#: Path segments that introduce a revision in a repository link.
REVISION_SEGMENTS = ("tree", "blob", "resolve")
#: Path prefixes that are not a model repository.
NON_MODEL_SECTIONS = ("datasets", "spaces")


def repo_from_link(text: str) -> str:
    """Read ``owner/name[@revision]`` out of a model repository link."""
    url = text if "://" in text else f"https://{text}"
    parsed = urllib.parse.urlparse(url)
    host = (parsed.netloc or "").split("@")[-1].split(":")[0].lower()
    if host not in HUB_HOSTS:
        raise ToolInputError(
            f"only Hugging Face repository links are supported, got {host or text!r}"
        )
    parts = [part for part in (parsed.path or "").split("/") if part]
    if parts and parts[0] == "models":
        parts = parts[1:]
    if not parts or parts[0] in NON_MODEL_SECTIONS:
        raise ToolInputError(
            "that link is not a model repository; use https://huggingface.co/owner/name"
        )
    if len(parts) < 2:
        raise ToolInputError(
            "a model link needs an owner and a name, like https://huggingface.co/owner/name"
        )
    owner, name = parts[0], parts[1]
    revision = ""
    if len(parts) >= 4 and parts[2] in REVISION_SEGMENTS:
        revision = parts[3]
    repo = f"{owner}/{name.removesuffix('.git')}"
    return f"{repo}@{revision}" if revision else repo


def parse_ref(value: str) -> tuple[str, str, str]:
    """Split a picker reference into ``(catalog_id, repo, revision)``.

    Accepts ``slimsam-77``, ``hf:owner/name``, ``owner/name``, any of those with
    an ``@revision`` suffix, and a pasted repository link
    (``https://huggingface.co/owner/name``, with ``/tree/<revision>`` if given).
    """
    text = str(value or "").strip()
    if not text:
        raise ToolInputError("choose a model first")
    lowered = text.lower()
    if "://" in text or lowered.startswith(tuple(f"{host}/" for host in HUB_HOSTS)):
        text = repo_from_link(text)
    revision = ""
    if "@" in text:
        text, _, revision = text.partition("@")
    if text.startswith(_HUB_PREFIX):
        text = text[len(_HUB_PREFIX) :]
    if "/" not in text:
        return text, "", revision
    return "", text, revision.strip()


def _installed(root: str) -> bool:
    """Whether the optional sandbox package providing ``root`` is installed."""
    package = packages.package_for(root)
    return bool(package) and packages.installed(package)


def _clean_labels(values) -> tuple[str, ...]:
    """Drop placeholder and duplicate labels, keeping the model's order."""
    seen: list[str] = []
    for value in values or ():
        text = str(value).strip()
        if not text or text.upper() == "N/A" or text in seen:
            continue
        seen.append(text)
    return tuple(seen)


def from_catalog(model_id: str) -> Profile | None:
    """The profile of a shipped model, or ``None`` when the id is not one."""
    for spec in catalog.MODELS:
        if spec.id != model_id:
            continue
        return Profile(
            ref=spec.id,
            repo=spec.repo,
            revision=spec.revision,
            route="onnx",
            tasks=(spec.task,),
            labels=_clean_labels(spec.labels),
            promptable=bool(spec.prompts),
            runnable=True,
            source="catalog",
            reason=(
                f"Built in: {spec.task} on ONNX Runtime, {spec.total_bytes / 1_000_000:.0f} MB, "
                f"{spec.license}. No download until the first run."
            ),
        )
    return None


def _tasks_from(info: dict) -> tuple[str, ...]:
    """Read the tasks a repo declares, from its pipeline tag and its tags."""
    found: list[str] = []
    pipeline = str(info.get("pipeline") or "").lower()
    if pipeline in PIPELINE_TASKS:
        found.append(PIPELINE_TASKS[pipeline])
    tags = {str(tag).lower() for tag in info.get("tags") or []}
    for pipeline_tag, task in PIPELINE_TASKS.items():
        if pipeline_tag in tags and task not in found:
            found.append(task)
    for architecture in info.get("architectures") or []:
        name = str(architecture).lower()
        if "segmentation" in name or "mask" in name or "sam" in name:
            if "segmentation" not in found:
                found.append("segmentation")
        elif "detection" in name:
            if "detection" not in found:
                found.append("detection")
        elif "classification" in name:
            if "classification" not in found:
                found.append("classification")
    return tuple(found)


def _is_promptable(info: dict) -> bool:
    """Whether the model takes prompts (points/boxes/text) instead of fixed labels."""
    tags = {str(tag).lower() for tag in info.get("tags") or []}
    if tags & {"sam", "slimsam", "segment-anything", "mask-generation", "zero-shot", "clip", "grounding"}:
        return True
    pipeline = str(info.get("pipeline") or "").lower()
    return pipeline in ("mask-generation", "zero-shot-object-detection")


def _route_for(info: dict, weights: dict[str, list[str]]) -> tuple[str, bool, str]:
    """Decide how a checkpoint would be run, and whether it can be right now."""
    if not (weights["onnx"] or weights["safetensors"] or weights["torch"]):
        return ("unusable", False, "No model weights were found in that repository.")
    library = str(info.get("library") or "").lower()
    tags = {str(tag).lower() for tag in info.get("tags") or []}
    if "terratorch" in tags or library == "terratorch":
        return (
            "terratorch",
            _installed("terratorch"),
            "TerraTorch checkpoint: the agent reads the checkpoint with skill(\"terratorch\") in run_python.",
        )
    if "torchgeo" in tags or library == "torchgeo":
        return (
            "torchgeo",
            _installed("torchgeo"),
            "TorchGeo checkpoint: the agent loads it in run_python with torchgeo's own model registry.",
        )
    if library in ("transformers", "transformers.js"):
        installed = _installed("transformers")
        return (
            "transformers",
            installed,
            "Hugging Face Transformers checkpoint: the sandbox loads its own "
            "classes and processor and runs it on the CPU."
            if installed
            else "Hugging Face Transformers checkpoint: needs the transformers "
            "library in the sandbox.",
        )
    if weights["safetensors"] or weights["torch"]:
        return (
            "research",
            _installed("torch"),
            "Weights only: the agent fetches them and writes the inference code in run_python.",
        )
    if weights["onnx"]:
        return (
            "onnx",
            False,
            "ONNX weights, but only the built-in models have a wired runner; pick a "
            "catalog model, or a checkpoint the sandbox can load (safetensors/.pt).",
        )
    return ("unusable", False, "No model weights were found in that repository.")


def classify_hub(info: dict, ref: str) -> Profile:
    """Build the profile of a Hub repository from its metadata alone."""
    weights = hub.weight_files(info.get("files") or [])
    route, runnable, route_reason = _route_for(info, weights)
    tasks = _tasks_from(info)
    labels = _clean_labels(info.get("labels"))
    promptable = _is_promptable(info)
    reason = route_reason
    if not runnable:
        package = {"terratorch": "terratorch", "torchgeo": "torchgeo", "transformers": "transformers"}.get(route)
        if package:
            reason = f"{route_reason} Install it with: uv pip install -e \".[geoai]\" (or pip install {package})."
    elif not tasks:
        reason = f"{route_reason} Its task is not declared, so the agent will check it before running."
    return Profile(
        ref=ref,
        repo=info.get("repo") or "",
        revision=info.get("revision") or "",
        route=route,
        tasks=tasks,
        labels=labels,
        promptable=promptable,
        runnable=runnable,
        source="heuristic",
        reason=reason,
    )


def facts_for(info: dict) -> dict:
    """The compact evidence handed to the model verdict."""
    weights = hub.weight_files(info.get("files") or [])
    return {
        "repo": info.get("repo"),
        "pipeline": info.get("pipeline"),
        "library": info.get("library"),
        "architectures": info.get("architectures") or [],
        "tags": (info.get("tags") or [])[:15],
        "weights": {kind: paths[:4] for kind, paths in weights.items() if paths},
        "labels": (info.get("labels") or [])[:20],
    }


def needs_verdict(profile: Profile, task: str) -> bool:
    """Whether the metadata left the question open enough to ask the model."""
    if not profile.tasks:
        return True
    if task and not profile.supports(task):
        return True
    return profile.route == "research" and not profile.labels and not profile.promptable


def merge_verdict(profile: Profile, verdict: dict) -> Profile:
    """Overlay a model verdict on what the metadata said."""
    tasks = profile.tasks
    task = str(verdict.get("task") or "").strip()
    if task and task not in tasks:
        tasks = tasks + (task,)
    if verdict.get("supported") and task and task not in tasks:
        tasks = tasks + (task,)
    labels = _clean_labels(verdict.get("labels")) or profile.labels
    return replace(
        profile,
        tasks=tasks,
        labels=labels,
        promptable=bool(verdict.get("promptable")) or profile.promptable,
        source="agent",
        reason=str(verdict.get("reason") or profile.reason).strip(),
    )


def apply_hint(profile: Profile, hint: dict) -> Profile:
    """Overlay a curated option's known facts on a metadata guess.

    A checkpoint this build has run with carries its own task and route; the Hub
    often does not say (TerraTorch and TorchGeo repos declare no pipeline), and a
    wrong "unknown" would block a model that works.
    """
    if not hint:
        return profile
    return replace(
        profile,
        ref=hint.get("ref") or profile.ref,
        repo=hint.get("repo") or profile.repo,
        revision=hint.get("revision") or profile.revision,
        route=hint.get("route") or profile.route,
        tasks=tuple(hint.get("tasks") or profile.tasks),
        labels=tuple(hint.get("labels") or profile.labels),
        promptable=bool(hint.get("promptable")) or profile.promptable,
        runnable=bool(hint.get("runnable", profile.runnable)),
        source=str(hint.get("source") or profile.source),
        reason=str(hint.get("reason") or profile.reason),
    )


def guess_from_search(row: dict, task: str = "") -> Profile:
    """Describe a search hit from the search response alone.

    The search response has no file list, so nothing here may claim a repo is
    unusable — that verdict needs the weights, and it is made when the user
    actually picks the row. This is the badge next to a result, not the decision.
    """
    info = {
        "repo": row.get("repo") or "",
        "revision": "",
        "pipeline": row.get("pipeline"),
        "library": row.get("library"),
        "tags": row.get("tags") or [],
        "labels": [],
        "files": [],
    }
    tasks = _tasks_from(info)
    promptable = _is_promptable(info)
    library = str(row.get("library") or "").lower()
    tags = {str(tag).lower() for tag in info["tags"]}
    if "terratorch" in tags or library == "terratorch":
        route, runnable = "terratorch", _installed("terratorch")
    elif "torchgeo" in tags or library == "torchgeo":
        route, runnable = "torchgeo", _installed("torchgeo")
    elif library in ("transformers", "transformers.js"):
        route, runnable = "transformers", _installed("transformers")
    else:
        route, runnable = "research", _installed("torch")
    reason = "Weights not inspected yet — selecting this will check them."
    if not runnable:
        reason = f"{route} weights cannot be loaded on this machine yet."
    return Profile(
        ref=row.get("ref") or row.get("repo") or "",
        repo=info["repo"],
        route=route,
        tasks=tasks,
        promptable=promptable,
        runnable=runnable,
        source="heuristic",
        reason=reason,
    )


def resolve(
    ref: str,
    task: str = "",
    verdict=None,
    hint: dict | None = None,
    refresh: bool = False,
) -> Profile:
    """Describe the model ``ref`` names, asking ``verdict`` only when needed.

    ``verdict`` is ``callable(*, ref, facts, task) -> dict | None`` — the agent
    layer supplies one; without it every answer comes from metadata. ``hint`` is
    a curated option's metadata, which outranks the Hub's own claims about a
    checkpoint this build already knows how to run.
    """
    model_id, repo, revision = parse_ref(ref)
    if model_id:
        profile = from_catalog(model_id)
        if profile is None:
            known = ", ".join(spec.id for spec in catalog.MODELS)
            raise ToolInputError(f"unknown built-in model {model_id!r}; known: {known}")
        return profile

    key = (f"{repo}@{revision}", task)
    with _lock:
        cached = None if refresh else _cache.get(key)
    if cached is not None:
        return cached

    try:
        info = hub.repo_info(repo)
    except ToolInputError:
        if not hint:
            raise
        info = {"repo": repo, "revision": revision}
    if revision:
        info["revision"] = revision
    profile = classify_hub(info, ref=f"hf:{repo}" + (f"@{revision}" if revision else ""))
    profile = apply_hint(profile, hint or {})
    if verdict is not None and needs_verdict(profile, task):
        try:
            judged = verdict(ref=profile.ref, facts=facts_for(info), task=task)
        except Exception:
            judged = None
        if isinstance(judged, dict):
            profile = merge_verdict(profile, judged)
    with _lock:
        _cache[key] = profile
    return profile


def clear_cache() -> None:
    """Forget every resolved profile (used by the tests)."""
    with _lock:
        _cache.clear()


__all__ = [
    "PIPELINE_TASKS",
    "PROMPT_TASKS",
    "ROUTES",
    "Profile",
    "apply_hint",
    "classify_hub",
    "clear_cache",
    "facts_for",
    "from_catalog",
    "guess_from_search",
    "merge_verdict",
    "needs_verdict",
    "parse_ref",
    "resolve",
]
