"""Which models this build can run, and what each one needs to be fed.

Data, not behaviour: adding a model is an entry here plus a pipeline for its
architecture, never a change to the store or the manager. Each entry pins the
upstream revision — HF revisions are immutable, so a pinned one cannot change
under a verified download — and every file by sha256 and size.

Tasks are what the catalog is organised around, because they are what the tools
are: a ``segmentation`` model takes prompts and returns masks, a ``detection``
model takes an image and returns labelled boxes. The extra fields carry the
answer to "what does this architecture need": ``canvas`` for the SAM family,
``shortest_edge``/``longest_edge`` for the DETR family, ``labels`` for a
classifier or detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..contracts.errors import ToolInputError


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One file of a model revision, pinned by hash and size."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A model this build knows how to run.

    ``task`` and ``prompts`` are read by the agent-facing tools (what the model
    is for, what it accepts); ``canvas`` is the square the architecture expects
    — SAM-family models embed a ``canvas/16`` grid and emit ``canvas/4`` masks,
    so both the pipeline and the prompt geometry derive from it.

    A detection model instead resizes by edge: ``shortest_edge`` is what the
    shorter side is scaled to and ``longest_edge`` caps the longer one, which is
    the reference preprocessing for the DETR family. ``labels`` are the class
    names in model order, so a detection result can be named without carrying a
    second table beside the catalog.
    """

    id: str
    repo: str
    revision: str
    files: tuple[ModelFile, ...]
    task: str = "segmentation"
    prompts: tuple[str, ...] = ("point", "box", "grid")
    canvas: int = 1024
    license: str = ""
    gated: bool = False
    notes: str = ""
    labels: tuple[str, ...] = ()
    shortest_edge: int = 0
    longest_edge: int = 0

    @property
    def total_bytes(self) -> int:
        """Bytes the revision occupies once downloaded."""
        return sum(file.size for file in self.files)


#: SlimSAM-77 uniform (transformers.js ONNX export, apache-2.0): a pruned
#: ViT-Tiny SAM. ~40 MB, CPU-friendly, and promptable the same way the full SAM
#: is, which is why it is the default for a first run.
SLIMSAM = ModelSpec(
    id="slimsam-77",
    repo="Xenova/slimsam-77-uniform",
    revision="69c9d2e880cd421621781e9ded1f0bf1c20e1f74",
    files=(
        ModelFile(
            path="onnx/vision_encoder.onnx",
            sha256="9f8433273a6750b587779baa0cf5508111001bf7e7acfcf585d370139fd366d0",
            size=23_276_014,
        ),
        ModelFile(
            path="onnx/prompt_encoder_mask_decoder.onnx",
            sha256="f4514391764fbd56e08e119060d874ecd7d52994bfb1968af159e12d4943b5bb",
            size=16_557_892,
        ),
    ),
    license="apache-2.0",
    notes="Pruned SAM (ViT-Tiny), ONNX, fp32. The default: fastest load, smallest download.",
)

#: The 91 class slots the detection model predicts, in output order, taken from
#: the model's own ``id2label``. The eleven COCO ids it was not trained on are
#: ``"N/A"`` — the reference table keeps the numbering rather than compacting it,
#: and a detection that lands on a padding slot is dropped. The graph's 92nd
#: output is the reserved "no object" slot, which the decode slices off.
COCO_LABELS: tuple[str, ...] = (
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

#: YOLOS-tiny: a DETR-family detector with a ViT-Tiny backbone, exported to ONNX
#: for transformers.js. 26 MB, COCO's 80 classes, and the smallest detector that
#: is honest to ship as the default — the useful classes on aerial imagery are
#: the vehicle and vessel ones (car, truck, boat, airplane), not the COCO indoor
#: set, which is why the tool docstring says so plainly.
YOLOS_TINY = ModelSpec(
    id="yolos-tiny",
    repo="Xenova/yolos-tiny",
    revision="e2f9c7673f0fa61849efe2b56a0d7774779ebb9d",
    files=(
        ModelFile(
            path="onnx/model.onnx",
            sha256="0dbdd37573fb6b6ed836aa980ab729c81b659deec900799f37b6a09a57762e88",
            size=26_227_993,
        ),
    ),
    task="detection",
    prompts=(),
    canvas=0,
    labels=COCO_LABELS,
    shortest_edge=512,
    longest_edge=1333,
    license="apache-2.0",
    notes=(
        "COCO-80 detector (person, car, truck, boat, airplane, ...). 100 queries, "
        "no text prompts. Weak on 10 m satellite pixels: best on aerial photos."
    ),
)

#: Everything this build can run, best default first.
MODELS: tuple[ModelSpec, ...] = (SLIMSAM, YOLOS_TINY)


def find(model_id: str) -> ModelSpec:
    """Return the spec for ``model_id``, naming the known ids otherwise."""
    for spec in MODELS:
        if spec.id == model_id:
            return spec
    known = ", ".join(spec.id for spec in MODELS) or "none"
    raise ToolInputError(f"unknown model {model_id!r}; available models: {known}")


def iter_specs(task: str | None = None) -> Iterator[ModelSpec]:
    """Yield every catalog entry, optionally filtered by task."""
    for spec in MODELS:
        if task is None or spec.task == task:
            yield spec


__all__ = [
    "COCO_LABELS",
    "MODELS",
    "SLIMSAM",
    "YOLOS_TINY",
    "ModelFile",
    "ModelSpec",
    "find",
    "iter_specs",
]
