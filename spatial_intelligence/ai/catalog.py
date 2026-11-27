"""Which models this build can run, and what each one needs to be fed.

Data, not behaviour: adding a model is an entry here plus a pipeline for its
architecture, never a change to the store or the manager. Each entry pins the
upstream revision — HF revisions are immutable, so a pinned one cannot change
under a verified download — and every file by hash and size.

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
    """One file of a model revision, pinned by hash and size.

    ``sha256`` is the LFS object id (the Hub publishes it for LFS files only);
    ``sha1`` is the git blob id, which every file has.
    """

    path: str
    sha256: str
    size: int
    sha1: str = ""


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

    A super-resolution model carries ``scale``, the number of ``dates`` its
    graph takes, and the per-date ``input_bands`` in model order. Its graph is
    unloaded as soon as the last reservation ends when ``unload_when_idle`` is
    set, because one variant is large and runs are sparse.
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
    scale: int = 1
    dates: int = 1
    input_bands: tuple[str, ...] = ()
    unload_when_idle: bool = False

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

#: Satlas ESRGAN ONNX exports, published from the upstream Apache-2.0 weights.
#: One graph per variant; the 4x scale and date-major bands are the only
#: architecture facts the pipeline needs.
SATLAS_REPO = "rahuldshetty/satlas-esrgan-onnx"
SATLAS_REVISION = "d0a4811582990b18056eac0b02dd5d03c7bb8363"

_SATLAS_RGB = ("tci", "tci", "tci")
_SATLAS_10M = _SATLAS_RGB + ("b08",)
_SATLAS_20M = _SATLAS_10M + ("b05", "b06", "b07", "b11", "b12")
_SATLAS_60M = _SATLAS_RGB + ("b01", "b08", "b09", "b10", "b05", "b06", "b07", "b11", "b12")


def _satlas(
    model_id: str,
    *,
    file: str,
    sha256: str,
    size: int,
    dates: int,
    input_bands: tuple[str, ...],
    notes: str,
) -> ModelSpec:
    """One Satlas ONNX variant, pinned to the published Hub revision."""
    return ModelSpec(
        id=model_id,
        repo=SATLAS_REPO,
        revision=SATLAS_REVISION,
        files=(ModelFile(path=file, sha256=sha256, size=size),),
        task="super-resolution",
        prompts=(),
        scale=4,
        dates=dates,
        input_bands=input_bands,
        unload_when_idle=True,
        license="apache-2.0",
        notes=notes,
    )


SATLAS_ESRGAN_1S2 = _satlas(
    "satlas-esrgan-1s2",
    file="esrgan_1S2.onnx",
    sha256="579a24618dbf552358a9d95b2593023dbeddbeabd2581fcd84a5e77d1cd983de",
    size=67_051_616,
    dates=1,
    input_bands=_SATLAS_RGB,
    notes="Satlas ESRGAN, 4x, single-date TCI RGB. The default: 67 MB, ONNX.",
)
SATLAS_ESRGAN_2S2 = _satlas(
    "satlas-esrgan-2s2",
    file="esrgan_2S2.onnx",
    sha256="3c297d6e6ea0ae0858a9af5000baeef60845c1f7f87e988393af8aa3d9aa47c9",
    size=67_058_528,
    dates=2,
    input_bands=_SATLAS_RGB,
    notes="Satlas ESRGAN, 4x, two co-registered TCI dates.",
)
SATLAS_ESRGAN_4S2 = _satlas(
    "satlas-esrgan-4s2",
    file="esrgan_4S2.onnx",
    sha256="980723d4e0bede5077a9c5a37cb3675b8339ecb33e8639a0f4845402f605b4c5",
    size=67_072_354,
    dates=4,
    input_bands=_SATLAS_RGB,
    notes="Satlas ESRGAN, 4x, four co-registered TCI dates.",
)
SATLAS_ESRGAN_8S2 = _satlas(
    "satlas-esrgan-8s2",
    file="esrgan_8S2.onnx",
    sha256="1e35b59a5cbe2b4d2622f2bf3695410e4331aa0e1761ed3fb654d384f26be540",
    size=67_100_002,
    dates=8,
    input_bands=_SATLAS_RGB,
    notes="Satlas ESRGAN, 4x, eight co-registered TCI dates.",
)
SATLAS_ESRGAN_16S2 = _satlas(
    "satlas-esrgan-16s2",
    file="esrgan_16S2.onnx",
    sha256="6dced05737d9d0194fe538ae882b7a260e77fb37e49569be4a19e717320f0a6c",
    size=67_155_298,
    dates=16,
    input_bands=_SATLAS_RGB,
    notes="Satlas ESRGAN, 4x, sixteen co-registered TCI dates.",
)
SATLAS_ESRGAN_10M = _satlas(
    "satlas-esrgan-10m",
    file="esrgan_10m.onnx",
    sha256="265f6f7813372bb5d9718fe171414657a0d69807a73e6e52f360c793246cf162",
    size=67_118_434,
    dates=8,
    input_bands=_SATLAS_10M,
    notes="Satlas ESRGAN, 4x, eight dates of TCI RGB + B08.",
)
SATLAS_ESRGAN_20M = _satlas(
    "satlas-esrgan-20m",
    file="esrgan_20m.onnx",
    sha256="6249a24d3e42147877dc14512044f3455f6d593a3a26eda806153005ea0dd529",
    size=67_210_594,
    dates=8,
    input_bands=_SATLAS_20M,
    notes="Satlas ESRGAN, 4x, eight dates of TCI RGB + B08/B05/B06/B07/B11/B12.",
)
SATLAS_ESRGAN_60M = _satlas(
    "satlas-esrgan-60m",
    file="esrgan_60m.onnx",
    sha256="98d6c2661d2f4bcc2e3d2b06a9e5eb8c8f08c22abdce498b7b31a307e99157d1",
    size=67_265_890,
    dates=8,
    input_bands=_SATLAS_60M,
    notes="Satlas ESRGAN, 4x, eight dates of TCI RGB + B01/B05/B06/B07/B08/B09/B10/B11/B12.",
)

#: Everything this build can run, best default first.
MODELS: tuple[ModelSpec, ...] = (
    SLIMSAM,
    YOLOS_TINY,
    SATLAS_ESRGAN_1S2,
    SATLAS_ESRGAN_2S2,
    SATLAS_ESRGAN_4S2,
    SATLAS_ESRGAN_8S2,
    SATLAS_ESRGAN_16S2,
    SATLAS_ESRGAN_10M,
    SATLAS_ESRGAN_20M,
    SATLAS_ESRGAN_60M,
)


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
