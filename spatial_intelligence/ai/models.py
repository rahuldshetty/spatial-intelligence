"""The local model catalog and the on-disk store that serves it.

One model ships wired up — SlimSAM, the pruned SAM that GeoLibre's own Segment
Everything panel uses — so the pipeline has a known-good target; adding another
is a :class:`ModelSpec` entry, not code.

Every file lands under::

    <GEOAI_HOME>/.models/<repo>/<revision>/<file...>

A revision is immutable upstream, so a file that verified once is never fetched
again, and the sha256 in the catalog is the one Hugging Face publishes as its
LFS object id: the download is checked against upstream metadata rather than
against itself. Files that carry their weights beside them (ONNX external data,
``*.onnx_data``) stay in the directory of their ``.onnx`` — ONNX Runtime
resolves them by relative path.

Downloads stream to a temporary file that is renamed into place only after the
hash matches, so a killed process cannot leave a half model that a later run
would accept.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from ..contracts.errors import ToolInputError
from ..settings import env

#: Where a catalog file is fetched from, given its repo path.
HUB_BASE_URL = "https://huggingface.co"
#: Bytes read (and hashed) per chunk.
CHUNK_BYTES = 1 << 20
#: Free space the volume must keep after a download. A dev checkout stores
#: workspaces and models on one volume, so a full disk is a real failure mode
#: and a precise message beats a truncated file.
DISK_HEADROOM_BYTES = 512 << 20
#: Manifest a fully fetched revision carries.
MANIFEST_NAME = ".complete.json"

#: Progress callback: ``(done_bytes, total_bytes, label)``.
ProgressHook = Callable[[float, float, str], None]


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

#: Everything this build can run, best default first.
MODELS: tuple[ModelSpec, ...] = (SLIMSAM,)


def find(model_id: str) -> ModelSpec:
    """Return the spec for ``model_id``, naming the known ids otherwise."""
    for spec in MODELS:
        if spec.id == model_id:
            return spec
    known = ", ".join(spec.id for spec in MODELS) or "none"
    raise ToolInputError(f"unknown model {model_id!r}; available models: {known}")


def models_dir() -> Path:
    """Return the model cache root (``GEOAI_MODELS_DIR`` or ``<GEOAI_HOME>/.models``)."""
    return env.models_dir()


def is_offline() -> bool:
    """Whether downloads are refused (``GEOAI_OFFLINE`` set to a truthy value)."""
    return os.getenv("GEOAI_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _token() -> str:
    """Return the Hugging Face token for a gated repository, if one is set."""
    return (
        os.getenv("HF_TOKEN", "").strip() or os.getenv("GEOAI_HF_TOKEN", "").strip()
    )


def spec_dir(spec: ModelSpec) -> Path:
    """Return the directory a revision's files live in."""
    return models_dir() / spec.repo / spec.revision


def file_path(spec: ModelSpec, file: ModelFile) -> Path:
    """Return the absolute path of one catalog file."""
    return spec_dir(spec) / file.path


def manifest_path(spec: ModelSpec) -> Path:
    """Return the path of the revision's completion manifest."""
    return spec_dir(spec) / MANIFEST_NAME


def on_disk_bytes(spec: ModelSpec) -> int:
    """Return the bytes of ``spec`` actually present (0 when nothing is fetched)."""
    total = 0
    for file in spec.files:
        path = file_path(spec, file)
        if path.is_file():
            total += path.stat().st_size
    return total


def is_downloaded(spec: ModelSpec) -> bool:
    """Whether every catalog file is present at its recorded size.

    Size is the cheap check; :func:`verify` re-hashes. `pull` hashes anyway on
    the way in, so a file that is here at the right size came from a verified
    download.
    """
    return all(
        file_path(spec, file).is_file()
        and file_path(spec, file).stat().st_size == file.size
        for file in spec.files
    )


def _sha256(path: Path) -> str:
    """Return the hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(spec: ModelSpec) -> list[str]:
    """Return the catalog paths of ``spec`` that are missing or hash differently."""
    bad: list[str] = []
    for file in spec.files:
        path = file_path(spec, file)
        if not path.is_file() or path.stat().st_size != file.size:
            bad.append(file.path)
        elif _sha256(path) != file.sha256:
            bad.append(file.path)
    return bad


def _free_bytes(path: Path) -> int:
    """Return free bytes on the volume holding ``path`` (walking up if needed)."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:  # a volume that cannot be measured is not a reason to refuse
        return DISK_HEADROOM_BYTES * 2


def _download(
    spec: ModelSpec,
    file: ModelFile,
    *,
    client: Any,
    progress: ProgressHook | None,
    done: float,
    total: float,
) -> None:
    """Stream one file into place, hashing as it goes; rename only on a match."""
    url = f"{HUB_BASE_URL}/{spec.repo}/resolve/{spec.revision}/{file.path}"
    destination = file_path(spec, file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.part")

    headers = {"User-Agent": "spatial-intelligence-model-store/1"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    digest = hashlib.sha256()
    written = 0
    try:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code in (401, 403):
                raise ToolInputError(
                    f"{spec.repo} refused the download (HTTP {response.status_code}); "
                    + (
                        "accept the model license on Hugging Face and set HF_TOKEN."
                        if spec.gated
                        else "the repository may have moved or become gated."
                    )
                )
            response.raise_for_status()
            with staging.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(done + written, total, file.path)
    except ToolInputError:
        staging.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001 - one message for any transport failure
        staging.unlink(missing_ok=True)
        raise ToolInputError(f"could not download {file.path} from {url}: {exc}") from exc

    if written != file.size or digest.hexdigest() != file.sha256:
        staging.unlink(missing_ok=True)
        raise ToolInputError(
            f"{file.path} failed verification (expected {file.size} bytes / "
            f"{file.sha256[:12]}…, got {written} bytes / {digest.hexdigest()[:12]}…); "
            "nothing was written to the model cache"
        )
    os.replace(staging, destination)


def pull(
    spec: ModelSpec,
    *,
    verify_hashes: bool = False,
    progress: ProgressHook | None = None,
    client: Any = None,
) -> dict:
    """Fetch a revision into the model cache and return a status dict.

    Files already present at the right size are skipped; ``verify_hashes``
    re-hashes them instead. Nothing is renamed into place before its hash
    matches, so an interrupted run leaves the cache as it was.
    """
    if is_offline():
        raise ToolInputError(
            "downloads are disabled (GEOAI_OFFLINE is set); unset it to fetch "
            f"{spec.id}, or place its files under {spec_dir(spec)}"
        )

    missing = verify(spec) if verify_hashes else [
        file.path for file in spec.files if not _present(spec, file)
    ]
    wanted = [file for file in spec.files if file.path in set(missing)]
    if wanted:
        needed = sum(file.size for file in wanted)
        free = _free_bytes(models_dir())
        if free - needed < DISK_HEADROOM_BYTES:
            raise ToolInputError(
                f"not enough disk space for {spec.id}: {needed / 1e6:.1f} MB needed, "
                f"{free / 1e6:.1f} MB free on {models_dir()}"
            )

    if client is None:
        import httpx

        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=300.0)) as owned:
            _fetch_all(spec, wanted, owned, progress)
    else:
        _fetch_all(spec, wanted, client, progress)

    manifest = {
        "repo": spec.repo,
        "revision": spec.revision,
        "model_id": spec.id,
        "files": {file.path: {"sha256": file.sha256, "size": file.size} for file in spec.files},
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = manifest_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{MANIFEST_NAME}.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)

    return {
        "model_id": spec.id,
        "path": str(spec_dir(spec)),
        "downloaded": len(wanted),
        "skipped": len(spec.files) - len(wanted),
        "bytes": spec.total_bytes,
        "files": [file.path for file in spec.files],
    }


def _present(spec: ModelSpec, file: ModelFile) -> bool:
    """Whether one catalog file is present at its recorded size."""
    path = file_path(spec, file)
    return path.is_file() and path.stat().st_size == file.size


def _fetch_all(
    spec: ModelSpec, wanted: list[ModelFile], client: Any, progress: ProgressHook | None
) -> None:
    """Fetch the wanted files in catalog order, reporting cumulative bytes."""
    total = float(sum(file.size for file in wanted)) or 1.0
    done = 0.0
    for file in wanted:
        _download(spec, file, client=client, progress=progress, done=done, total=total)
        done += file.size


def summary(spec: ModelSpec) -> dict:
    """Return the catalog row for ``spec`` (disk state only, no residency)."""
    present = on_disk_bytes(spec)
    return {
        "id": spec.id,
        "task": spec.task,
        "prompts": list(spec.prompts),
        "repo": spec.repo,
        "revision": spec.revision,
        "license": spec.license,
        "bytes": spec.total_bytes,
        "on_disk_bytes": present,
        "downloaded": is_downloaded(spec),
        "notes": spec.notes,
    }


def iter_specs(task: str | None = None) -> Iterator[ModelSpec]:
    """Yield every catalog entry, optionally filtered by task."""
    for spec in MODELS:
        if task is None or spec.task == task:
            yield spec


__all__ = [
    "MODELS",
    "SLIMSAM",
    "ModelFile",
    "ModelSpec",
    "file_path",
    "find",
    "is_downloaded",
    "is_offline",
    "iter_specs",
    "manifest_path",
    "models_dir",
    "on_disk_bytes",
    "pull",
    "spec_dir",
    "summary",
    "verify",
]
