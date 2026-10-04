"""Model residency: load on first use, keep a couple, unload what is idle.

A loaded model is an ONNX Runtime session — tens to hundreds of megabytes that
stay resident whether or not anything is being segmented. This holds the line:

- **Load on demand** — nothing is loaded until a tool asks for it.
- **A cap** (``GEOAI_MODEL_CACHE_SIZE``, default 2) evicts the least recently
  used model when a new one would exceed it.
- **An idle TTL** (``GEOAI_MODEL_TTL``, default 300 s) drops models nobody has
  touched, checked when the next reservation arrives rather than by a
  background thread: the only entry point is a tool call, so there is nothing
  to do between calls, and no thread to stop at shutdown.
- **A reservation is a promise** — a model being used is never evicted, even
  when it is over the cap or past its TTL.

``reserve`` is the whole lifecycle: with it, a caller gets a loaded session and
the guarantee that nothing unloads it while the block runs.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..contracts.errors import ToolInputError
from ..settings import env
from . import catalog, store
from .catalog import ModelSpec

#: Providers tried in order; the first one this machine actually has wins.
#: AzureExecutionProvider is deliberately absent: it is a stub that reports
#: itself available and then falls back to CPU anyway.
PROVIDER_ORDER = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
)

#: Embeddings kept per loaded model. One is enough for the common case (encode
#: a tile, run many prompts against it); a second keeps a tile warm across the
#: boundary between two tiles of the same scene.
EMBEDDING_CACHE_SIZE = 2

#: Runtime service key the session's manager is published under. One string, so
#: the pack that uses it and the session that closes it cannot disagree.
MANAGER_SERVICE = "ai.manager"


@dataclass(slots=True)
class ModelSession:
    """A loaded model: its ONNX sessions plus the state a caller shares."""

    spec: ModelSpec
    encoder: Any
    decoder: Any
    provider: str
    threads: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Image embeddings by caller-chosen key (see :mod:`~.segmentation`).
    embeddings: dict[str, Any] = field(default_factory=dict)
    loaded_at: float = 0.0
    last_used: float = 0.0
    refcount: int = 0

    def cache_embedding(self, key: str, value: Any) -> Any:
        """Remember an embedding, dropping the oldest key past the cap."""
        self.embeddings[key] = value
        while len(self.embeddings) > EMBEDDING_CACHE_SIZE:
            self.embeddings.pop(next(iter(self.embeddings)))
        return value

    def cached_embedding(self, key: str) -> Any | None:
        """Return a cached embedding, refreshing its position in the order."""
        value = self.embeddings.pop(key, None)
        if value is not None:
            self.embeddings[key] = value
        return value


def _session_options() -> Any:
    """Return ONNX Runtime session options sized for a desktop app.

    Memory pattern reuse is off: the encoder always takes one canvas, but the
    decoder takes a different number of prompts per call, and a pattern built
    for the first shape is wasted planning for the next. Graph optimizations
    stay on — they are what makes the first call slower and every later one
    faster.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = False
    options.intra_op_num_threads = env.onnx_threads()
    options.log_severity_level = 3
    return options


def _resolve_providers() -> list[str]:
    """Return the providers to load with, best available first."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    preferred = [
        name for name in PROVIDER_ORDER if name in available
    ]
    return preferred or ["CPUExecutionProvider"]


class ModelManager:
    """Owns every loaded model in one session, and the policy that bounds them."""

    def __init__(self, *, cache_size: int | None = None, ttl_seconds: float | None = None) -> None:
        self._sessions: dict[str, ModelSession] = {}
        self._lock = threading.RLock()
        self._loading: dict[str, threading.Event] = {}
        self._cache_size = cache_size if cache_size is not None else env.model_cache_size()
        self._ttl = ttl_seconds if ttl_seconds is not None else env.model_ttl()

    # -- lifecycle --------------------------------------------------------

    @contextmanager
    def reserve(
        self,
        model_id: str,
        *,
        download: bool = True,
        progress: store.ProgressHook | None = None,
    ) -> Iterator[ModelSession]:
        """Yield a loaded session for ``model_id``, held until the block exits.

        Downloads the model when it is missing and ``download`` is set. Raises
        :class:`~spatial_intelligence.contracts.errors.ToolInputError` when the
        model is unknown, undownloadable, or onnxruntime is not installed.
        """
        spec = catalog.find(model_id)
        if not store.is_downloaded(spec):
            if not download:
                raise ToolInputError(
                    f"{spec.id} is not downloaded; call ai_pull_model({spec.id!r}) first"
                )
            store.pull(spec, progress=progress)

        session = self._acquire(spec)
        session.refcount += 1
        try:
            yield session
        finally:
            with self._lock:
                session.refcount -= 1
                session.last_used = time.monotonic()

    def _acquire(self, spec: ModelSpec) -> ModelSession:
        """Return a loaded session for ``spec``, loading or evicting as needed."""
        with self._lock:
            existing = self._sessions.get(spec.id)
            if existing is not None:
                return existing

            # One loader at a time per model: a second caller waits rather than
            # loading the same 40 MB session twice.
            waiter = self._loading.get(spec.id)
            if waiter is not None:
                pending = waiter
            else:
                pending = threading.Event()
                self._loading[spec.id] = pending

        if waiter is not None:
            pending.wait(timeout=600)
            with self._lock:
                loaded = self._sessions.get(spec.id)
            if loaded is None:
                raise ToolInputError(f"loading {spec.id} failed; try again")
            return loaded

        try:
            session = self._load(spec)
        finally:
            with self._lock:
                self._loading.pop(spec.id, None)
            pending.set()

        with self._lock:
            self._sessions[spec.id] = session
            self._evict_locked(keep=spec.id)
        return session

    def _load(self, spec: ModelSpec) -> ModelSession:
        """Build the ONNX sessions for ``spec`` (the expensive call)."""
        try:
            import onnxruntime as ort
        except ImportError as exc:  # noqa: BLE001 - one actionable message
            raise ToolInputError(
                "onnxruntime is not installed; install the AI extra "
                "(pip install 'spatial-intelligence[ai]') to run local models"
            ) from exc

        options = _session_options()
        providers = _resolve_providers()
        missing = [
            file.path for file in spec.files if not store.file_path(spec, file).is_file()
        ]
        if missing:
            raise ToolInputError(
                f"{spec.id} is missing {', '.join(missing)}; call ai_pull_model first"
            )
        # The catalog is ordered encoder-first; the last entry is the decoder. A
        # one-file model is a single graph, so both roles point at it.
        encoder_file = spec.files[0]
        decoder_file = spec.files[-1]
        encoder = ort.InferenceSession(
            str(store.file_path(spec, encoder_file)), sess_options=options, providers=providers
        )
        decoder = (
            encoder
            if encoder_file is decoder_file
            else ort.InferenceSession(
                str(store.file_path(spec, decoder_file)), sess_options=options, providers=providers
            )
        )
        now = time.monotonic()
        return ModelSession(
            spec=spec,
            encoder=encoder,
            decoder=decoder,
            provider=encoder.get_providers()[0],
            threads=options.intra_op_num_threads,
            loaded_at=now,
            last_used=now,
        )

    # -- eviction ---------------------------------------------------------

    def _evict_locked(self, keep: str | None = None) -> None:
        """Drop expired and surplus models; callers hold the lock.

        ``keep`` is the model that was just loaded: evicting it would make the
        caller pay a load and immediately throw it away whenever another model
        is still in use.
        """
        now = time.monotonic()
        for model_id, session in list(self._sessions.items()):
            if model_id == keep:
                continue
            if session.refcount == 0 and now - session.last_used > self._ttl:
                self._sessions.pop(model_id, None)

        while len(self._sessions) > max(1, self._cache_size):
            idle = [
                (session.last_used, model_id)
                for model_id, session in self._sessions.items()
                if session.refcount == 0 and model_id != keep
            ]
            if not idle:
                break  # everything resident is in use; the cap waits its turn
            self._sessions.pop(min(idle)[1], None)

    def unload(self, model_id: str = "") -> dict:
        """Unload one model, or every idle model when ``model_id`` is empty."""
        with self._lock:
            if model_id:
                spec = catalog.find(model_id)
                session = self._sessions.get(spec.id)
                if session is None:
                    return {"unloaded": [], "resident": sorted(self._sessions)}
                if session.refcount:
                    raise ToolInputError(f"{spec.id} is in use and cannot be unloaded")
                self._sessions.pop(spec.id, None)
                return {"unloaded": [spec.id], "resident": sorted(self._sessions)}

            unloaded = [
                model_id
                for model_id, session in list(self._sessions.items())
                if session.refcount == 0
            ]
            for name in unloaded:
                self._sessions.pop(name, None)
            return {"unloaded": unloaded, "resident": sorted(self._sessions)}

    def close(self) -> None:
        """Unload everything (workspace close, or app shutdown)."""
        with self._lock:
            self._sessions.clear()

    # -- reporting --------------------------------------------------------

    def status(self, task: str | None = None) -> list[dict]:
        """Return one row per catalog model: disk state plus residency."""
        now = time.monotonic()
        with self._lock:
            rows = []
            for spec in catalog.iter_specs(task):
                session = self._sessions.get(spec.id)
                row = store.summary(spec)
                row.update(
                    {
                        "loaded": session is not None,
                        "in_use": bool(session and session.refcount),
                        "provider": session.provider if session else None,
                        "threads": session.threads if session else None,
                        "idle_seconds": (
                            round(now - session.last_used, 1) if session else None
                        ),
                        "cached_embeddings": len(session.embeddings) if session else 0,
                    }
                )
                rows.append(row)
            return rows

    @property
    def cache_size(self) -> int:
        """The residency cap this manager runs with."""
        return self._cache_size

    @property
    def ttl_seconds(self) -> float:
        """The idle TTL this manager runs with."""
        return self._ttl

    def loaded_ids(self) -> list[str]:
        """Ids of the models currently resident, newest first."""
        with self._lock:
            return sorted(self._sessions)


def get_manager(runtime: Any) -> ModelManager:
    """Return the session's manager, creating it on first use.

    Kept on the runtime's service bag, which lives as long as the open
    workspace: models survive every cell and every run, and die with it.
    """
    return runtime.service(MANAGER_SERVICE, ModelManager)


__all__ = [
    "EMBEDDING_CACHE_SIZE",
    "MANAGER_SERVICE",
    "ModelManager",
    "ModelSession",
    "get_manager",
]
