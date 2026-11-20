"""The dataset cache: what one workspace's searches remembered.

Search and add are separate tool calls, sometimes in separate runs.  The cache
maps ``source:id`` to the normalized dataset so the second call never has to
repeat the search.  It stores unsigned records only; a source's ``present``
method signs a copy for the model when needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from ..workspace import Workspace
from .base import Dataset

#: Name of the persisted cache inside ``traces/``.
CACHE_FILENAME = "data-datasets.json"
#: Most datasets kept on disk; older ones are dropped.
MAX_CACHED_DATASETS = 1000


class DatasetCache:
    """One workspace's remembered datasets, in memory and on disk.

    The registry UI can search before a workspace is open; in that case the
    cache has no workspace and keeps results in memory for the request only.
    """

    def __init__(self, workspace: Workspace | None) -> None:
        self._workspace = workspace
        self._lock = Lock()
        self._datasets: dict[str, Dataset] = {}

    @property
    def path(self) -> Path | None:
        """The file the cache persists to, or ``None`` without a workspace."""
        if self._workspace is None:
            return None
        return self._workspace.traces / CACHE_FILENAME

    def remember(self, dataset: Dataset) -> Dataset:
        """Store ``dataset`` and return it unchanged."""
        with self._lock:
            self._datasets[dataset.key] = dataset
            path = self.path
            if path is None:
                return dataset
            try:
                stored = self._read_file()
                stored[dataset.key] = dataset.as_dict()
                stored = dict(list(stored.items())[-MAX_CACHED_DATASETS:])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(stored, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            except (OSError, ValueError, json.JSONDecodeError):
                # A search that succeeded must not fail because its cache could
                # not be written; the in-memory copy still serves this process.
                pass
        return dataset

    def get(self, dataset_key: str) -> Dataset | None:
        """Return a remembered dataset, or ``None`` when the key is unknown."""
        with self._lock:
            dataset = self._datasets.get(dataset_key)
            if dataset is not None:
                return dataset
            try:
                stored = self._read_file()
                raw = stored.get(dataset_key)
                if isinstance(raw, dict):
                    dataset = Dataset.from_dict(raw)
                    self._datasets[dataset_key] = dataset
                    return dataset
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return None

    def remember_all(self, datasets: tuple[Dataset, ...]) -> None:
        """Remember every dataset in one page."""
        for dataset in datasets:
            self.remember(dataset)

    # -- internals -------------------------------------------------------

    def _read_file(self) -> dict:
        """Return the persisted datasets, or ``{}`` when the file is unusable."""
        path = self.path
        if path is None or not path.exists():
            return {}
        stored = json.loads(path.read_text(encoding="utf-8"))
        return stored if isinstance(stored, dict) else {}


__all__ = ["CACHE_FILENAME", "DatasetCache", "MAX_CACHED_DATASETS"]
