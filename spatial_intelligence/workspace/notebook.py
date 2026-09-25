"""In-memory cell model and its ``.ipynb`` (nbformat 4.5) serialization.

The server owns the notebook schema; no ``nbformat`` dependency. The internal
cell dict is JSON-serializable and carries a runtime-only ``status`` that is
derived (never serialized) on load.

Ported from ``geoai/server/notebook.py``; the on-disk ``.ipynb`` layout and the
namespaced ``metadata.geoai`` keys are unchanged so existing notebooks load.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..contracts.ids import new_id

VALID_KINDS = frozenset({"markdown", "python", "prompt", "tool", "interaction"})

_NBFORMAT = 4
_NBFORMAT_MINOR = 5


def new_cell(
    kind: str,
    source: str = "",
    index: int | None = None,
    *,
    metadata: dict | None = None,
) -> dict:
    """Return a fresh cell dict (uuid4 id, status "idle").

    ``index`` is applied by the caller (``state.add_cell``); it is accepted here
    only for API parity and is otherwise unused.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"invalid cell kind: {kind!r}")
    return {
        "id": new_id(),
        "kind": kind,
        "source": source,
        "outputs": [],
        "execution_count": None,
        "status": "idle",
        "metadata": metadata or {},
    }


def _outputs_to_nb(outputs: list[dict]) -> list[dict]:
    """Convert internal (normalized) outputs to nbformat output objects."""
    result: list[dict] = []
    for out in outputs:
        if out.get("output_type") == "error":
            result.append(
                {
                    "output_type": "error",
                    "ename": out.get("ename"),
                    "evalue": out.get("evalue"),
                    "traceback": out.get("traceback") or [],
                }
            )
        else:
            result.append(
                {
                    "output_type": "stream",
                    "name": out.get("name") or "stdout",
                    "text": out.get("text") or "",
                }
            )
    return result


def _outputs_from_nb(nb_outputs: list[dict]) -> list[dict]:
    """Convert nbformat output objects to internal (normalized) outputs."""
    result: list[dict] = []
    for out in nb_outputs:
        if out.get("output_type") == "error":
            result.append(
                {
                    "output_type": "error",
                    "name": None,
                    "text": None,
                    "ename": out.get("ename"),
                    "evalue": out.get("evalue"),
                    "traceback": out.get("traceback"),
                }
            )
        elif out.get("output_type") == "stream":
            result.append(
                {
                    "output_type": "stream",
                    "name": out.get("name"),
                    "text": out.get("text"),
                    "ename": None,
                    "evalue": None,
                    "traceback": None,
                }
            )
        # Other nbformat output types are not produced by this harness and are
        # dropped (we own the schema).
    return result


def cell_to_nb(cell: dict) -> dict:
    """Map an internal cell to its nbformat 4.5 dict."""
    kind = cell["kind"]
    source_lines = cell.get("source", "").splitlines(keepends=True)
    metadata = dict(cell.get("metadata") or {})
    geoai = dict(metadata.get("geoai") or {})
    metadata["geoai"] = geoai
    if kind == "markdown":
        if not geoai:
            metadata.pop("geoai", None)
        return {
            "cell_type": "markdown",
            "id": cell["id"],
            "metadata": metadata,
            "source": source_lines,
        }
    if kind == "prompt":
        # Prompt cells are executable inside the app, but serialize as Markdown
        # so the user's request reads naturally in any standard notebook.
        # Runtime details stay in namespaced metadata; agent actions and the
        # final response are recorded as subsequent notebook cells.
        geoai.update(
            {
                "kind": "prompt",
                "execution_count": cell.get("execution_count"),
                "status": cell.get("status", "idle"),
                "interaction_history": cell.get("interaction_history", []),
            }
        )
        return {
            "cell_type": "markdown",
            "id": cell["id"],
            "metadata": metadata,
            "source": source_lines,
        }
    if kind == "interaction":
        geoai.update(
            {
                "kind": "interaction",
                "status": cell.get("status", "idle"),
                "interaction": cell.get("interaction"),
                "answers": cell.get("answers"),
            }
        )
        return {
            "cell_type": "markdown",
            "id": cell["id"],
            "metadata": metadata,
            "source": source_lines,
        }
    nb = {
        "cell_type": "code",
        "id": cell["id"],
        "metadata": metadata,
        "execution_count": cell.get("execution_count"),
        "outputs": _outputs_to_nb(cell.get("outputs", [])),
        "source": source_lines,
    }
    if kind == "tool":
        geoai["kind"] = "tool"
        geoai["status"] = cell.get("status", "idle")
    return nb


def nb_to_cell(nb_cell: dict) -> dict:
    """Map an nbformat cell to an internal cell dict (derives ``status``)."""
    cell_type = nb_cell.get("cell_type")
    metadata = dict(nb_cell.get("metadata") or {})
    geoai = metadata.get("geoai", {})
    geoai_kind = geoai.get("kind")
    if geoai_kind == "prompt":
        kind = "prompt"
    elif geoai_kind == "interaction":
        kind = "interaction"
    elif geoai_kind == "tool":
        kind = "tool"
    elif cell_type == "markdown":
        kind = "markdown"
    else:
        kind = "python"

    outputs = _outputs_from_nb(nb_cell.get("outputs", []))
    if kind in {"markdown", "interaction"}:
        execution_count = None
    elif kind == "prompt" and cell_type == "markdown":
        execution_count = geoai.get("execution_count")
    else:
        execution_count = nb_cell.get("execution_count")
    status = geoai.get("status") if kind in {"prompt", "tool", "interaction"} else None
    if status not in {"idle", "running", "waiting_for_input", "done", "error", "stopped"}:
        status = "done" if (outputs or execution_count is not None) else "idle"

    cell = {
        "id": nb_cell.get("id") or new_id(),
        "kind": kind,
        "source": "".join(nb_cell.get("source", [])),
        "outputs": outputs,
        "execution_count": execution_count,
        "status": status,
        "metadata": metadata,
    }
    if kind == "prompt" and isinstance(geoai.get("interaction"), dict):
        cell["interaction"] = geoai["interaction"]
        cell["status"] = "waiting_for_input"
    if kind == "prompt" and isinstance(geoai.get("interaction_history"), list):
        cell["interaction_history"] = geoai["interaction_history"]
    if kind == "interaction":
        if isinstance(geoai.get("interaction"), dict):
            cell["interaction"] = geoai["interaction"]
        if isinstance(geoai.get("answers"), dict):
            cell["answers"] = geoai["answers"]
    return cell


#: Attempts and delay for :func:`_replace_with_retry`.
_REPLACE_ATTEMPTS = 6
_REPLACE_DELAY = 0.02


def _replace_with_retry(source: Path, target: Path) -> None:
    """Rename ``source`` over ``target``, retrying a transient Windows refusal.

    ``os.replace`` is atomic, but Windows still refuses it with
    ``PermissionError`` while another handle has the destination open without
    delete sharing — most often a reader that arrived microseconds earlier. A few
    short retries cover that window, and the reader keeps the guarantee that it
    sees either the old file or the new one.
    """
    import time

    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_DELAY)


def read_nb(path: Path) -> list[dict]:
    """Load cells from ``path``; a missing file yields ``[]``."""
    if not path.exists():
        return []
    nb = json.loads(path.read_text(encoding="utf-8"))
    return [nb_to_cell(c) for c in nb.get("cells", [])]


def write_nb(path: Path, cells: list[dict]) -> None:
    """Write ``cells`` as an nbformat 4.5 JSON notebook (mkdir parents).

    Written to a temporary sibling and then renamed over the target, so a reader
    (the browser reloading the Cells tab, a test asserting on the document) sees
    either the previous notebook or the complete new one. Writing in place let a
    reader catch a truncated file, which parsed as an empty document and failed
    as a JSON error rather than as a notebook.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [cell_to_nb(c) for c in cells],
        "metadata": {},
        "nbformat": _NBFORMAT,
        "nbformat_minor": _NBFORMAT_MINOR,
    }
    payload = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        _replace_with_retry(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
