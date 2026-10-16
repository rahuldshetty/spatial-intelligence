"""Hugging Face Hub metadata: keyword search and one repository's file list.

Metadata only — the weights themselves are :mod:`ai.store`'s job, which verifies
and caches them. Both calls here are the public JSON API that ``ai_fetch_model``
already talks to, so they take the same token when one is set, and they answer
with plain dicts a picker can render.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from ..contracts.errors import ToolInputError
from .store import _token, is_offline

HUB_BASE_URL = "https://huggingface.co"
USER_AGENT = "spatial-intelligence/0.1 hub-client"
TIMEOUT_SECONDS = 30.0
#: Cap on one metadata document; a repo detail with siblings is tens of KB.
MAX_BYTES = 8 * 1024 * 1024

DEFAULT_LIMIT = 8
MAX_LIMIT = 25

#: Name suffixes that mean "there are weights in here we could load".
WEIGHT_SUFFIXES = (".onnx", ".safetensors", ".pt", ".pth", ".bin", ".ckpt")

#: Tags that carry a license in the Hub's tag list.
LICENSE_PREFIX = "license:"


def _get_json(url: str) -> object:
    """GET one Hub JSON document, honouring the offline switch and the token."""
    if is_offline():
        raise ToolInputError("offline mode is on, so the model hub cannot be reached")
    headers = {"User-Agent": USER_AGENT}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_BYTES)
    except Exception as exc:  # urllib raises a wide family here
        raise ToolInputError(f"the model hub could not be reached: {exc}") from exc
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise ToolInputError("the model hub did not return JSON") from exc


def _license_of(tags: list[str]) -> str | None:
    """Return the license a repo's tags name, if any."""
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(LICENSE_PREFIX):
            return tag[len(LICENSE_PREFIX) :]
    return None


def search_models(query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """Search the Hub by keyword, most downloaded first.

    The rows carry what the search response holds — enough to render a picker.
    Anything that needs the file list (whether the repo is usable at all) is a
    second call, made when the user actually picks something.
    """
    text = " ".join(str(query or "").split())
    if not text:
        raise ToolInputError("a search term is required")
    count = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    url = f"{HUB_BASE_URL}/api/models?" + urllib.parse.urlencode(
        {
            "search": text,
            "limit": count,
            "sort": "downloads",
            "direction": -1,
            "full": "true",
        }
    )
    payload = _get_json(url)
    if not isinstance(payload, list):
        raise ToolInputError("the model hub returned an unexpected search response")
    results: list[dict] = []
    for entry in payload:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        tags = [tag for tag in (entry.get("tags") or []) if isinstance(tag, str)]
        results.append(
            {
                "ref": f"hf:{entry['id']}",
                "repo": entry["id"],
                "downloads": int(entry.get("downloads") or 0),
                "likes": int(entry.get("likes") or 0),
                "library": entry.get("library_name"),
                "pipeline": entry.get("pipeline_tag"),
                "license": _license_of(tags),
                "gated": bool(entry.get("gated")),
                "private": bool(entry.get("private")),
                "tags": [tag for tag in tags if not tag.startswith("region:")][:12],
            }
        )
    return results


def repo_info(repo: str) -> dict:
    """Return one repository's revision, file list, and declared metadata."""
    name = str(repo or "").strip().strip("/")
    if not name or "/" not in name:
        raise ToolInputError(f"a repository id like 'owner/name' is required, got {repo!r}")
    payload = _get_json(f"{HUB_BASE_URL}/api/models/{urllib.parse.quote(name)}?full=true")
    if not isinstance(payload, dict):
        raise ToolInputError(f"the model hub returned no repository for {name!r}")
    files = [
        sibling["rfilename"]
        for sibling in (payload.get("siblings") or [])
        if isinstance(sibling, dict) and sibling.get("rfilename")
    ]
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    id2label = config.get("id2label") if isinstance(config.get("id2label"), dict) else {}
    labels = [str(value) for value in id2label.values()]
    tags = [tag for tag in (payload.get("tags") or []) if isinstance(tag, str)]
    return {
        "repo": payload.get("id") or name,
        "revision": str(payload.get("sha") or ""),
        "pipeline": payload.get("pipeline_tag"),
        "library": payload.get("library_name"),
        "gated": bool(payload.get("gated")),
        "private": bool(payload.get("private")),
        "license": _license_of(tags),
        "tags": [tag for tag in tags if not tag.startswith("region:")],
        "architectures": list(config.get("architectures") or []),
        "labels": labels,
        "files": files,
        "downloads": int(payload.get("downloads") or 0),
        "likes": int(payload.get("likes") or 0),
    }


def weight_files(files: list[str]) -> dict[str, list[str]]:
    """Group a repo's files by the kind of weights they hold."""
    grouped: dict[str, list[str]] = {"onnx": [], "safetensors": [], "torch": []}
    for path in files:
        lowered = str(path).lower()
        if lowered.endswith(".onnx"):
            grouped["onnx"].append(path)
        elif lowered.endswith(".safetensors"):
            grouped["safetensors"].append(path)
        elif lowered.endswith((".pt", ".pth", ".bin", ".ckpt")):
            grouped["torch"].append(path)
    return grouped


__all__ = [
    "HUB_BASE_URL",
    "WEIGHT_SUFFIXES",
    "repo_info",
    "search_models",
    "weight_files",
]
