"""Skills: the API reference for the heavy sandbox packages, read on demand.

One tool, three questions: which packages have a tree, what is in one package's
tree, and what does a file (or one query inside it) say. Nothing here is loaded
into the prompt — the agent asks for the page it needs. Each call reports a
progress job, which the browser draws as a card in the asking cell.
"""

from __future__ import annotations

from ...contracts.effects import Effect
from ...pythonruntime import skill as skills
from ..runtime import ToolRuntime
from ..spec import pack, tool

#: Files named in one search card before the rest are left to the result.
MAX_CARD_FILES = 3


def _label(package: str, path: str, query: str) -> str:
    """Return the card title for one call."""
    if not package:
        return f"Searched skills for {query!r}" if query else "Loaded skill index"
    if query and not path:
        return f"Searched {package} for {query!r}"
    target = path or "index.md"
    suffix = f" matching {query!r}" if query else ""
    return f"Loaded skill {package}/{target}{suffix}"


def _unit(package: str, path: str, query: str) -> str:
    """Return what the card counts in — known from the request, not the read."""
    if not package:
        return "hits" if query else "trees"
    return "hits" if query and not path else "lines"


def _outcome(result: dict) -> tuple[float, str]:
    """Return the card's amount and closing line for whatever came back."""
    kind = result.get("kind")
    if kind == "index":
        rows = result.get("packages", [])
        return float(len(rows)), ", ".join(row["package"] for row in rows)
    if kind == "search":
        hits = result.get("hits", [])
        if not hits:
            return 0.0, str(result.get("doc", "no line matched"))
        files = sorted({hit["file"] for hit in hits})
        shown = ", ".join(files[:MAX_CARD_FILES])
        return float(len(hits)), f"in {shown}{' …' if len(files) > MAX_CARD_FILES else ''}"
    if kind == "error":
        return 0.0, str(result.get("doc", "not found"))
    matched = result.get("matched_lines")
    amount = matched if matched is not None else result.get("total_lines", 0)
    return float(amount), f"{result.get('file', '')} ({result.get('source', '')})"


@pack(category="skills", effects=frozenset({Effect.READ}))
class SkillsPack:
    """The committed API references, served a page at a time."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime

    @tool(core=True)
    def skill(self, package: str = "", path: str = "", query: str = "", start: int = 0) -> dict:
        """Read the API reference for a heavy package in the run_python sandbox.

        Call it with no arguments to list what has a reference — the packages
        that can be imported in the sandbox (torchgeo, terratorch, ...) and the
        shared `models` tree of checkpoints. Call it with a name to get that
        tree's index: what it is, how it is meant to be used, and which file
        answers which question. Then read one file, e.g.
        skill("torchgeo", path="api/models.md") or skill("models", path="prithvi.md")
        — or ask a question of every tree with skill(query="sentinel-2 pretrained")
        and follow the file it names. Each result reports the version a tree was
        generated from and whether that library is importable here, so a reference
        is never mistaken for something this machine can run.

        Use python_help("torchgeo.models.ResNet18_Weights") for one symbol's
        exact signature; this tool is for finding out what exists and how it is
        meant to be used. Weights are not here: fetch them with ai_fetch_model.
        """
        package, path, query = package.strip(), path.strip(), query.strip()
        job = self._rt.reporter.job("skill", _label(package, path, query), unit=_unit(package, path, query))
        with job:
            result = self._read(package, path, query, start)
            if result.get("kind") == "error":
                job.fail(result.get("doc", "not found"))
            else:
                amount, detail = _outcome(result)
                # A read is instant, so the card opens and fills in one step: the
                # amount it reports is the amount it read.
                job.progress(amount, total=amount, detail=detail)
                job.done()
        return result

    @staticmethod
    def _read(package: str, path: str, query: str, start: int) -> dict:
        """Answer one call: the index, a search, or a page of a file."""
        if not package:
            if query:
                return {"kind": "search", **skills.search(query)}
            return {"kind": "index", "packages": skills.available()}
        if query and not path:
            # A question about a whole package: search the tree, then read the
            # file the hits name. With a path, narrow that one file instead.
            return {"kind": "search", **skills.search(query, package=package)}
        return skills.read(package, path, query=query, start=start)


__all__ = ["SkillsPack"]
