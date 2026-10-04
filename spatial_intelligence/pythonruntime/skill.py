"""Reading the skills tree: what a package's API reference says, one file at a time.

The agent should never hold a whole API reference. This resolves the tree for a
package (the user cache wins over the shipped one), searches it by line, and
reads one file with paging — the same shape as ``inspect_output``. The package
itself is only imported when a tree is built, never to answer a question about
one, so a machine without the extra can still read what we shipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..settings import env
from . import packages, spec

#: Lines one read returns unless the caller asks for more.
DEFAULT_LINES = 200
#: Hard cap on lines per read, so a huge file is always paged.
MAX_LINES = 500
#: Hits one search returns.
MAX_HITS = 40


@dataclass(frozen=True, slots=True)
class Tree:
    """A package's skills tree, wherever it was found."""

    package: str
    root: Path
    source: str  # "cache" or "repo"

    @property
    def version(self) -> str:
        """The package version the tree was generated from."""
        meta = self.root / "api" / "meta.json"
        if not meta.is_file():
            return ""
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("version", "")
        except (OSError, ValueError):
            return ""

    def files(self) -> list[str]:
        """Every markdown file in the tree, relative and sorted."""
        return sorted(
            path.relative_to(self.root).as_posix() for path in self.root.rglob("*.md")
        )


def _is_tree(root: Path) -> bool:
    """Whether ``root`` holds a skills tree: an authored index, or generated api/."""
    return (root / "index.md").is_file() or (root / "api").is_dir()


def _default_index(tree: Tree) -> str:
    """Return the file to read when the caller names none."""
    return "index.md" if (tree.root / "index.md").is_file() else "api/index.md"


def resolve(package: str) -> Tree | None:
    """Return the tree for ``package``: the user cache first, then the repo."""
    cached = spec.cache_dir(package)
    if _is_tree(cached):
        return Tree(package, cached, "cache")
    shipped = spec.tree_dir(package)
    if _is_tree(shipped):
        return Tree(package, shipped, "repo")
    return None


def available() -> list[dict]:
    """Return one row per package that has a tree."""
    names = set(spec.packages_with_skills())
    cache = env.skills_dir()
    if cache.is_dir():
        names |= {entry.name for entry in cache.iterdir() if entry.is_dir()}
    rows = []
    for name in sorted(names):
        tree = resolve(name)
        if tree is None:
            continue
        rows.append(
            {
                "package": name,
                "source": tree.source,
                "version": tree.version,
                "installed": bool(
                    (package := packages.package_for(name)) and packages.installed(package)
                ),
                "files": tree.files(),
            }
        )
    return rows


def search(query: str, package: str = "", limit: int = MAX_HITS) -> dict:
    """Return the lines matching ``query``, across one package or all of them."""
    trees = [tree for tree in (resolve(package),) if tree] if package else [
        tree for name in {row["package"] for row in available()} if (tree := resolve(name))
    ]
    if not trees:
        return {"query": query, "hits": [], "doc": "no skills tree matched that package"}

    needle = query.lower()
    hits: list[dict] = []
    for tree in trees:
        for path in sorted(tree.root.rglob("*.md")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, start=1):
                if needle not in line.lower():
                    continue
                hits.append(
                    {
                        "package": tree.package,
                        "file": path.relative_to(tree.root).as_posix(),
                        "line": number,
                        "text": line.strip()[:200],
                    }
                )
                if len(hits) >= limit:
                    return {"query": query, "hits": hits, "truncated": True}
    return {"query": query, "hits": hits}


def read(
    package: str,
    path: str = "",
    *,
    query: str = "",
    start: int = 0,
    count: int = DEFAULT_LINES,
) -> dict:
    """Read a file from a package's tree, or a window of the lines that match.

    ``path`` defaults to the package's ``index.md``; ``query`` narrows the read
    to matching lines (with a little context) so a caller can ask a question of a
    large file without pulling it all.
    """
    tree = resolve(package)
    if tree is None:
        return {
            "kind": "error",
            "doc": f"no skills tree for {package!r}; known: "
            f"{[row['package'] for row in available()]}",
        }

    target = (tree.root / path) if path else (tree.root / _default_index(tree))
    if not target.is_file() or tree.root not in target.resolve().parents:
        return {
            "kind": "error",
            "doc": f"no such file {path!r} in {package}; files: {tree.files()}",
        }

    all_lines = target.read_text(encoding="utf-8").splitlines()
    windowed = all_lines
    matched: int | None = None
    if query:
        needle = query.lower()
        keep = [number for number, line in enumerate(all_lines) if needle in line.lower()]
        if not keep:
            return {
                "kind": "search",
                "package": package,
                "file": target.relative_to(tree.root).as_posix(),
                "query": query,
                "total_lines": len(all_lines),
                "doc": "no line matched; read the file, or search() across the tree",
            }
        matched = len(keep)
        window: list[str] = []
        for number in keep[: MAX_LINES // 2]:
            window.extend(all_lines[number : number + 3])
        windowed = window

    count = max(1, min(int(count), MAX_LINES))
    start = max(0, int(start))
    page = windowed[start : start + count]
    return {
        "kind": "file",
        "package": package,
        "source": tree.source,
        "version": tree.version,
        "file": target.relative_to(tree.root).as_posix(),
        "start": start,
        "count": len(page),
        "total_lines": len(all_lines),
        "matched_lines": matched,
        "has_more": start + count < len(windowed),
        "text": "\n".join(page),
    }


__all__ = ["DEFAULT_LINES", "MAX_HITS", "MAX_LINES", "Tree", "available", "read", "resolve", "search"]
