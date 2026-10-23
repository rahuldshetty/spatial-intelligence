"""Tool declarations: what a tool *is*, separately from what it does.

Three things live here:

* :class:`ToolSpec` — the resolved, fully-populated declaration the registry
  hands to the agent and session layers.
* :class:`ToolDeclaration` — the partial record ``@tool`` attaches to a
  function, whose unset fields fall back to the enclosing ``@pack``.
* :func:`tool` / :func:`pack` — the decorators tools are declared with.

The spec is the single source of truth for tool metadata. Nothing else may keep
a list of tool names: core visibility, replay-safety, approval, timeouts,
serialization, and the capability index all read from it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, TypeVar

from ..contracts.effects import MUTATING_EFFECTS, READ_ONLY, Effect

T = TypeVar("T", bound=type)

#: Attribute ``@tool`` attaches to a function.
DECLARATION_ATTR = "__si_tool__"
#: Attribute ``@pack`` attaches to a class.
PACK_ATTR = "__si_pack__"


class ToolKind(StrEnum):
    """How a tool behaves during a run."""

    SYNC = "sync"
    """Runs to completion and returns a value."""

    REPORTING = "reporting"
    """Emits :class:`~spatial_intelligence.contracts.progress.ProgressEvent` while running."""

    INTERACTIVE = "interactive"
    """Defers the run and resumes with structured user answers."""

    HANDOFF = "handoff"
    """Advertised capability with no callable implementation (prose handoff)."""


def _first(*values: Any) -> Any:
    """Return the first non-``None`` value."""
    for value in values:
        if value is not None:
            return value
    return None


def summary_of(fn: Callable[..., Any]) -> str:
    """Return a one-line registry summary from a function's docstring."""
    doc = inspect.getdoc(fn) or ""
    first = doc.strip().split("\n\n", 1)[0].strip()
    return " ".join(first.split()) or getattr(fn, "__name__", "tool").replace("_", " ")


@dataclass(frozen=True, slots=True)
class PackDefaults:
    """Fallbacks a ``@pack`` supplies to the ``@tool`` methods it contains."""

    category: str | None = None
    effects: frozenset[Effect] | None = None
    kind: ToolKind | None = None
    core: bool | None = None
    requires_approval: bool | None = None
    sequential: bool | None = None
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A resolved tool declaration plus its implementation.

    Attributes:
        name: Model-facing tool name.
        summary: One-line registry summary (the model-facing description stays
            the docstring, which pydantic-ai parses itself).
        category: Grouping used by discovery and by the runner (``plan`` marks
            tools the harness registers, without the runner hardcoding names).
        fn: The callable the agent invokes; ``None`` for declarative entries.
        effects: What the tool may touch; replay-safety derives from this.
        kind: How the tool behaves during a run.
        core: Always visible in the model request (no deferred loading).
        requires_approval: Runnable only when the user granted approval.
        timeout: Per-tool timeout in seconds.
        sequential: Tool calls must not run concurrently with others.
        capabilities: Discovery tags (e.g. ``"catalog"``, ``"sar"``).
        origin: Where the tool comes from (``builtin``, ``harness``, ``toolsearch``).
    """

    name: str
    summary: str
    category: str
    fn: Callable[..., Any] | None = None
    effects: frozenset[Effect] = READ_ONLY
    kind: ToolKind = ToolKind.SYNC
    core: bool = False
    requires_approval: bool = False
    timeout: float | None = None
    sequential: bool = False
    capabilities: tuple[str, ...] = ()
    origin: str = "builtin"

    @property
    def callable(self) -> Callable[..., Any]:
        """Return the implementation, raising when the spec is declarative."""
        if self.fn is None:
            raise TypeError(f"tool {self.name!r} has no implementation")
        return self.fn

    @property
    def implemented(self) -> bool:
        """Whether this spec carries a callable (vs. a foreign/handoff entry)."""
        return self.fn is not None

    @property
    def replay_safe(self) -> bool:
        """Whether a failed run may be replayed after this tool completed."""
        return not (self.effects & MUTATING_EFFECTS)

    @property
    def approval_effects(self) -> frozenset[Effect]:
        """Effects that cannot run without approval."""
        from ..contracts.effects import APPROVAL_EFFECTS

        return self.effects & APPROVAL_EFFECTS

    def bind(self, fn: Callable[..., Any]) -> "ToolSpec":
        """Return this spec with a different implementation (a bound method)."""
        return replace(self, fn=fn)

    def model_metadata(self) -> dict[str, Any]:
        """Return the JSON-safe metadata carried on the pydantic-ai tool.

        Kept to strings and lists: tool definitions travel to the model.
        """
        return {
            "si_tool": self.name,
            "si_category": self.category,
            "si_effects": sorted(effect.value for effect in self.effects),
            "si_kind": self.kind.value,
        }


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    """The partial record ``@tool`` attaches; unset fields inherit from the pack."""

    name: str | None = None
    category: str | None = None
    summary: str | None = None
    effects: frozenset[Effect] | None = None
    kind: ToolKind | None = None
    core: bool | None = None
    requires_approval: bool | None = None
    timeout: float | None = None
    sequential: bool | None = None
    capabilities: tuple[str, ...] = ()
    origin: str = "builtin"

    def resolve(
        self, fn: Callable[..., Any], *, pack: PackDefaults | None = None
    ) -> ToolSpec:
        """Combine this declaration, its pack defaults, and package defaults."""
        defaults = pack or PackDefaults()
        name = self.name or getattr(fn, "__name__", "")
        if not name:
            raise TypeError("a tool needs a name; pass name=... to @tool")
        category = _first(self.category, defaults.category)
        if not category:
            raise TypeError(
                f"tool {name!r} has no category; pass category=... to @tool or @pack"
            )
        effects = _first(self.effects, defaults.effects)
        effects = READ_ONLY if effects is None else frozenset(effects)
        return ToolSpec(
            name=name,
            summary=self.summary or summary_of(fn),
            category=category,
            fn=fn,
            effects=effects,
            kind=_first(self.kind, defaults.kind, ToolKind.SYNC),
            core=bool(_first(self.core, defaults.core, False)),
            requires_approval=bool(
                _first(self.requires_approval, defaults.requires_approval, False)
            ),
            timeout=self.timeout,
            # Map mutations are read-copy-reassign operations on one shared
            # project. Running two in parallel (pydantic-ai does by default)
            # lets one overwrite the other's layer, so they are barriers even
            # unless a pack explicitly says otherwise.
            sequential=bool(_first(self.sequential, defaults.sequential, False))
            or Effect.MAP_WRITE in effects,
            capabilities=tuple(defaults.capabilities) + tuple(self.capabilities),
            origin=self.origin,
        )


def tool(
    *,
    name: str | None = None,
    category: str | None = None,
    summary: str | None = None,
    effects: frozenset[Effect] | None = None,
    kind: ToolKind | None = None,
    core: bool | None = None,
    requires_approval: bool | None = None,
    timeout: float | None = None,
    sequential: bool | None = None,
    capabilities: tuple[str, ...] = (),
    origin: str = "builtin",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare a tool: a module-level function or a ``@pack`` method.

    Unset fields inherit from the enclosing pack, then from package defaults
    (read-only, synchronous, deferred, no approval, no timeout).
    """
    declaration = ToolDeclaration(
        name=name,
        category=category,
        summary=summary,
        effects=effects,
        kind=kind,
        core=core,
        requires_approval=requires_approval,
        timeout=timeout,
        sequential=sequential,
        capabilities=capabilities,
        origin=origin,
    )

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, DECLARATION_ATTR, declaration)
        return fn

    return decorate


def pack(
    *,
    category: str | None = None,
    effects: frozenset[Effect] | None = None,
    kind: ToolKind | None = None,
    core: bool | None = None,
    requires_approval: bool | None = None,
    sequential: bool | None = None,
    capabilities: tuple[str, ...] = (),
) -> Callable[[T], T]:
    """Declare a tool pack: a class whose ``@tool`` methods are its tools.

    The class takes one constructor argument, the :class:`ToolRuntime`, and is
    instantiated once per session by the registry. Class-level defaults are
    inherited by every method that does not override them, so a pack of
    ``results/`` writers states its effects and category once.
    """
    defaults = PackDefaults(
        category=category,
        effects=effects,
        kind=kind,
        core=core,
        requires_approval=requires_approval,
        sequential=sequential,
        capabilities=capabilities,
    )

    def decorate(cls: T) -> T:
        setattr(cls, PACK_ATTR, defaults)
        return cls

    return decorate


def declaration_of(fn: Callable[..., Any]) -> ToolDeclaration | None:
    """Return the declaration attached to ``fn``, if any."""
    return getattr(fn, DECLARATION_ATTR, None)


def pack_defaults_of(cls: type) -> PackDefaults:
    """Return a class's pack defaults, raising when it is not a ``@pack``."""
    defaults = getattr(cls, PACK_ATTR, None)
    if defaults is None:
        raise TypeError(f"{cls.__name__} is not declared with @pack")
    return defaults


def declared_methods(cls: type) -> list[tuple[str, ToolDeclaration]]:
    """Return ``(attribute, declaration)`` for each decorated method, in order."""
    found: list[tuple[str, ToolDeclaration]] = []
    for attribute, value in vars(cls).items():
        declaration = getattr(value, DECLARATION_ATTR, None)
        if isinstance(declaration, ToolDeclaration):
            found.append((attribute, declaration))
    return found


def declared_functions(module: Any) -> list[tuple[str, ToolDeclaration]]:
    """Return ``(name, declaration)`` for each decorated module-level function."""
    found: list[tuple[str, ToolDeclaration]] = []
    for attribute, value in vars(module).items():
        if not inspect.isfunction(value):
            continue
        declaration = getattr(value, DECLARATION_ATTR, None)
        if isinstance(declaration, ToolDeclaration):
            found.append((attribute, declaration))
    return found
