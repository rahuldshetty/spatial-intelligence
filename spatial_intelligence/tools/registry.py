"""The registry: one ordered manifest of every tool the agent can call.

Replaces the three hand-maintained name lists that used to live beside the
tools (the registered list, the always-visible list, and the replay-safe list):
those queries are now derived from :class:`~spatial_intelligence.tools.spec.ToolSpec`
metadata, and foreign tools (the planning toolset, tool search) are registered
as external specs so replay-safety covers them too.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import ModuleType
from typing import Any

from ..contracts.effects import READ_ONLY, Effect
from .runtime import ToolRuntime
from .spec import (
    ToolKind,
    ToolSpec,
    declared_functions,
    declared_methods,
    pack_defaults_of,
)

#: Capability tag marking the harness plan tools that change the plan.
PLAN_MUTATION_TAG = "plan_mutation"


class ToolRegistry:
    """An ordered collection of tool specs plus the queries other layers need."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._order: list[str] = []

    # -- registration ----------------------------------------------------

    def add(self, spec: ToolSpec) -> ToolSpec:
        """Register one spec; tool names are unique across the registry."""
        if spec.name in self._specs:
            existing = self._specs[spec.name]
            raise ValueError(
                f"tool {spec.name!r} is already registered by {existing.origin!r}; "
                f"cannot register it again from {spec.origin!r}"
            )
        self._specs[spec.name] = spec
        self._order.append(spec.name)
        return spec

    def add_all(self, specs: Iterable[ToolSpec]) -> list[ToolSpec]:
        """Register several specs, returning them in order."""
        return [self.add(spec) for spec in specs]

    def remove(self, name: str) -> None:
        """Drop one spec (a capability that is no longer available)."""
        if name not in self._specs:
            return
        del self._specs[name]
        self._order.remove(name)

    def add_module(self, module: ModuleType) -> list[ToolSpec]:
        """Register every ``@tool`` function declared at module level."""
        added = [
            self.add(declaration.resolve(getattr(module, attribute)))
            for attribute, declaration in declared_functions(module)
        ]
        return added

    def add_pack(self, cls: type, runtime: ToolRuntime) -> list[ToolSpec]:
        """Instantiate a ``@pack`` class and register its ``@tool`` methods.

        The instance is created once and keeps the runtime for the whole
        session, which is how a pack composes services the way a class would.
        """
        defaults = pack_defaults_of(cls)
        instance = cls(runtime)
        added: list[ToolSpec] = []
        for attribute, declaration in declared_methods(cls):
            bound = getattr(instance, attribute)
            added.append(self.add(declaration.resolve(bound, pack=defaults)))
        return added

    def add_external(
        self,
        name: str,
        *,
        category: str,
        origin: str,
        summary: str = "",
        effects: frozenset[Effect] = READ_ONLY,
        kind: ToolKind = ToolKind.SYNC,
        core: bool = False,
        capabilities: tuple[str, ...] = (),
    ) -> ToolSpec:
        """Register a tool implemented elsewhere (a harness or capability toolset).

        These carry no callable: the owning capability registers the
        implementation on the agent itself. They exist so replay-safety,
        categories, and discovery cover the *whole* tool surface rather than
        only the tools this package declares.
        """
        return self.add(
            ToolSpec(
                name=name,
                summary=summary or name.replace("_", " "),
                category=category,
                fn=None,
                effects=effects,
                kind=kind,
                core=core,
                capabilities=capabilities,
                origin=origin,
            )
        )

    # -- queries ---------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self.specs())

    def get(self, name: str) -> ToolSpec:
        """Return the spec for ``name``, raising ``KeyError`` when unknown."""
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}") from None

    def specs(self) -> list[ToolSpec]:
        """Every spec, in registration order."""
        return [self._specs[name] for name in self._order]

    def names(self) -> list[str]:
        """Every registered tool name, in registration order."""
        return list(self._order)

    def implemented(self) -> list[ToolSpec]:
        """Specs with a callable of our own to register on the agent."""
        return [spec for spec in self.specs() if spec.implemented]

    def core_names(self) -> frozenset[str]:
        """Tools kept visible in every model request (no deferred loading)."""
        return frozenset(spec.name for spec in self.specs() if spec.core)

    def categories(self) -> dict[str, tuple[str, ...]]:
        """Registered tools grouped by category."""
        grouped: dict[str, list[str]] = {}
        for spec in self.specs():
            grouped.setdefault(spec.category, []).append(spec.name)
        return {category: tuple(names) for category, names in grouped.items()}

    def category_of(self, name: str | None) -> str | None:
        """Return a tool's category, or ``None`` when it is not registered."""
        if name is None:
            return None
        spec = self._specs.get(name)
        return spec.category if spec is not None else None

    def replay_safe(self, name: str | None) -> bool:
        """Whether a failed run may be replayed after ``name`` completed.

        Unknown tools are treated as state-changing: a tool this registry has
        never seen cannot be assumed harmless.
        """
        if name is None:
            return False
        spec = self._specs.get(name)
        return spec.replay_safe if spec is not None else False

    def requires_approval(self, name: str) -> bool:
        """Whether ``name`` runs only with the user's approval."""
        return self.get(name).requires_approval

    def kind_of(self, name: str) -> ToolKind:
        """Return how ``name`` behaves during a run."""
        return self.get(name).kind

    def by_category(self, category: str) -> list[ToolSpec]:
        """Return every spec in ``category``, in registration order."""
        return [spec for spec in self.specs() if spec.category == category]

    def with_capability(self, tag: str) -> list[ToolSpec]:
        """Return every spec tagged with ``tag``."""
        return [spec for spec in self.specs() if tag in spec.capabilities]

    def interactive(self) -> list[ToolSpec]:
        """Handoff tools that defer the run for structured user input."""
        return [spec for spec in self.specs() if spec.kind is ToolKind.INTERACTIVE]

    def plan_mutation(self, name: str | None) -> bool:
        """Whether ``name`` changes the agent's task plan.

        The plan tools come from the harness, so they are registered as
        external specs tagged ``plan_mutation``. They stay *replay-safe* because
        the runner restores the plan snapshot before retrying, which is why
        this is a separate question from :meth:`replay_safe`.
        """
        if name is None:
            return False
        spec = self._specs.get(name)
        return spec is not None and PLAN_MUTATION_TAG in spec.capabilities

    def handoffs(self) -> list[ToolSpec]:
        """Capabilities the agent must hand back to the user, prose-first."""
        return [spec for spec in self.specs() if spec.kind is ToolKind.HANDOFF]

    # -- agent integration -----------------------------------------------

    def build(self, agent: Any) -> None:
        """Register every implemented spec on a pydantic-ai ``Agent``.

        ``requires_approval`` is deliberately *not* forwarded to pydantic-ai:
        approval is enforced in-tool against the runtime's ``approved`` flag,
        which the existing Settings toggle already drives. Forwarding it would
        instead turn the call into a deferred request this UI has no form for.
        """
        for spec in self.implemented():
            agent.tool_plain(
                spec.fn,
                name=spec.name,
                defer_loading=not spec.core,
                timeout=spec.timeout,
                sequential=spec.sequential,
                metadata=spec.model_metadata(),
            )
