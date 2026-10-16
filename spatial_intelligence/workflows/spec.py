"""Workflow declarations: the wizard's fields and the prompt they render into.

A workflow is configuration, not an engine. It names the fields the wizard asks
for, the prompt those answers fill in, and the tools that prompt expects. The
seeded prompt cell is what runs, so the agent, the run queue, traces, and
progress stay exactly the code that already exists.

Layering: ``workflows`` sits with the services (it may read ``contracts``,
``ai``, ``geo``, ``pythonruntime``, and ``workspace``); it never imports
``tools`` or ``agent``, and it refers to tools by name in ``requires``, which a
test checks against the registry.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from ..contracts.forms import InteractionField


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """One wizard: its fields, its prompt template, and what the prompt needs."""

    id: str
    label: str
    summary: str
    params: tuple[InteractionField, ...] = ()
    prompt: str = ""
    #: Tool names the rendered prompt is expected to use. Checked by a test, so
    #: a template cannot quietly name a tool that no longer exists.
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    @property
    def param_ids(self) -> tuple[str, ...]:
        """Field ids, in declaration order."""
        return tuple(param.id for param in self.params)

    @property
    def placeholders(self) -> set[str]:
        """Field names the prompt template interpolates."""
        return {name for _, name, _, _ in string.Formatter().parse(self.prompt) if name}

    def param(self, field_id: str) -> InteractionField | None:
        """Return the field with this id, or ``None``."""
        for param in self.params:
            if param.id == field_id:
                return param
        return None

    def validate(self) -> None:
        """Fail loudly on a template the wizard could not fill."""
        if not self.id or not self.label:
            raise ValueError("a workflow needs an id and a label")
        ids = self.param_ids
        if len(set(ids)) != len(ids):
            raise ValueError(f"workflow {self.id!r} declares a duplicate field id")
        unknown = self.placeholders - set(ids)
        if unknown:
            raise ValueError(
                f"workflow {self.id!r} prompt uses undeclared fields: {sorted(unknown)}"
            )
        if self.prompt and not self.params:
            raise ValueError(f"workflow {self.id!r} has a prompt but no fields")


__all__ = ["WorkflowSpec"]
