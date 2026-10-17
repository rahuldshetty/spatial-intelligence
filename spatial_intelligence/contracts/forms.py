"""The structured-form schema shared by agent questions and workflow wizards.

Two surfaces ask the user for the same kind of thing: ``request_user_input``
defers a run with a list of fields, and a workflow's wizard asks for the
parameters that render its prompt. Both send this model to the browser, so
there is one renderer and one answer shape instead of two dialects.

Layering: ``contracts`` imports nothing else in the package.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

#: Field types the browser knows how to render.
FieldType = Literal[
    "radio", "multi_select", "text", "textarea", "confirmation", "aoi", "model"
]


class ChoiceOption(BaseModel):
    """One selectable option on a ``radio`` or ``multi_select`` field."""

    value: str
    label: str
    description: str | None = None
    recommended: bool = False
    thumbnail_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InteractionField(BaseModel):
    """One question: what to ask, how to render it, and what it defaults to.

    ``aoi`` is the composite area-of-interest field: the browser fills it from a
    drawn shape, the live map view, a typed bounding box, or a place name, and
    every route ends in the same ``{"mode", "bbox", "geojson", "label"}`` value.
    """

    id: str
    label: str
    type: FieldType
    description: str | None = None
    required: bool = True
    options: list[ChoiceOption] = Field(default_factory=list)
    default: Any = None
    placeholder: str | None = None
    #: The task a ``model`` field must be able to do ("segmentation", ...). The
    #: wizard shows it as the question a candidate model has to pass.
    task: str | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.type in {"radio", "multi_select"} and not self.options:
            raise ValueError(f"{self.type} field {self.id!r} requires options")
        if self.type == "model" and not self.task and not self.options:
            raise ValueError(f"model field {self.id!r} needs a task or some options")
        return self


__all__ = ["ChoiceOption", "FieldType", "InteractionField"]
