"""The GeoLibre layer-style model: keys, aliases, and the two-place write.

GeoLibre's app stores a layer's style twice and resolves them shallowly, with
the project-level copy winning::

    rendered = DEFAULT_LAYER_STYLE | layer.style | project.styles[layerId]

The app rebuilds ``project.styles[layerId]`` from ``layer.style`` on every
serialize, so the project-level entry exists for every layer after the first
round trip through the browser. A writer that touches only ``layer.style`` --
which is what geolibre's own ``Layer.set_style`` and ``authoring.apply_style``
do -- is therefore outranked by the stale project-level copy: the change never
renders, and the app's next echo writes the old resolved value back over
``layer.style``. :func:`store` writes both copies so a change cannot be
shadowed.

``STYLE_KEYS``/``LABEL_KEYS``/``DEFAULT_LABELS`` are derived from
``geolibre.project``'s ``DEFAULT_LAYER_STYLE``/``DEFAULT_LABEL_STYLE``, the
schema the bundled app is built against (GeoLibre 3.10.1). Deriving them rather
than copying them keeps the vocabulary in step with the installed app: a hand
copy silently went stale as upstream added keys. They exist so an unknown key
is a loud failure instead of the silent no-op it is for the app: the earlier
behaviour accepted ``textField``/``textSize``, reported ``applied``, and drew
nothing.
"""

from __future__ import annotations

from difflib import get_close_matches
from typing import Any, Iterable

from geolibre import authoring as geolibre_authoring
from geolibre.project import DEFAULT_LABEL_STYLE, DEFAULT_LAYER_STYLE

from ..contracts.errors import ToolInputError

__all__ = [
    "ALIASES",
    "DEFAULT_LABELS",
    "LABEL_KEYS",
    "STYLE_KEYS",
    "apply_style",
    "effective",
    "mirror",
    "normalize",
    "resolve",
    "style_keys",
    "summary",
]

#: Style keys GeoLibre accepts at the top level of a layer style.
STYLE_KEYS: frozenset[str] = frozenset(DEFAULT_LAYER_STYLE)

#: Keys of the nested ``labels`` object (text drawn on a layer's features).
LABEL_KEYS: frozenset[str] = frozenset(DEFAULT_LABEL_STYLE)

#: The app's default ``labels`` object, used to complete a partial one. The app
#: replaces ``labels`` wholesale (its merge is shallow), so a request that sets
#: only ``field`` must still carry the rest of the object.
DEFAULT_LABELS: dict[str, Any] = dict(DEFAULT_LABEL_STYLE)

#: Tolerated foreign names: MapLibre-style text keys and Leaflet-style path
#: keys, translated onto GeoLibre's own vocabulary.
ALIASES: dict[str, str] = {
    "textField": "labels.field",
    "textSize": "labels.size",
    "textColor": "labels.color",
    "textHaloColor": "labels.haloColor",
    "textHaloWidth": "labels.haloWidth",
    "textAnchor": "labels.anchor",
    "textMaxWidth": "labels.maxWidth",
    "textPlacement": "labels.placement",
    "textTransform": "labels.transform",
    "textDedupe": "labels.dedupe",
    "textAllowOverlap": "labels.allowOverlap",
    "color": "strokeColor",
    "weight": "strokeWidth",
    "radius": "circleRadius",
}

#: Values a caller may pass as ``type`` meaning "draw this layer as text".
_SYMBOL_TYPES = frozenset({"symbol", "text", "label"})

_LABEL_PREFIX = "labels."

#: Keys :func:`summary` reports: the visual state a caller checks after styling.
_SUMMARY_KEYS = (
    "fillColor",
    "strokeColor",
    "strokeWidth",
    "fillOpacity",
    "circleRadius",
    "pointRenderer",
)
_SUMMARY_LABEL_KEYS = ("enabled", "field", "size", "color", "haloColor", "haloWidth", "anchor")


def style_keys() -> dict[str, Any]:
    """Return the accepted style vocabulary, for discovery by a caller/model."""
    return {
        "style": sorted(STYLE_KEYS),
        "labels": sorted(LABEL_KEYS),
        "aliases": dict(sorted(ALIASES.items())),
        "labelExample": {"labels": {"enabled": True, "field": "<feature property>"}},
    }


def resolve(project: dict[str, Any], ref: str) -> dict[str, Any]:
    """Resolve a layer by id or display name, as a :class:`ToolInputError`."""
    try:
        return geolibre_authoring.find_layer(project, ref)
    except ValueError as exc:
        raise ToolInputError(str(exc)) from exc


def normalize(style: dict[str, Any]) -> dict[str, Any]:
    """Translate aliases in ``style`` and reject keys GeoLibre would ignore.

    Text-label keys are folded into the nested ``labels`` object, completed
    against :data:`DEFAULT_LABELS`; a ``field`` with no explicit ``enabled``
    turns labels on, since a field is only ever supplied to draw it.

    Raises:
        ToolInputError: If ``style`` is not a mapping, or names a key the app
            does not know (a silent no-op for the app, so it fails here).
    """
    if not isinstance(style, dict):
        raise ToolInputError(f"style must be an object of style keys, got {type(style).__name__}")
    normalized: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in style.items():
        if key == "labels":
            nested = _normalize_labels(value)
            labels.update(nested)
        elif key == "type":
            if str(value).lower() in _SYMBOL_TYPES:
                labels["enabled"] = True
            else:
                unknown.append(key)
        elif key in ALIASES:
            target = ALIASES[key]
            if target.startswith(_LABEL_PREFIX):
                labels[target[len(_LABEL_PREFIX) :]] = value
            else:
                normalized[target] = value
        elif key in STYLE_KEYS:
            normalized[key] = value
        else:
            unknown.append(key)
    if labels:
        complete = {**DEFAULT_LABELS, **labels}
        if labels.get("field") and "enabled" not in labels:
            complete["enabled"] = True
        normalized["labels"] = complete
    if unknown:
        raise ToolInputError(_unknown_keys_message(unknown))
    return normalized


def effective(project: dict[str, Any], layer: dict[str, Any]) -> dict[str, Any]:
    """Return the style GeoLibre renders for ``layer``.

    Mirrors the app's shallow resolution order exactly: defaults, then the
    layer's own style, then the project-level copy keyed by layer id.
    """
    merged = dict(DEFAULT_LAYER_STYLE)
    own = layer.get("style")
    if isinstance(own, dict):
        merged.update(own)
    stored = _styles(project).get(str(layer.get("id")))
    if isinstance(stored, dict):
        merged.update(stored)
    return merged


def store(project: dict[str, Any], layer: dict[str, Any], style: dict[str, Any]) -> None:
    """Write ``style`` to both places GeoLibre reads, so nothing shadows it."""
    layer["style"] = dict(style)
    _styles(project)[str(layer.get("id"))] = dict(style)


def apply_style(
    project: dict[str, Any], ref: str, changes: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge normalized ``changes`` onto a layer's rendered style.

    The merge is applied to :func:`effective`, so a request that names only its
    own keys keeps whatever the app would have drawn for the rest, and a style
    previously shadowed by a stale project-level copy is repaired.

    Returns:
        The resolved layer record and its complete style after the merge.
    """
    layer = resolve(project, ref)
    merged = effective(project, layer)
    merged.update(normalize(changes))
    store(project, layer, merged)
    return layer, merged


def mirror(project: dict[str, Any], ref: str) -> dict[str, Any]:
    """Copy a layer's rendered style into ``project.styles``.

    Used after a geolibre authoring call (e.g. ``add_gdf(column=...)``) has set
    ``layer.style`` directly, so the app resolves the same values.

    Returns:
        The layer's style after the mirror.
    """
    layer = resolve(project, ref)
    style = effective(project, layer)
    store(project, layer, style)
    return style


def summary(project: dict[str, Any], layer: dict[str, Any]) -> dict[str, Any]:
    """Return the layer's rendered style, compacted for a tool result."""
    style = effective(project, layer)
    raw_labels = style.get("labels")
    labels = {**DEFAULT_LABELS, **(raw_labels if isinstance(raw_labels, dict) else {})}
    report: dict[str, Any] = {key: style.get(key) for key in _SUMMARY_KEYS}
    report["labels"] = {key: labels.get(key) for key in _SUMMARY_LABEL_KEYS}
    if style.get("vectorStyleMode") not in (None, "single"):
        report["vectorStyleMode"] = style.get("vectorStyleMode")
        report["vectorStyleProperty"] = style.get("vectorStyleProperty")
    return report


# -- internals -------------------------------------------------------------


def _styles(project: dict[str, Any]) -> dict[str, Any]:
    """Return the project's ``styles`` map, creating it when absent."""
    styles = project.get("styles")
    if not isinstance(styles, dict):
        styles = {}
        project["styles"] = styles
    return styles


def _normalize_labels(value: Any) -> dict[str, Any]:
    """Validate a nested ``labels`` object."""
    if not isinstance(value, dict):
        raise ToolInputError(f"labels must be an object of label keys, got {type(value).__name__}")
    unknown = [key for key in value if key not in LABEL_KEYS]
    if unknown:
        raise ToolInputError(_unknown_keys_message(unknown, labels=True))
    return dict(value)


def _unknown_keys_message(unknown: Iterable[str], *, labels: bool = False) -> str:
    """Name the rejected keys, near misses, and how to list the whole schema."""
    keys = sorted(unknown)
    known = LABEL_KEYS if labels else STYLE_KEYS
    scope = "label" if labels else "style"
    near = {
        key: matches
        for key in keys
        if (matches := get_close_matches(key, sorted(known), n=2, cutoff=0.6))
    }
    hint = f" Closest {scope} keys: {near}." if near else ""
    return (
        f"unknown {scope} keys: {', '.join(repr(key) for key in keys)}."
        f" Call list_style_keys for the full vocabulary.{hint}"
        ' Text labels use {"labels": {"enabled": true, "field": "<feature property>"}}.'
        " GeoLibre ignores style keys it does not know, so nothing changed."
    )
