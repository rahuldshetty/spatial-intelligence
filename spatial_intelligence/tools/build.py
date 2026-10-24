"""The registry manifest: every tool the agent can call, in declaration order.

This module is the one place that decides *which* tools exist and in what order
they are handed to the model. Order matters for comprehension (routing and
interaction first, then data acquisition, then map, then processing), so it is
stated here rather than derived from import order.

Tools implemented outside this package — the harness planning toolset and the
tool-search fallback — are registered as external specs. They carry no callable
(the owning capability registers the implementation on the agent) but they must
be classified here, because replay-safety and plan detection depend on knowing
about them.
"""

from __future__ import annotations

from ..contracts.effects import READ_ONLY, Effect
from ..contracts.services import VISION_SERVICE
from .packs.ai import AIPack
from .packs.capabilities import CapabilityPack
from .packs.catalog import CatalogPack
from .packs.detection import DetectionPack
from .packs.files import FilesPack
from .packs.interaction import InteractionPack
from .packs.layers import LayersPack
from .packs.python_ import PythonPack
from .packs.raster import RasterPack
from .packs.segmentation import SegmentationPack
from .packs.skills import SkillsPack
from .packs.vector import VectorPack
from .packs.vision import VisionPack
from .registry import PLAN_MUTATION_TAG, ToolRegistry
from .runtime import ToolRuntime
from .spec import ToolSpec, declared_methods, pack_defaults_of

#: Plan tools registered by ``pydantic_ai_harness.Planning``. They are declared
#: here because the runner must recognize a plan-changing tool call to snapshot
#: the plan into the trace; ``tests/spatial_intelligence/test_registry_manifest.py``
#: asserts these names still match the installed harness.
PLAN_TOOL_NAMES: tuple[str, ...] = (
    "write_plan",
    "read_plan",
    "add_task",
    "update_task_status",
    "update_task_statuses",
    "remove_task",
)

#: The subset of plan tools that changes the plan (the rest are reads).
PLAN_MUTATION_NAMES: frozenset[str] = frozenset(
    {
        "write_plan",
        "add_task",
        "update_task_status",
        "update_task_statuses",
        "remove_task",
    }
)

#: The function tool the auto-injected tool-search capability exposes locally.
TOOL_SEARCH_NAME = "search_tools"

#: The local function tool the ``WebSearch`` capability falls back to. The
#: capability registers the implementation on the agent; this name only
#: classifies it here (replay-safety, discovery, category).
WEB_SEARCH_NAME = "web_search"

_PLAN_SUMMARIES = {
    "write_plan": "Create or replace the task plan (whole-list replacement).",
    "read_plan": "Read the current task plan with step ids and a progress summary.",
    "add_task": "Append one pending step to the task plan.",
    "update_task_status": "Move one plan step to a new status by id.",
    "update_task_statuses": "Apply several plan step status changes atomically.",
    "remove_task": "Remove one plan step by id.",
}


#: Packs in the order the model should read them. ``VisionPack`` is skipped
#: when the endpoint cannot accept images.
PACK_ORDER: tuple[type, ...] = (
    # Routing and human-in-the-loop first: these decide what the rest of the
    # run looks like.
    InteractionPack,
    CapabilityPack,
    # Orientation and data acquisition.
    FilesPack,
    CatalogPack,
    # Map authoring, then image understanding when the model supports it.
    LayersPack,
    VisionPack,
    # Processing.
    RasterPack,
    VectorPack,
    # Local models: the cache and its memory first, then the tasks that use them.
    AIPack,
    SegmentationPack,
    DetectionPack,
    PythonPack,
    SkillsPack,
)


def default_registry(runtime: ToolRuntime) -> ToolRegistry:
    """Build the session's registry: every pack, then the external tools.

    Packs are stated in the order the model should read them. Any pack raising
    on construction (a duplicate tool name, a bad declaration) fails the
    workspace open rather than silently dropping a tool.
    """
    registry = ToolRegistry()
    vision = runtime.services.get(VISION_SERVICE)
    vision_supported = vision is not None and getattr(vision, "supported", False)
    for pack in PACK_ORDER:
        if pack is VisionPack and not vision_supported:
            continue
        registry.add_pack(pack, runtime)
    register_external_tools(registry)
    return registry


def declared_specs() -> list[ToolSpec]:
    """Return every declared tool without instantiating a pack.

    Tool Configuration needs the complete catalog even before a workspace is
    open, when no runtime exists to construct a pack against. The declarations
    carry the same name, summary, category, core flag, and origin the registry
    would use; ``fn`` is the unbound method, which is enough for display and
    never called. ``VisionPack`` is included here unconditionally and marked
    unavailable by the catalog payload when the model cannot accept images.
    """
    specs: list[ToolSpec] = []
    for pack in PACK_ORDER:
        defaults = pack_defaults_of(pack)
        for attribute, declaration in declared_methods(pack):
            specs.append(declaration.resolve(getattr(pack, attribute), pack=defaults))
    specs.extend(external_specs())
    return specs


def external_specs() -> list[ToolSpec]:
    """Return the declarative specs for tools other capabilities register."""
    specs: list[ToolSpec] = []
    for name in PLAN_TOOL_NAMES:
        capabilities = (PLAN_MUTATION_TAG,) if name in PLAN_MUTATION_NAMES else ()
        specs.append(
            ToolSpec(
                name=name,
                category="plan",
                origin="harness",
                summary=_PLAN_SUMMARIES.get(name, name.replace("_", " ")),
                effects=READ_ONLY,
                capabilities=capabilities,
            )
        )
    specs.append(
        ToolSpec(
            name=TOOL_SEARCH_NAME,
            category="search",
            origin="toolsearch",
            summary="Search the deferred tools by name and description.",
            effects=READ_ONLY,
        )
    )
    specs.append(
        ToolSpec(
            name=WEB_SEARCH_NAME,
            category="web",
            origin="capability",
            summary="Search the web for documentation, provider terms, or current facts.",
            effects=frozenset({Effect.READ, Effect.NETWORK}),
        )
    )
    return specs


def register_external_tools(registry: ToolRegistry) -> list[str]:
    """Classify the tools other capabilities register on the agent."""
    specs = external_specs()
    registry.add_all(specs)
    return [spec.name for spec in specs]
