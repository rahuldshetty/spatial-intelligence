"""Local AI models: what this build can run, and the memory it holds.

Two layers, deliberately separate:

- the model layer — :mod:`~spatial_intelligence.ai.catalog` (which models exist,
  pinned by revision and hash), :mod:`~spatial_intelligence.ai.store` (fetching
  and verifying them under ``<GEOAI_HOME>/.models``), and
  :mod:`~spatial_intelligence.ai.manager` (what is loaded, and what is given
  back when memory is needed).
- the task layer — one module per problem: :mod:`~spatial_intelligence.ai.segmentation`
  answers "where does this object end" with prompts and masks,
  :mod:`~spatial_intelligence.ai.detection` answers "what known objects are
  here" with boxes. They share only
  :mod:`~spatial_intelligence.ai.tiles`, which is how a raster too big for one
  model pass is read.

Adding a task means a new module and a new tool pack; adding a model for an
existing task means a catalog entry.

Nothing here imports ``onnxruntime`` at module scope: the package must import
(and the app must start) on an install that never asked for the ``ai`` extra.
"""
