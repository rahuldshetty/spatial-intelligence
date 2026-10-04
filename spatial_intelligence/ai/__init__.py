"""Local AI models: what this build can run, and the memory it holds.

Three modules, one job each:

- :mod:`~spatial_intelligence.ai.models` — the catalog and the on-disk store
  (``<GEOAI_HOME>/.models``) that feeds it.
- :mod:`~spatial_intelligence.ai.manager` — residency: load on first use, keep
  at most a couple of models in memory, unload what is idle.
- :mod:`~spatial_intelligence.ai.segmentation` — the first task built on top:
  SAM-family promptable segmentation, run locally through ONNX Runtime.

Nothing here imports ``onnxruntime`` at module scope: the package must import
(and the app must start) on an install that never asked for the ``ai`` extra.
"""
