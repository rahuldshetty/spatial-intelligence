# Skills

API references the agent reads on demand — `skill()` serves them, nothing here
is loaded into the prompt.

## Layout

```
skills/<package>/index.md      authored: what it is, docs, inference usage, pitfalls
skills/<package>/api/*.md      GENERATED: one file per topic, plus *.catalog.md for bulk topics
skills/<package>/api/meta.json GENERATED: package version, generator, sha256 per file
skills/models/*.md             authored: checkpoints, what bands they expect, what they predict
```

The `models/` tree is deliberately separate: package skills describe how a
library works, model skills describe which weights to fetch and what they need.

Rules:

- Everything outside `api/` is hand-written and reviewed like code — `index.md`
  per package, the `models/` tree, this file.
- Everything under `api/` is generated — never edit it; run the generator instead.
- `api/meta.json` records the package version the tree describes. When the
  installed version differs, regenerate; `skill()` reports the mismatch rather
  than serving a stale answer as current.
- One topic per file, capped at ~32 KB, so a package with hundreds of modules
  never becomes one document.

## Regenerating

```sh
python -m spatial_intelligence.pythonruntime.spec torchgeo terratorch
```
Writes the committed tree when the directory is writable and the package is
installed; falls back to `<GEOAI_HOME>/.skills/` otherwise. Commit the diff —
it is how an API change becomes reviewable.

A user can override or extend a package without touching the repo by dropping
files in `<GEOAI_HOME>/.skills/<package>/`; the cache is read first.
