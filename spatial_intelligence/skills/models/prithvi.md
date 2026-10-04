# Prithvi-EO

NASA/IBM Earth-observation foundation models (HLS imagery), used through
terratorch. Two kinds of checkpoint: a **backbone** you attach a head to, and a
**fine-tuned head** that already answers a specific question.

## Checkpoints

| Repo (`ibm-nasa-geospatial/…`) | What it is | Predicts |
|---|---|---|
| `Prithvi-EO-2.0-300M`, `-600M`, `-tiny-TL` | backbone, Apache-2.0 | features for your own head |
| `Prithvi-EO-2.0-300M-TL` | backbone tuned for transfer | features, usually a better starting point |
| `Prithvi-EO-2.0-300M-TL-Sen1Floods11` | segmentation head | water / flood, 2 classes |
| `Prithvi-EO-2.0-300M-BurnScars` | segmentation head | burn scar |
| `Prithvi-EO-1.0-100M-sen1floods11`, `-burn-scar`, `-multi-temporal-crop-classification` | the 1.0 generation | floods, burn scars, crops |

## Multi-temporal crop classification

`ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification`
(1.7 GB) classifies 13 crop and land-cover classes from **three** Sentinel-2/HLS
dates in one pass:

- **Input**: one 224×224 px chip per timestep, six bands each, ordered Blue,
  Green, Red, Narrow NIR (`B08A` on Sentinel-2), SWIR 1 (`B11`), SWIR 2 (`B12`).
  The model's own pipeline permutes to channels-first and reshapes to
  `(6, 3, 224, 224)`, so the frame axis sits between channels and height; batch
  as `(B, 6, 3, 224, 224)`.
- **Normalization is part of the checkpoint**: the config carries per-band
  `means` 494.9, 815.2, 924.3, 2968.9, 2634.6, 1739.6 and `stds` 284.9, 357.8,
  575.6, 896.6, 951.9, 921.4, repeated for each timestep. Feed reflectance in
  that scale — not raw DN, not 0–1.
- **Classes** (US CDL names, ids 1-based in this order): Natural Vegetation,
  Forest, Corn, Soybeans, Wetlands, Developed/Barren, Open Water, Winter Wheat,
  Alfalfa, Fallow/Idle Cropland, Cotton, Sorghum, Other. The checkpoint itself
  stores only the ids (`meta["CLASSES"]` is `1..13`, `meta["PALETTE"]` is null),
  so these names come from the model card, not from the file.
- **Tiling**: `test_cfg` is mmseg slide mode — 224 px windows at 112 px stride,
  which is also a hard shape requirement: the backbone flattens `patch16` tokens
  across all three frames (768×3 = 2304 channels into the head), so a 209 px chip
  dies with `shape '[-1, 2304, …]' is invalid`.
- **Trained on US HLS + CDL**, so a scene outside the US is a domain shift: the
  classes that transfer best are the water/developed/vegetation ones, and a crop
  missing from the label set (rice, for instance) can only land in `Other` or
  `Wetlands`. Treat the output as a pipeline check, not an agronomic result.

### Loading it

This one is **not** a terratorch checkpoint, whatever the model card's
`library_name` says: it is MMSegmentation (`{"meta", "state_dict", "optimizer"}`,
`meta["mmseg_version"]` 0.30). No terratorch factory matches its
`TemporalEncoderDecoder`, and `terratorch.NECK_REGISTRY` does not exist in 1.2.x
at all — neither is a bug to chase.

Its config lives inside the file as **Python source** (mmseg style, opening
`dist_params = dict(...)`). That source is **not executable here**, so do not try:
`run_python` rejects `exec()` outright in safe mode, the source imports `mmseg`
and `geospatial_fm` (neither installed), and `terratorch` has no `__version__`
attribute to probe with first. Read the config as **text**, and read the
architecture off the tensors themselves:

```python
ckpt = torch.load(
    models.path("ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification",
                "multi_temporal_crop_classification_Prithvi_100M.pth"),
    map_location="cpu", weights_only=False)
ws.write_text("results/ckpt_config.txt", ckpt["meta"]["config"])  # read_file it
sd = ckpt["state_dict"]                                           # 112 tensors
# backbone.cls_token (1,1,768)  backbone.pos_embed (1,589,768)   # 1 + 3*14*14
# backbone.patch_embed.proj.weight (768,6,1,16,16)              # in_chans 6
# backbone.blocks.0..5 — depth 6, num_heads 8, embed_dim 768
# decode_head.convs.0.conv.weight (256,2304,3,3) / .bn.* / .activate  ← ConvModule
# decode_head.conv_seg.weight (13,256,1,1)
# auxiliary_head.convs.{0,1}.conv.* — present, unused by forward
```

`backbone.patch_embed.proj.weight` is the whole input contract in one line: a
`Conv3d(6, 768, kernel=(1,16,16))` over `(B, 6, 3, 224, 224)`, i.e. channels are
bands, depth is timesteps.

Build a module whose parameter names match those keys and load `sd` into it —
nothing else works without mmseg, and a **strict** `load_state_dict` passing with
no missing/unexpected keys *is* the proof the reconstruction is exact. Two shapes
are easy to get wrong, and both fail that strict load loudly:
- the head is a **`ConvModule`** (`convs.0.conv.weight`, `convs.0.bn.*`,
  `convs.0.activate`), not an `nn.Sequential` (`convs.0.0.weight`); the decode
  head is `num_convs=1`.
- the neck flattens the three frame tokens into the head's 2304 channels, so a
  chip that is not 224 px dies with `shape '[-1, 2304, …]' is invalid`.

Do not expect help from the repo's own files either: its `config.yaml` is
**empty** (0 bytes), and `multi_temporal_crop_classification_Prithvi_100M.py` is a
*training* config that imports `geospatial_fm` and mmseg, neither installed here.

The checkpoint path comes from **`models`**, a read-only global already bound in
`run_python`: `models.path(repo, filename)` (revision-aware), `models.dir(repo)`,
`models.list(repo)`. Do **not** write `import models` — importing that name is
refused, while the global itself is right there. `ai_fetch_model`'s return also
carries `path`. `run_python` has no network, no `importlib` and no `open()`, so
fetch weights with the tool first; `ws.write_text` / `ws.read_text`
(`python_help("ws")`) are how text moves in and out.

## Fetching

```python
ai_fetch_model("ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification")
ai_fetch_model("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
               filenames=["Prithvi-EO-V2-300M-TL-Sen1Floods11.pt", "config.yaml"])
```

Each call returns `{"path": "<abs revision dir>", "downloaded": n, "skipped": n,
"files": [...]}`. `models.path(repo, filename)` resolves the same file from the
cache, and either works — but fetch first: the sandbox has no network. An
already-cached repo comes back with `downloaded: 0`, so the call is cheap to
repeat.

Checkpoint names differ per repo (`Prithvi-EO-V2-300M-TL-Sen1Floods11.pt` here,
`Prithvi_EO_V2_300M_TL.pt` for the plain TL backbone), so list the repo before
fetching: `ai_fetch_model(repo, filenames=[])` reports what is there. Each is
about 1.3 GB. Fine-tuned repos also ship `config.yaml` — the training config with
the band list and size the head expects — and `examples/*_S2Hand.tif`, a small
Sentinel-2 chip in the right layout, which is the cheapest way to prove an
inference path before pointing it at real data.

## Loading

```python
from terratorch.tasks import SemanticSegmentationTask
from lightning.pytorch import Trainer

task = SemanticSegmentationTask.load_from_checkpoint(
    ckpt,                       # "<ai_fetch_model path>/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"
    map_location="cpu",
)
task.eval()
preds = Trainer(accelerator="cpu").predict(task, datamodule=dm)
```

Or build the model explicitly when you want your own head or a different decoder:

```python
from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands

factory = EncoderDecoderFactory()
model = factory.build_model(
    task="segmentation", backbone="prithvi_eo_v2_300",
    backbone_ckpt_path=ckpt,    # "<ai_fetch_model path>" for the backbone repo
    backbone_bands=[HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
                    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2],
    backbone_img_size=512,
    necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]},
           {"name": "ReshapeTokensToImage"},
           {"name": "LearnedInterpolateToPyramidal"}],
    decoder="UperNetDecoder", decoder_channels=256, num_classes=2,
)
```

## What it expects

- **Bands**: HLS-ordered, six of them by default — Blue, Green, Red,
  NIR_NARROW, SWIR_1, SWIR_2 — as `terratorch.datasets.HLSBands`. Sentinel-2
  L2A gives the same bands at 10/20 m; resample to a common grid and order them
  exactly as above. A fine-tuned head may declare a different list: read its card.
- **Size**: 224 or 512 px chips depending on the checkpoint's `img_size`. Tiles
  larger than that need `tiled_inference`, not a resize.
- **Scaling**: HLS reflectance is the training domain; a plain 8-bit RGB GeoTIFF
  is out of distribution even ignoring the missing bands.

## Pitfalls

- The fine-tuned heads ship their training config in the checkpoint; overriding
  `num_classes` or the band list without retraining breaks them silently.
- `band_math`/`spectral_index` in this harness can assemble the six bands from
  Sentinel-2 (`spectral_index` names them `blue`, `green`, `red`, `nir`,
  `swir16`, `swir22`) — build the stack before the model, not after.
- CPU inference is minutes per 512 px chip for the 300M model.
- **Putting the class raster on the map**: `add_raster` writes a `<stem>_cog.tif`
  beside a striped GeoTIFF itself, so do not hand-roll a COG copy and do not
  report one as your own output. Its `colormap=` is a *continuous* ramp over the
  class ids — it makes a 13-class result look like a gradient with no legend, not
  a categorical map. Pass `bands=[1]`, `fit_bounds` the AOI, and say in the
  summary that the class colours are a ramp (or add a legend) rather than the
  checkpoint's own palette.
