# torchgeo

Datasets, samplers, and **multispectral pretrained backbones** for geospatial
deep learning — the place to get a model that has seen Sentinel-2 or Landsat
rather than photographs. Docs: <https://docs.torchgeo.org>.

Version note: Lightning modules live in `torchgeo.trainers` before 0.9 and in
`torchgeo.tasks` from 0.9 on. Check which this machine has with
`python_help("torchgeo.tasks")` — the generated `api/` (see below) reflects the
installed version.

## Load a pretrained backbone

```python
from torchgeo.models import get_model, get_model_weights, list_models

weights = get_model_weights("resnet18").SENTINEL2_ALL_MOCO   # 13-band Sentinel-2 TOA
model = get_model("resnet18", weights=weights)               # downloads — see the pitfall
model.eval()

print(weights.meta["in_chans"], weights.meta["bands"])       # band order the weights expect
x = weights.transforms(chip)                                 # resize/crop/normalise as trained
with torch.no_grad():
    features = model(x.unsqueeze(0))
```

`list_models()` names everything registered; `get_weight("ResNet18_Weights.SENTINEL2_ALL_MOCO")`
takes the enum name as a string. `get_model` forwards extra keywords to
`timm.create_model`, so `num_classes=` re-heads the backbone in one call.

## Draw field boundaries (agriculture)

`Unet_Weights.SENTINEL2_2CLASS_FTW` / `..._3CLASS_FTW` are the pretrained
**Fields of the World** baselines that ship with torchgeo 0.8 — an smp U-Net
(efficientnet-b3) that outlines agricultural fields from **two** Sentinel-2
dates. Its contract, all of it readable from `weights.meta`:

- `in_chans=8`: `B4, B3, B2, B8` for the first image, then the same four for the
  second (`weights.meta["bands"]` is exactly that list).
- `num_classes` 2 = field / non-field; the 3-class variant adds `boundary`.
- `weights.transforms` is `Normalize(mean=[0.], std=[3000.])` — feed **uint16
  reflectance** (~0–12000), never 0–1.
- Trained at 256 px chips; fully convolutional, so other sizes run, but 256 is
  the scale it learned.
- `ai_fetch_model("torchgeo/ftw",
  filenames=["commercial/2-class/sentinel2_unet_effb3-9c04b7c6.pth"])` (~53 MB).
  The `commercial/` checkpoints are CC-BY-4.0, the `_NC_` ones non-commercial.

## Read the raster the model expects

```python
from torchgeo.datasets import RasterDataset

ds = RasterDataset(paths="data/scene.tif", crs="EPSG:4326", res=10)
print(ds.crs, ds.bounds, ds.index.bounds)   # geographic index, not pixel corners
chip = ds[ds.bounds]                        # whole extent as a tensor; use a sampler for patches
```

`RasterDataset`/`VectorDataset`/`GeoDataset` compose with `&` (intersection) and
`|` (union), and `torchgeo.samplers` (`RandomGeoSampler`, `GridGeoSampler`,
`RandomPatchSampler`) yield patches by CRS coordinate — the right way to tile a
scene for inference. For one window of a big file, `rasterio` plus
`windows.Window(col_off, row_off, w, h)` is simpler and stays in the sandbox's
existing stack.

## Inference from a fine-tuned checkpoint

```python
from lightning.pytorch import Trainer
from torchgeo.trainers import SemanticSegmentation        # torchgeo.tasks on 0.9+

task = SemanticSegmentation.load_from_checkpoint(
    ckpt, model="unet", backbone="resnet50", in_channels=6, task="binary",
    loss="bce", ignore_index=-1, weights=True,
)
preds = Trainer(accelerator="cpu").predict(task, datamodule=dm)
```

`predict` runs `predict_step` per batch; `dm` is any datamodule whose
`predict_dataloader` yields the tiles to score. Without a datamodule, call
`task.predict_step(batch, 0)` directly on a tensor batch.

## Pitfalls that cost the most time

- **Band order beats everything.** A 13-band model handed `[R, G, B]` raises or,
  worse, silently scores noise. Read `weights.meta["bands"]` and match the
  GeoTIFF's bands before loading.
- **Use the weights' own transform.** `weights.transforms` is the resize, crop,
  and per-sensor normalisation the checkpoint was trained with; ImageNet stats
  are wrong for multispectral input.
- **`weights=` downloads, and the sandbox has no network.** Passing `weights=` to
  `get_model` calls `weights.get_state_dict()` → `torch.hub.load_state_dict_from_url`,
  which needs the internet even though `TORCH_HOME` points at `.models`. Fetch the
  file first with `ai_fetch_model("torchgeo/<repo>", filenames=[...])`, then build
  and load it yourself:

  ```python
  from torchgeo.models import get_model
  model = get_model("unet", weights=None, classes=2,
                    encoder_name="efficientnet-b3", in_channels=8)
  model.load_state_dict(torch.load(models.path("torchgeo/ftw", "<file>.pth"),
                                   map_location="cpu"))
  ```

  The kwargs must mirror `weights.meta` — `in_chans`, `encoder`, `num_classes` —
  or the state dict will not fit.
- **CPU inference is slow**: minutes per tile for a large backbone. Tile small,
  keep one model loaded across cells, and pass `timeout_seconds` to `run_python`
  when a run legitimately needs longer than the default.

## Where to look next

- `api/` — generated from the installed version: one file per topic, plus
  `datasets.catalog.md` for the dataset zoo. `skill("torchgeo", path="api/models.md")`.
- Model checkpoints and what they expect: `skill("models")` (Prithvi and the
  SSL4EO backbones live there, since they are shared with terratorch).
