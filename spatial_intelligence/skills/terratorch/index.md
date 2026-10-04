# terratorch

Task-level inference and fine-tuning for Earth-observation foundation models
(Prithvi, TerraMind, SatMAE, DOFA, Clay) — the layer that turns a backbone into a
segmentation, classification, or regression model. It builds on PyTorch
Lightning and on torchgeo. Docs: <https://torchgeo.github.io/terratorch/>.

TerraTorch hosts no models; it loads them from the Hub (see `skill("models")`).
Its public surface is `terratorch.registry` (component registries),
`terratorch.models` (factories), `terratorch.tasks` (LightningModules), and
`terratorch.datamodules`.

## Backbone only

```python
from terratorch import BACKBONE_REGISTRY

print([n for n in BACKBONE_REGISTRY if "prithvi" in n])
# ['terratorch_prithvi_eo_tiny', 'terratorch_prithvi_eo_v1_100', 'terratorch_prithvi_eo_v2_300', ...]

model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300", pretrained=True)
model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300", num_frames=1, ckpt_path="/path/to/model.pt")
```

Names may carry a registry prefix (`timm_resnet50`); without one, every registry
is searched for the first match. Use this level when you want features and will
write the head yourself.

## A full task model

```python
import terratorch  # registers the terratorch models with timm
from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands

factory = EncoderDecoderFactory()
model = factory.build_model(
    task="segmentation",
    backbone="prithvi_eo_v2_300",
    backbone_pretrained=True,
    backbone_bands=[HLSBands.BLUE, HLSBands.GREEN, HLSBands.RED,
                    HLSBands.NIR_NARROW, HLSBands.SWIR_1, HLSBands.SWIR_2],
    necks=[{"name": "SelectIndices", "indices": [-1]},
           {"name": "ReshapeTokensToImage"}],
    decoder="FCNDecoder",
    decoder_channels=128,
    head_dropout=0.1,
    num_classes=4,
)
```

Prefixes route the arguments: `backbone_*` to the backbone, `decoder_*` to the
decoder, `head_*` to the head. `SemanticSegmentationTask` wraps exactly this when
you want Lightning features on top:

```python
from terratorch.tasks import SemanticSegmentationTask
from lightning.pytorch import Trainer

task = SemanticSegmentationTask(model_args, "EncoderDecoderFactory", loss="dice", ignore_index=-1)
preds = Trainer(accelerator="cpu").predict(task, datamodule=dm)
```

## Inference from a checkpoint

```python
task = SemanticSegmentationTask.load_from_checkpoint("/path/to/best.ckpt", map_location="cpu")
task.eval()
preds = Trainer(accelerator="cpu").predict(task, datamodule=dm)   # predict_dataloader tiles
```

or the CLI, which needs no Python:

```sh
terratorch predict -c config.yaml --ckpt_path /path/to/best.ckpt \
  --predict_output_dir results/ --data.init_args.predict_data_root data/ \
  --data.init_args.predict_dataset_bands '[BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2]'
```

See `terratorch.tasks.InferenceTask` for a task that does inference only, and
`terratorch.tasks.tiled_inference` for large scenes:

```python
from terratorch.tasks.tiled_inference import tiled_inference

out = tiled_inference(model_forward, batch, h_crop=224, h_stride=200,
                      w_crop=224, w_stride=200, delta=4,
                      blend_overlaps=True, batch_size=16)
```

`tiled_inference` also accepts a **source** and a **sink**, so tiles can be read
from a raster window and written straight to a GeoTIFF instead of held in RAM —
`python_help("terratorch.tasks.tiled_inference")` shows both protocols with a
rasterio example. That is the path to use for anything bigger than one chip.

## Pitfalls

- **The checkpoint's training config is the contract**: band list, band order,
  `img_size`, and normalisation must match, or the output is garbage that still
  looks like a plausible mask. Model cards state them; `skill("models")` records
  what the common ones need.
- `backbone_pretrained=True` downloads the backbone; a fine-tuned checkpoint
  already contains it, so pass `ckpt_path` and skip the download.
- Freeze what you are not training (`freeze_backbone`, `freeze_decoder`) — for
  pure inference, `task.eval()` and `torch.no_grad()` are enough.
- CPU inference on a 300M-parameter backbone is minutes per tile: keep tiles at
  the model's `img_size`, reuse one loaded task across cells, and raise
  `timeout_seconds` on `run_python` when a run needs it.

## Where to look next

- `api/` — generated from the installed version: `skill("terratorch", path="api/tasks.md")`,
  `api/models.md`, `api/datamodules.catalog.md`.
- Checkpoints and their bands: `skill("models", path="prithvi.md")`.
