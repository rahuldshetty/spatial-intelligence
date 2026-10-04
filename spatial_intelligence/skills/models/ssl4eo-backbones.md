# SSL4EO backbones (torchgeo)

Self-supervised checkpoints trained on unlabelled Sentinel-2, Landsat, and
Sentinel-1 by the SSL4EO projects, published by torchgeo through torchvision's
multi-weight API. Use them for feature extraction, classification heads, and
change detection — they are backbones, not segmentation models.

## Loading

```python
import torch
from torchgeo.models import get_model, get_model_weights

weights = get_model_weights("resnet18").SENTINEL2_ALL_MOCO
model = get_model("resnet18", weights=weights)
model.eval()

weights.meta["in_chans"]    # 13
weights.meta["bands"]       # ['B1', 'B2', ..., 'B12'] — the order the weights expect
weights.transforms          # resize + centre-crop + per-sensor normalisation, as trained

with torch.no_grad():
    features = model(weights.transforms(chip).unsqueeze(0))
```

`get_model` forwards its extra keywords to `timm.create_model`, so
`get_model("resnet18", weights=weights, num_classes=10)` gives you a classifier
head on the pretrained trunk in one call. `list_models()` names every registered
builder; `get_weight("ResNet18_Weights.SENTINEL2_ALL_MOCO")` resolves the enum by
string, which is convenient when the name comes from a prompt.

## What is available

| Builder | Weights | Sensors |
|---|---|---|
| `resnet18`, `resnet50`, `resnet152` | `ResNet18_Weights`, … | Sentinel-2 (all 13 bands, or RGB), Landsat TM/ETM/OLI (TOA and SR), Sentinel-1 GRD (VV, VH) |
| `swin_t`, `swin_s`, `swin_b`, `swin_v2_t`, `swin_v2_b` | `Swin_*_Weights` | Satlas-pretrained Sentinel-2 and other bands |
| `vit_small_patch16_224`, `vit_base_patch16_224`, `vit_large_patch16_224`, `vit_huge_patch14_224` | `ViT*_Weights` | SSL4EO-S12 / Satlas |
| `vit_small_patch14_dinov2`, `vit_base_patch14_dinov2` | `ViT*_DINOv2_Weights` | DINOv2 weights, geospatial fine-tunes |
| `dofa_base_patch16_224`, `dofa_large_patch16_224` | `DOFA*_Weights` | any sensor: DOFA takes the wavelengths as input, so one checkpoint spans many bands |
| `scalemae_large_patch16`, `satclip`, `panopticon_vitb14`, `tilenet`, `tessera`, `presto`, `croma_*`, `copernicusfm_base`, `olmoearth_v1` | matching `*_Weights` | Earth-observation foundation models for embeddings and classification |

Check the live list rather than trusting this table:
`list_models()` and `get_model_weights("<builder>")` on the installed version.

## Fetching the checkpoints

torchgeo's weight URLs point at Hub repos, so they cache normally, and can be
pre-fetched with `ai_fetch_model` (this is verified for the ResNet-18 families):

```python
ai_fetch_model("torchgeo/resnet18_sentinel2_all_moco")   # 45 MB, .pth inside
```

## Pitfalls

- **`SENTINEL2_ALL_MOCO` wants 13 bands, `SENTINEL2_RGB_MOCO` wants `B4,B3,B2`.**
  Passing RGB to the 13-band weights raises; passing the 13-band TOA stack to the
  RGB weights fails silently and looks like a bad model.
- **Use `weights.transforms`.** These checkpoints normalise by per-sensor band
  statistics or reflectance scaling, not ImageNet statistics.
- **A backbone is not a segmentation model.** Its output is a feature vector
  (classification) or a feature map (with `features_only`); attach a decoder or
  use terratorch's factories for per-pixel output.
- Sentinel-1 weights expect `VV, VH` in that order, with the dB statistics the
  SSL4EO-S12 transforms apply (`mean=[-12.59, -20.26]`, `std=[5.26, 5.91]`).
