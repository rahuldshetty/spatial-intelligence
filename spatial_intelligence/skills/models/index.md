# Models

Checkpoints worth knowing, and what each one expects. Kept separate from the
package references (`skill("torchgeo")`, `skill("terratorch")`): a library skill
says how the code works, this tree says which weights to fetch, what bands they
were trained on, and what they predict.

## Getting the weights

```python
ai_fetch_model("torchgeo/resnet18_sentinel2_all_moco")                     # whole repo
ai_fetch_model("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
               filenames=["Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"])        # just the checkpoint
```

The tool verifies the Hub's sha256 and returns absolute paths under `.models`.
Inside the sandbox, `models.path(repo, filename)` returns the same path and
refuses a file that was never fetched. A library's own downloader keeps a
*different* layout under `HF_HOME`/`TORCH_HOME` (`torch.hub` hashes the URL into
the filename), so a `weights=` argument still needs the network even after a
fetch — load the state dict from `models.path(...)` yourself, as
`skill("torchgeo")` shows. The sandbox has no network, so any fetch inside
`run_python` fails by design.

## Choosing one

| Model | Task | Input | Notes |
|---|---|---|---|
| [Prithvi-EO-2.0](prithvi.md) (300M/600M/tiny) | backbone for segmentation, classification, regression | 6-band HLS or the sensor bands of the tuned head | NASA/IBM foundation model; fine-tuned heads for floods, burn scars, crops |
| [SSL4EO backbones](ssl4eo-backbones.md) | features, classification, change detection | Sentinel-2 (13-band or RGB), Landsat, Sentinel-1 | torchgeo's multi-weight ResNet/Swin/ViT/DOFA checkpoints |
| Fields of the World U-Net (`skill("torchgeo")`) | agricultural **field boundaries** | Sentinel-2 `B4,B3,B2,B8` × 2 dates, uint16 | pretrained smp U-Net, 2- or 3-class, ~53 MB |
| SlimSAM (`slimsam-77`, in the catalog) | promptable segmentation | RGB | the built-in `segment_image` model, ONNX, no torch needed |
| YOLOS-tiny (`yolos-tiny`, in the catalog) | object detection | RGB | the built-in `detect_objects` model, COCO classes |
| Satlas ESRGAN (`satlas-esrgan-1s2` … `-16s2`, in the catalog) | 4x super-resolution | RGB, 1-16 dates | the built-in `super_resolve` model, ONNX, 67 MB each |
| Satlas ESRGAN (`satlas-esrgan-10m`/`-20m`/`-60m`, in the catalog) | 4x super-resolution | TCI + extra S2 bands, 8 dates | 2.5 m RGB output from 10 m input; detail is synthetic |

The built-in ONNX models are the default for segmentation, detection, and
super-resolution. Reach for the models in this tree when the task needs
multispectral input, a task-specific fine-tune, or features rather than masks.

## Super-resolution bands

The Satlas checkpoints take date-major channels, each date contributing its
bands in this order:

| Model | Dates | Bands per date | Input channels |
|---|---:|---|---:|
| `satlas-esrgan-1s2` … `-16s2` | 1/2/4/8/16 | TCI R,G,B | 3N |
| `satlas-esrgan-10m` | 8 | TCI R,G,B, B08 | 32 |
| `satlas-esrgan-20m` | 8 | TCI R,G,B, B08, B05, B06, B07, B11, B12 | 72 |
| `satlas-esrgan-60m` | 8 | TCI R,G,B, B01, B08, B09, B10, B05, B06, B07, B11, B12 | 96 |

8-bit bands are divided by 255; 16-bit Sentinel-2 bands by 8160 and clipped.
TCI (L1C) is the matched input; an arbitrary stretched RGB composite is out of
distribution. The extra 20/60 m bands must already be resampled to the TCI
grid. Output is always 4x the input, not a measured resolution gain.

## Rules

- **Verify bands and normalisation before loading.** Band order is the most
  common silent failure: a 6-band Prithvi head fed RGB produces a confident,
  wrong mask. Model cards and `weights.meta` are the source of truth.
- **Record the licence when you use a checkpoint.** The Hub repos carry their
  own terms; TerraTorch and TorchGeo host none themselves.
- **Pin the revision for a reproducible run**: `ai_fetch_model(repo, revision="...")`
  — a floating `main` can change under you between runs.
