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
| [OpenSR SEN2-SR](https://srgan.opensr.eu) (`sen2-sr`, in the catalog) | Sentinel-2 4x super-resolution | B04/B03/B02/B08 raw DN, 10 m | the built-in `super_resolve` model, MIT, 68 MB, texture-oriented |

The built-in models are the default for segmentation, detection, and
super-resolution. Reach for the models in this tree when the task needs
multispectral input, a task-specific fine-tune, or features rather than masks.

## SEN2-SR input

`sen2-sr` is Sentinel-2 specific. It takes exactly four **10 m raw-DN** bands in
this order:

| Band | Name | Role |
|---|---|---|
| 1 | B04 | red |
| 2 | B03 | green |
| 3 | B02 | blue |
| 4 | B08 | NIR |

Pass uint16 reflectance DN as stored (do not divide by 10000). A three-band
TCI/RGB image, an 8-bit composite, a float stack, or a 20 m stack is refused:
it is not the model's input. A standard 12-band L2A stack usually needs
`bands=[4, 3, 2, 8]`. The output is four uint16 bands at 2.5 m; use bands 1-3 for
RGB and band 4 for NIR. The OpenSR package applies measurement-consistency
back-projection, but the detail is generated, not measured.

## Rules

- **Verify bands and normalisation before loading.** Band order is the most
  common silent failure: a 6-band Prithvi head fed RGB produces a confident,
  wrong mask. Model cards and `weights.meta` are the source of truth.
- **Record the licence when you use a checkpoint.** The Hub repos carry their
  own terms; TerraTorch and TorchGeo host none themselves.
- **Pin the revision for a reproducible run**: `ai_fetch_model(repo, revision="...")`
  — a floating `main` can change under you between runs.
