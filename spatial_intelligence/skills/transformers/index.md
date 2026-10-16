# transformers

Hugging Face's model classes, processors, and pipelines — what the app uses to
run a checkpoint you picked from the model search. The built-in ONNX runners
(`slimsam-77`, `yolos-tiny`) are faster for what they do; this tree is for any
other Hub repository: SegFormer, Mask2Former, DETR, SAM, or a fine-tune of one.

Docs: <https://huggingface.co/docs/transformers>. The generated `api/` in this
tree reflects the installed version — read it with
`skill("transformers", path="api/models.md")` or search it with
`skill("transformers", query="from_pretrained")`.

## The one rule: fetch first, then load from disk

The sandbox has no network. Weights must be fetched with the `ai_fetch_model`
tool before `run_python` runs, which pins the revision and verifies the hash.
Inside the sandbox, `models.path(repo, filename)` returns that file's path and
refuses a file that was never fetched.

```python
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

repo = "nvidia/segformer-b0-finetuned-ade-512-512"
weights = models.path(repo, "model.safetensors")          # fetched by ai_fetch_model
processor = AutoImageProcessor.from_pretrained(weights.parent)
model = AutoModelForSemanticSegmentation.from_pretrained(weights.parent)
model.eval()

with torch.no_grad():
    inputs = processor(images=rgb_uint8, return_tensors="pt")   # rgb_uint8: HxWx3 numpy
    logits = model(**inputs).logits
labels = logits.argmax(1)[0].numpy()                      # per-pixel class ids
```

For a promptable checkpoint (SAM family), `SamModel` + `SamProcessor` take points
or boxes and return masks: build the prompt with the processor
(`input_points=[[[x, y]]]`), run
`model(**inputs, multimask_output=False)`, then
`processor.image_processor.post_process_masks(...)`. `skill("models")` says which
checkpoints this app already knows, and what bands each expects.

## Before you trust an output

- **Check the bands.** A processor normalises for **RGB**; Sentinel-2 10 m
  imagery is 4-band and Landsat is 6+. Feeding the wrong band order produces a
  confident, wrong mask. Read `processor.image_processor.image_mean/std` and the
  model card, and select or composite the bands to match.
- **Sizes.** Many heads want 512 or 1024 px and a specific channel count; resize
  with the processor (or a tile of `results/` raster), not with a guess.
- **Raster to tiles.** Read a window of the raster with `rasterio` (or the
  `clip` tool), build an HxWx3 uint8 array, and run the model per tile — the
  checkpoint knows nothing about georeferencing. Attach the transform back when
  you write the result out.
- **CPU cost.** A ViT-B-class model over a 1024 px tile takes seconds; a whole
  scene does not fit. Clip to the area of interest first, or use `bounds` on the
  analysis tool that calls this.

## Turning a mask into something the map shows

`skimage.measure.find_contours` or `rasterio.features.shapes` gives polygons;
keep the affine transform so the GeoJSON lands in the raster's CRS, then
`add_geojson` it. That is exactly what the built-in `segment_image` tool does
after its own ONNX decode — do the same with the classes this checkpoint
predicts.
