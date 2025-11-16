# fire-detection

Fire detection software for Machine Learning class project.

## Phase 1 — Classification dataset

**Goal:** Build a balanced three-class dataset (fire / smoke / nothing) in an
ImageFolder-compatible structure for training a coarse classifier. The script
also emits per-split metadata CSVs so you can later audit the preprocessing.

```
python phase1_classification_preprocess.py \
  --dataset-dir D-Fire \
  --output-dir processed_dfire \
  --splits train validation test \
  --val-percent 10 \
  --resample-percent 30 \
  --log-every 25
```

Key options:

- `--dataset-dir`: path containing the D-Fire layout (`train/images`, `train/labels`, ...).
- `--output-dir`: directory where processed PNGs + metadata CSVs will be written.
- `--splits`: list of split names to produce. If you request `validation` (or `val`)
  and it does not exist on disk, the script automatically carves out `--val-percent`
  of the *resampled* training data and writes those images under
  `processed_dfire/validation/...`.
- `--val-percent`: fraction of the training split to reserve for the derived
  validation split (default 10%).
- `--image-size` / `--crop-strategy`: how each frame is converted to a square RGB crop.
- `--positive-class`: YOLO label id treated as “fire” (default 1). All other
  labeled boxes become “smoke”; frames without labels become “nothing”.
- `--resample-percent` / `--resample-seed`: optional down-sampling to keep the
  fire:smoke:nothing ratio intact while reducing dataset size.
- `--skip-existing`: useful for incremental reruns when the output directory already contains PNGs.

Outputs:

- `processed_dfire/<split>/{fire,smoke,nothing}/<image>.png`
- `<output-dir>/<split>_metadata.csv` describing every processed sample
  (`image_id`, original path, assigned label, number of bounding boxes, etc.).

## Phase 2 — Detection metadata

**Goal:** Convert YOLO label files into a single JSON manifest that stores both
the normalized (`bbox_yolo`) and absolute (`bbox_xyxy`) bounding boxes, ready
for detector training (e.g., FCOS). This script shares the sampling/validation
options with Phase 1, but keeps the original image files untouched.

```
python phase2_detection_metadata.py \
  --dataset-dir D-Fire \
  --output dfire_detection_annotations.json \
  --splits train test validation \
  --val-percent 10 \
  --resample-percent 25
```

Important arguments:

- `--dataset-dir`: root of the YOLO dataset.
- `--output`: path of the JSON manifest.
- `--splits`: ordered list of splits to emit. Validation/val entries are derived
  from the training split when they don’t exist on disk (same logic as Phase 1).
- `--val-percent`: percentage of the *resampled* training split moved to validation when derived.
- `--fire-class-id` / `--smoke-class-id`: map YOLO class ids to our fire vs. smoke
  categories (default 1/0). Frames with no annotations end up in the “nothing” bucket.
- `--resample-percent` / `--resample-seed`: sample images per split while retaining the 3-class ratios.

Each JSON entry contains both the per-image label summary and every bounding box:

```json
{
  "image_id": "AoF07128",
  "image_path": ".../D-Fire/train/images/AoF07128.jpg",
  "width": 640,
  "height": 512,
  "image_label_idx": 1,
  "image_label_name": "fire",
  "annotations": [
    {
      "class_idx": 1,
      "class_name": "fire",
      "yolo_class_id": 1,
      "bbox_xyxy": [163.0, 397.0, 314.0, 487.0],
      "bbox_yolo": [0.2162, 0.7391, 0.1369, 0.1505]
    }
  ]
}
```

`bbox_yolo` stores the original normalized YOLO coordinates while `bbox_xyxy`
contains absolute pixel boxes ready for training. A dataloader can iterate
through each `entries` item in the JSON, open `image_path`, and convert the
annotation list into a tensor of shape `(num_boxes, 5)` (`x1, y1, x2, y2, class_idx`)
to pass into a detector. `image_label_idx` is a handy three-class summary
(1 = fire, 2 = smoke, 3 = nothing) that mirrors the classification pipeline and
is also used when resampling.

## Phase 2 model training (FCOS example)

An FCOS-style detector consumes full images plus ground-truth boxes. A typical
PyTorch training loop looks like:

1. Parse the JSON manifest once and keep the per-split `entries`.
2. For every entry, load the RGB image, apply any torchvision transforms
   (resize, normalization, augmentation) and scale `bbox_xyxy` accordingly.
3. Stack the boxes into a tensor and feed `(image_tensor, gt_boxes_tensor)`
   into the FCOS model (`FCOS.forward(images, gt_boxes=...)`).
4. The helper functions inside `fcos.py` handle matching FPN locations to boxes,
   computing deltas, and generating centerness targets automatically.

During inference you only pass `images` to `FCOS.forward` (with
`test_score_thresh`/`test_nms_thresh` if desired) and the model returns
predicted boxes, classes, and scores, letting you declare “fire”, “smoke”, or
“nothing” and draw the bounding boxes on the frame.
