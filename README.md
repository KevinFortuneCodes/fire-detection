# fire-detection

Fire detection software for Machine Learning class project.

## Preparing detection metadata

The raw D-Fire archive ships with YOLO-style `train/` and `test/` folders.
To feed those annotations into detectors such as FCOS we convert them into a
single JSON manifest that stores absolute bounding boxes alongside every image.

```
python prepare_detection_metadata.py \
  --dataset-dir D-Fire \
  --output dfire_detection_annotations.json
```

Arguments:

- `--dataset-dir`: path containing `train/images`, `train/labels`, etc. Default is `D-Fire` relative to the repo root.
- `--output`: where to write the manifest. Defaults to `dfire_detection_annotations.json`.
- `--max-images-per-split`: optional cap for smoke tests.

Each JSON entry looks like:

```json
{
  "image_id": "AoF07128",
  "image_path": ".../D-Fire/train/images/AoF07128.jpg",
  "width": 640,
  "height": 512,
  "annotations": [
    {
      "class_id": 1,
      "class_name": "fire",
      "bbox_xyxy": [163.0, 397.0, 314.0, 487.0],
      "bbox_yolo": [0.2162, 0.7391, 0.1369, 0.1505]
    }
  ]
}
```

`bbox_yolo` stores the original normalized YOLO coordinates while `bbox_xyxy`
contains absolute pixel boxes ready for training. A dataloader can iterate
through each `entries` item in the JSON, open `image_path`, and convert the
annotation list into a tensor of shape `(num_boxes, 5)` (`x1, y1, x2, y2, class_id`)
to pass into a detector.

## Using FCOS with the metadata

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
