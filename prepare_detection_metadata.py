#!/usr/bin/env python3
"""
Convert YOLO-formatted D-Fire annotations into a detection-friendly metadata
JSON file that stores absolute bounding boxes alongside every image.

This keeps the original dataset layout untouched; it simply records where every
image lives and the set of fire/smoke boxes (if any) so detection pipelines
like FCOS can load them without re-reading the YOLO txt files on every epoch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image

LABEL_MAP = {0: "smoke", 1: "fire"}
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Annotation:
    class_id: int
    class_name: str
    bbox_xyxy: Sequence[float]
    bbox_yolo: Sequence[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare detection metadata with absolute bounding boxes."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("D-Fire"),
        help="Path to the D-Fire dataset root (with train/test splits).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dfire_detection_annotations.json"),
        help="Where to write the metadata JSON file.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "test"),
        help="Which dataset splits to include.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional limit on number of images processed per split (debugging).",
    )
    return parser.parse_args()


def read_yolo_file(label_path: Path) -> List[List[float]]:
    if not label_path.exists():
        return []
    entries = []
    for raw in label_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls, x_c, y_c, w, h = parts[:5]
        entries.append([float(cls), float(x_c), float(y_c), float(w), float(h)])
    return entries


def yolo_to_xyxy(
    x_c: float, y_c: float, w: float, h: float, width: int, height: int
) -> List[float]:
    """Convert normalized YOLO box to absolute XYXY pixels."""
    box_w = w * width
    box_h = h * height
    center_x = x_c * width
    center_y = y_c * height
    x1 = max(0.0, center_x - box_w / 2.0)
    y1 = max(0.0, center_y - box_h / 2.0)
    x2 = min(float(width), center_x + box_w / 2.0)
    y2 = min(float(height), center_y + box_h / 2.0)
    return [x1, y1, x2, y2]


def collect_split_metadata(
    dataset_dir: Path, split: str, max_images: int | None = None
) -> Dict:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    metadata = []
    image_iter = sorted(image_dir.iterdir())
    if max_images:
        image_iter = image_iter[:max_images]

    for image_path in image_iter:
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as img:
            width, height = img.size

        annotations = []
        for cls, x_c, y_c, w, h in read_yolo_file(label_path):
            class_id = int(cls)
            annotations.append(
                Annotation(
                    class_id=class_id,
                    class_name=LABEL_MAP.get(class_id, f"class_{class_id}"),
                    bbox_xyxy=yolo_to_xyxy(x_c, y_c, w, h, width, height),
                    bbox_yolo=[x_c, y_c, w, h],
                ).__dict__
            )
        metadata.append(
            {
                "image_id": image_path.stem,
                "split": split,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "width": width,
                "height": height,
                "num_annotations": len(annotations),
                "annotations": annotations,
            }
        )
    return {"split": split, "num_images": len(metadata), "entries": metadata}


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    splits = args.splits
    results = {"dataset_dir": str(dataset_dir.resolve()), "splits": {}}

    for split in splits:
        split_meta = collect_split_metadata(
            dataset_dir, split, max_images=args.max_images_per_split
        )
        results["splits"][split] = split_meta
        print(
            f"[{split}] images: {split_meta['num_images']} "
            f"(with {sum(e['num_annotations'] for e in split_meta['entries'])} boxes)"
        )

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote metadata to {args.output.resolve()}")


if __name__ == "__main__":
    main()
