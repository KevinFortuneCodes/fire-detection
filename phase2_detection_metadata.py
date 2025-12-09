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
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

LABEL_MAP = {1: "fire", 2: "smoke", 3: "nothing", 4: "fire_and_smoke"}
FIRE_LABEL_IDX = 1
SMOKE_LABEL_IDX = 2
NOTHING_LABEL_IDX = 3
FIRE_AND_SMOKE_LABEL_IDX = 4
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# FCOS detection class mapping: metadata uses 1-indexed (1=fire, 2=smoke)
# but FCOS expects 0-indexed classes (0=fire, 1=smoke)
# Use this function to convert metadata class_idx to FCOS class_idx
def metadata_to_fcos_class(metadata_class_idx: int) -> int:
    """
    Convert metadata class_idx to FCOS class_idx.
    
    Metadata: 1=fire, 2=smoke
    FCOS: 0=fire, 1=smoke
    
    Args:
        metadata_class_idx: Class index from metadata (1 or 2)
    
    Returns:
        FCOS class index (0 or 1)
    """
    if metadata_class_idx == FIRE_LABEL_IDX:
        return 0  # fire
    elif metadata_class_idx == SMOKE_LABEL_IDX:
        return 1  # smoke
    else:
        raise ValueError(f"Invalid metadata class_idx: {metadata_class_idx}")


def fcos_to_metadata_class(fcos_class_idx: int) -> int:
    """
    Convert FCOS class_idx back to metadata class_idx.
    
    Args:
        fcos_class_idx: FCOS class index (0 or 1)
    
    Returns:
        Metadata class index (1 or 2)
    """
    if fcos_class_idx == 0:
        return FIRE_LABEL_IDX  # fire
    elif fcos_class_idx == 1:
        return SMOKE_LABEL_IDX  # smoke
    else:
        raise ValueError(f"Invalid FCOS class_idx: {fcos_class_idx}")


@dataclass
class Annotation:
    class_idx: int
    class_name: str
    yolo_class_id: int
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
        help=(
            "Dataset splits to include. If a requested split does not exist but "
            "is named 'validation' (or 'val'), it will be derived from the "
            "training split using --val-percent."
        ),
    )
    parser.add_argument(
        "--fire-class-id",
        type=int,
        default=1,
        help="YOLO class id that corresponds to fire in annotations.",
    )
    parser.add_argument(
        "--smoke-class-id",
        type=int,
        default=0,
        help="YOLO class id that corresponds to smoke in annotations.",
    )
    parser.add_argument(
        "--resample-percent",
        type=float,
        default=None,
        help="Optional percentage (0-100] of images to retain per split while preserving class ratios.",
    )
    parser.add_argument(
        "--resample-seed",
        type=int,
        default=1234,
        help="Random seed used when resampling is enabled.",
    )
    parser.add_argument(
        "--val-percent",
        type=float,
        default=10.0,
        help="Percentage of the (resampled) training split to carve out for validation when requested.",
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


def determine_image_label(annotation_list: List[Dict]) -> int:
    has_fire = any(ann["class_idx"] == FIRE_LABEL_IDX for ann in annotation_list)
    has_smoke = any(ann["class_idx"] == SMOKE_LABEL_IDX for ann in annotation_list)
    if has_fire and has_smoke:
        return FIRE_AND_SMOKE_LABEL_IDX
    if has_fire:
        return FIRE_LABEL_IDX
    if has_smoke:
        return SMOKE_LABEL_IDX
    return NOTHING_LABEL_IDX


def resample_entries(
    entries: List[Dict], percent: float | None, seed: int
) -> List[Dict]:
    if percent is None or percent >= 100:
        return entries
    if percent <= 0:
        raise ValueError("--resample-percent must be > 0")

    rng = random.Random(seed)
    fraction = percent / 100.0
    buckets: Dict[int, List[Dict]] = {}
    for entry in entries:
        buckets.setdefault(entry["image_label_idx"], []).append(entry)

    sampled: List[Dict] = []
    for label_idx, bucket in buckets.items():
        if not bucket:
            continue
        target = max(1, int(round(len(bucket) * fraction)))
        target = min(target, len(bucket))
        if target >= len(bucket):
            sampled.extend(bucket)
        else:
            sampled.extend(rng.sample(bucket, target))
    return sampled


def compute_label_counts(entries: List[Dict]) -> Dict[str, int]:
    return {
        LABEL_MAP[idx]: sum(1 for entry in entries if entry["image_label_idx"] == idx)
        for idx in LABEL_MAP
    }


def derive_validation_split(
    train_meta: Dict,
    val_percent: float,
    seed: int,
    split_name: str = "validation",
) -> Tuple[Dict, Dict]:
    if val_percent <= 0:
        raise ValueError("--val-percent must be greater than 0 to derive validation.")
    entries = train_meta["entries"]
    if not entries:
        raise ValueError("Cannot derive validation split because training split is empty.")

    rng = random.Random(seed + 101)
    fraction = val_percent / 100.0
    buckets: Dict[int, List[Dict]] = {}
    for entry in entries:
        buckets.setdefault(entry["image_label_idx"], []).append(entry)

    val_entries: List[Dict] = []
    remaining_entries: List[Dict] = []
    for label_idx, bucket in buckets.items():
        if not bucket:
            continue
        bucket_copy = bucket[:]
        rng.shuffle(bucket_copy)
        target = int(round(len(bucket_copy) * fraction))
        target = min(max(target, 0), len(bucket_copy))
        val_entries.extend(bucket_copy[:target])
        remaining_entries.extend(bucket_copy[target:])

    # If nothing selected due to tiny dataset, keep training untouched.
    if not val_entries:
        return train_meta, {
            "split": split_name,
            "num_images": 0,
            "label_counts": compute_label_counts(val_entries),
            "entries": [],
            "original_count": 0,
        }

    # Update training metadata
    train_meta["entries"] = remaining_entries
    train_meta["num_images"] = len(remaining_entries)
    train_meta["label_counts"] = compute_label_counts(remaining_entries)

    val_meta = {
        "split": split_name,
        "num_images": len(val_entries),
        "label_counts": compute_label_counts(val_entries),
        "entries": val_entries,
        "original_count": len(val_entries),
    }
    return train_meta, val_meta


def collect_split_metadata(
    dataset_dir: Path,
    split: str,
    fire_class_id: int = 1,
    smoke_class_id: int = 0,
    resample_percent: float | None = None,
    resample_seed: int = 1234,
) -> Dict:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    class_idx_map = {
        fire_class_id: FIRE_LABEL_IDX,
        smoke_class_id: SMOKE_LABEL_IDX,
    }

    metadata = []
    image_iter = sorted(image_dir.iterdir())
    
    # Count total images first for progress tracking
    total_images = sum(1 for img in image_iter if img.suffix.lower() in VALID_IMAGE_SUFFIXES)
    image_iter = sorted(image_dir.iterdir())  # Reset iterator
    
    print(f"  [{split}] Processing {total_images} images...")
    
    processed = 0
    for image_path in image_iter:
        if image_path.suffix.lower() not in VALID_IMAGE_SUFFIXES:
            continue
        
        processed += 1
        if processed % 500 == 0 or processed == total_images:
            print(f"  [{split}] Processed {processed}/{total_images} images ({100*processed/total_images:.1f}%)")
        
        label_path = label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as img:
            width, height = img.size

        annotations: List[Dict] = []
        for cls, x_c, y_c, w, h in read_yolo_file(label_path):
            yolo_class_id = int(cls)
            class_idx = class_idx_map.get(yolo_class_id)
            if class_idx is None:
                continue
            annotations.append(
                Annotation(
                    class_idx=class_idx,
                    class_name=LABEL_MAP[class_idx],
                    yolo_class_id=yolo_class_id,
                    bbox_xyxy=yolo_to_xyxy(x_c, y_c, w, h, width, height),
                    bbox_yolo=[x_c, y_c, w, h],
                ).__dict__
            )

        image_label_idx = determine_image_label(annotations)
        metadata.append(
            {
                "image_id": image_path.stem,
                "split": split,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "width": width,
                "height": height,
                "image_label_idx": image_label_idx,
                "image_label_name": LABEL_MAP[image_label_idx],
                "num_annotations": len(annotations),
                "annotations": annotations,
            }
        )
    original_count = len(metadata)
    
    if resample_percent is not None:
        print(f"  [{split}] Resampling to {resample_percent}% of {original_count} images...")
    metadata = resample_entries(metadata, resample_percent, resample_seed)
    
    print(f"  [{split}] Complete: {len(metadata)} images (from {original_count} original)")

    label_counts = compute_label_counts(metadata)

    return {
        "split": split,
        "num_images": len(metadata),
        "label_counts": label_counts,
        "entries": metadata,
        "original_count": original_count,
    }


def main() -> None:
    args = parse_args()
    print(f"Starting metadata generation...")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Splits to process: {args.splits}")
    if args.resample_percent:
        print(f"Resample percentage: {args.resample_percent}%")
    print("-" * 60)
    
    dataset_dir = args.dataset_dir
    splits = args.splits
    results = {"dataset_dir": str(dataset_dir.resolve()), "splits": {}}

    missing_splits = []

    for split in splits:
        split_dir = dataset_dir / split
        if split_dir.exists():
            print(f"\nProcessing split: {split}")
            split_meta = collect_split_metadata(
                dataset_dir,
                split,
                fire_class_id=args.fire_class_id,
                smoke_class_id=args.smoke_class_id,
                resample_percent=args.resample_percent,
                resample_seed=args.resample_seed,
            )
            results["splits"][split] = split_meta
        else:
            missing_splits.append(split)

    for split in missing_splits:
        split_lower = split.lower()
        if split_lower in {"val", "validation"}:
            if "train" not in results["splits"]:
                raise ValueError(
                    "Cannot derive validation split because training split is missing."
                )
            train_meta, val_meta = derive_validation_split(
                results["splits"]["train"],
                args.val_percent,
                args.resample_seed,
                split_name=split,
            )
            results["splits"]["train"] = train_meta
            results["splits"][split] = val_meta
            print(
                f"[{split}] derived {val_meta['num_images']} samples "
                f"({args.val_percent}% of train)."
            )
        else:
            raise FileNotFoundError(
                f"Requested split '{split}' does not exist under {dataset_dir}."
            )

    for split in splits:
        split_meta = results["splits"][split]
        label_counts = ", ".join(
            f"{label}: {count}" for label, count in split_meta["label_counts"].items()
        )
        box_total = sum(e["num_annotations"] for e in split_meta["entries"])
        print(
            f"[{split}] images: {split_meta['num_images']} "
            f"(original {split_meta['original_count']}), "
            f"boxes: {box_total}; label distribution → {label_counts}"
        )

    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote metadata to {args.output.resolve()}")


if __name__ == "__main__":
    main()
