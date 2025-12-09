#!/usr/bin/env python3
"""
Utility to preprocess the D-Fire dataset for CNN training.

The script reads images from the YOLO-style D-Fire layout:
<root>/<split>/{images,labels}/...

It converts every image into a square RGB image, buckets it into
fire/smoke/no_label directories (based on whether any bounding box belongs to
the chosen positive class), and emits metadata csv files that can be
consumed by torchvision's ImageFolder, tf.data pipelines, etc.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageOps

# Default semantic mapping deduced from the D-Fire labels
LABEL_MAP = {1: "fire", 2: "smoke", 3: "nothing", 4: "fire_and_smoke"}
FIRE_LABEL_IDX = 1
SMOKE_LABEL_IDX = 2
NO_LABEL_IDX = 3
FIRE_AND_SMOKE_LABEL_IDX = 4
NO_LABEL_NAME = LABEL_MAP[NO_LABEL_IDX]
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class SampleMeta:
    """Metadata for a single processed sample."""

    image_id: str
    split: str
    original_path: Path
    processed_path: Path
    label_idx: int
    label_name: str
    num_boxes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process D-Fire images into a classification-friendly layout."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("D-Fire"),
        help="Path to the downloaded D-Fire dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_dfire"),
        help="Directory where processed images/metadata will be stored.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "test","validation"),
        help=(
            "Dataset splits to process. If a requested split (e.g., validation) "
            "does not exist on disk it will be derived from the training split "
            "using --val-percent."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Target square size (pixels) for the processed images.",
    )
    parser.add_argument(
        "--crop-strategy",
        choices=("center", "stretch"),
        default="center",
        help="How to resize images: center crop to square (default) or stretch.",
    )
    parser.add_argument(
        "--positive-class",
        type=int,
        default=1,
        help="YOLO class id that is considered a 'fire' sample.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Only process images that are missing from the output directory.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print a progress message after this many processed images.",
    )
    parser.add_argument(
        "--resample-percent",
        type=float,
        default=20.0,
        help="Optional percentage (0-100] of images to keep per split while preserving class ratios.",
    )
    parser.add_argument(
        "--resample-seed",
        type=int,
        default=1234,
        help="Random seed used when --resample-percent is specified.",
    )
    parser.add_argument(
        "--val-percent",
        type=float,
        default=20.0,
        help="Percentage of the training split to reserve for a derived validation split.",
    )
    return parser.parse_args()


def collect_images(image_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.suffix.lower() in VALID_IMAGE_SUFFIXES and p.is_file()
    )


def read_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().splitlines()
    boxes: List[Tuple[int, float, float, float, float]] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            logging.warning("Skipping malformed label in %s: %s", label_path, line)
            continue
        cls, x_center, y_center, width, height = parts
        try:
            boxes.append(
                (
                    int(float(cls)),
                    float(x_center),
                    float(y_center),
                    float(width),
                    float(height),
                )
            )
        except ValueError:
            logging.warning("Non-numeric label entry in %s: %s", label_path, line)
    return boxes


def determine_label(
    boxes: Sequence[Tuple[int, float, float, float, float]], positive_class: int
) -> Tuple[int, str]:
    if not boxes:
        return NO_LABEL_IDX, NO_LABEL_NAME
    # Determine which classes are present (fire = positive_class, smoke = 0)
    has_fire = any(cls == positive_class for cls, *_ in boxes)
    has_smoke = any(cls == 0 for cls, *_ in boxes)
    
    if has_fire and has_smoke:
        return FIRE_AND_SMOKE_LABEL_IDX, LABEL_MAP[FIRE_AND_SMOKE_LABEL_IDX]
    elif has_fire:
        return FIRE_LABEL_IDX, LABEL_MAP[FIRE_LABEL_IDX]
    else:
        return SMOKE_LABEL_IDX, LABEL_MAP[SMOKE_LABEL_IDX]


def resample_records(
    records: List[Dict[str, object]], percent: float | None, seed: int
) -> List[Dict[str, object]]:
    if percent is None or percent >= 100:
        return records
    if percent <= 0:
        raise ValueError("--resample-percent must be > 0")

    rng = random.Random(seed)
    fraction = percent / 100.0
    buckets: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        buckets.setdefault(record["label_idx"], []).append(record)

    sampled: List[Dict[str, object]] = []
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


def collect_split_records(
    dataset_dir: Path, split: str, positive_class: int
) -> List[Dict[str, object]]:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {label_dir}")

    records: List[Dict[str, object]] = []
    for image_path in collect_images(image_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = read_label_file(label_path)
        label_idx, label_name = determine_label(boxes, positive_class)
        records.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "boxes": boxes,
                "label_idx": label_idx,
                "label_name": label_name,
            }
        )
    return records


def derive_validation_records(
    records: List[Dict[str, object]], val_percent: float, seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if val_percent <= 0:
        raise ValueError("--val-percent must be > 0 to derive validation split.")
    if not records:
        return records, []

    rng = random.Random(seed + 101)
    fraction = val_percent / 100.0
    buckets: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        buckets.setdefault(record["label_idx"], []).append(record)

    val_records: List[Dict[str, object]] = []
    remaining_records: List[Dict[str, object]] = []
    for label_idx, bucket in buckets.items():
        bucket_copy = bucket[:]
        rng.shuffle(bucket_copy)
        target = int(round(len(bucket_copy) * fraction))
        target = min(max(target, 0), len(bucket_copy))
        val_records.extend(bucket_copy[:target])
        remaining_records.extend(bucket_copy[target:])

    if not val_records:
        return records, []

    return remaining_records, val_records


def process_image(image_path: Path, size: int, crop_strategy: str) -> Image.Image:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        if crop_strategy == "center":
            return ImageOps.fit(rgb, (size, size), method=Image.Resampling.BILINEAR)
        return rgb.resize((size, size), Image.Resampling.BILINEAR)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_metadata_csv(samples: Iterable[SampleMeta], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "image_id",
        "split",
        "original_path",
        "processed_path",
        "label_idx",
        "label_name",
        "num_boxes",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "image_id": sample.image_id,
                    "split": sample.split,
                    "original_path": str(sample.original_path),
                    "processed_path": str(sample.processed_path),
                    "label_idx": sample.label_idx,
                    "label_name": sample.label_name,
                    "num_boxes": sample.num_boxes,
                }
            )


def process_records(
    split: str,
    records: List[Dict[str, object]],
    output_dir: Path,
    image_size: int,
    crop_strategy: str,
    skip_existing: bool,
    log_every: int,
) -> List[SampleMeta]:
    samples: List[SampleMeta] = []
    total = len(records)
    for idx, record in enumerate(records, start=1):
        image_path = record["image_path"]
        label_path = record["label_path"]
        boxes = record["boxes"]
        label_idx = record["label_idx"]
        label_name = record["label_name"]

        dest_dir = output_dir / split / label_name
        ensure_dir(dest_dir)
        dest_filename = f"{image_path.stem}_{split}.png"
        dest_path = dest_dir / dest_filename

        if skip_existing and dest_path.exists():
            logging.debug("Skipping existing %s", dest_path)
        else:
            try:
                processed = process_image(image_path, image_size, crop_strategy)
                processed.save(dest_path, format="PNG")
            except Exception as exc:
                logging.warning("Failed to process %s: %s", image_path, exc)
                continue

        samples.append(
            SampleMeta(
                image_id=image_path.stem,
                split=split,
                original_path=image_path.resolve(),
                processed_path=dest_path.resolve(),
                label_idx=label_idx,
                label_name=label_name,
                num_boxes=len(boxes),
            )
        )

        if log_every and idx % log_every == 0:
            logging.info(
                "[%s] processed %d/%d images", split, idx, total
            )

    logging.info("[%s] completed %d images.", split, total)
    return samples


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    split_data: Dict[str, Dict[str, object]] = {}
    missing_splits: List[str] = []
    for split in args.splits:
        split_dir = args.dataset_dir / split
        if split_dir.exists():
            records = collect_split_records(args.dataset_dir, split, args.positive_class)
            original_count = len(records)
            resampled = resample_records(records, args.resample_percent, args.resample_seed)
            if original_count and len(resampled) != original_count:
                logging.info(
                    "[%s] resampled %d → %d images (%.1f%%)",
                    split,
                    original_count,
                    len(resampled),
                    (len(resampled) / original_count) * 100,
                )
            split_data[split] = {
                "records": resampled,
                "original_count": original_count,
            }
        else:
            missing_splits.append(split)

    for split in missing_splits:
        split_lower = split.lower()
        if split_lower in {"val", "validation"}:
            if "train" not in split_data:
                raise ValueError(
                    f"Cannot derive {split} split because training split is unavailable."
                )
            train_records = split_data["train"]["records"]
            remaining, val_records = derive_validation_records(
                train_records, args.val_percent, args.resample_seed
            )
            if not val_records:
                logging.warning(
                    "Unable to derive %s split because training split is too small.",
                    split,
                )
                split_data[split] = {
                    "records": [],
                    "original_count": 0,
                }
            else:
                split_data["train"]["records"] = remaining
                split_data[split] = {
                    "records": val_records,
                    "original_count": len(val_records),
                }
                logging.info(
                    "[%s] derived %d samples (%.1f%% of train).",
                    split,
                    len(val_records),
                    args.val_percent,
                )
        else:
            raise FileNotFoundError(
                f"Requested split '{split}' does not exist under {args.dataset_dir}."
            )

    all_samples: List[SampleMeta] = []
    stats: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        data = split_data.get(split)
        if data is None:
            continue
        split_samples = process_records(
            split=split,
            records=data["records"],
            output_dir=args.output_dir,
            image_size=args.image_size,
            crop_strategy=args.crop_strategy,
            skip_existing=args.skip_existing,
            log_every=args.log_every,
        )
        all_samples.extend(split_samples)
        per_label: Dict[str, int] = {}
        for sample in split_samples:
            per_label[sample.label_name] = per_label.get(sample.label_name, 0) + 1
        stats[split] = per_label

        csv_path = args.output_dir / f"{split}_metadata.csv"
        write_metadata_csv(split_samples, csv_path)
        logging.info("Wrote metadata to %s", csv_path)

    logging.info("Dataset summary:")
    for split, counter in stats.items():
        total = sum(counter.values())
        details = ", ".join(f"{label}: {count}" for label, count in counter.items())
        logging.info("  %s → %d samples (%s)", split, total, details)


if __name__ == "__main__":
    main()
