#!/usr/bin/env python3
"""
Utility to preprocess the D-Fire dataset for CNN training.

The script reads images from the YOLO-style D-Fire layout:
<root>/<split>/{images,labels}/...

It converts every image into a square RGB image, buckets it into a
fire/no_fire directory (based on whether any bounding box belongs to
the chosen positive class), and emits metadata csv files that can be
consumed by torchvision's ImageFolder, tf.data pipelines, etc.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageOps

# Default semantic mapping deduced from the D-Fire labels
LABEL_MAP = {0: "no_fire", 1: "fire"}
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
        default=("train", "test"),
        help="Dataset splits to process (must exist under dataset-dir).",
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
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional cap on number of images processed per split (useful for smoke tests).",
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
    label_idx = 1 if any(cls == positive_class for cls, *_ in boxes) else 0
    label_name = LABEL_MAP.get(label_idx, f"class_{label_idx}")
    return label_idx, label_name


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


def process_split(
    split: str,
    dataset_dir: Path,
    output_dir: Path,
    image_size: int,
    crop_strategy: str,
    positive_class: int,
    max_images: int | None,
    skip_existing: bool,
    log_every: int,
) -> List[SampleMeta]:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {label_dir}")

    samples: List[SampleMeta] = []
    images = collect_images(image_dir)
    if max_images:
        images = images[:max_images]

    for idx, image_path in enumerate(images, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = read_label_file(label_path)
        label_idx, label_name = determine_label(boxes, positive_class)

        dest_dir = output_dir / split / label_name
        ensure_dir(dest_dir)
        dest_filename = f"{image_path.stem}_{split}.png"
        dest_path = dest_dir / dest_filename

        if skip_existing and dest_path.exists():
            logging.debug("Skipping existing %s", dest_path)
        else:
            processed = process_image(image_path, image_size, crop_strategy)
            processed.save(dest_path, format="PNG")

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

        if idx % log_every == 0:
            logging.info(
                "[%s] processed %d/%d images", split, idx, len(images)
            )

    logging.info("[%s] completed %d images.", split, len(images))
    return samples


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    all_samples: List[SampleMeta] = []
    stats: Dict[str, Dict[str, int]] = {}
    for split in args.splits:
        split_samples = process_split(
            split=split,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
            crop_strategy=args.crop_strategy,
            positive_class=args.positive_class,
            max_images=args.max_images_per_split,
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
