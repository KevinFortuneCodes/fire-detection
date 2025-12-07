#!/usr/bin/env python3
"""
Dataset and dataloader for FCOS fire/smoke detection training.

This module provides a PyTorch Dataset class that loads images and annotations
from the JSON metadata created by phase2_detection_metadata.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Import the mapping function from phase2 (parent directory)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase2_detection_metadata import metadata_to_fcos_class


class FireSmokeDetectionDataset(Dataset):
    """Dataset for fire/smoke detection using phase2 metadata."""
    
    def __init__(
        self,
        metadata_path: Path,
        split: str,
        target_size: int = 800,
        augment: bool = False,
    ):
        """
        Args:
            metadata_path: Path to dfire_detection_annotations.json
            split: Which split to use (train/test/validation)
            target_size: Target image size for model input (default 800 for FCOS)
            augment: Whether to apply data augmentation (for training)
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if split not in metadata["splits"]:
            raise ValueError(f"Split '{split}' not found in metadata")
        
        self.entries = metadata["splits"][split]["entries"]
        self.target_size = target_size
        self.augment = augment
        
        # Base transforms
        self.resize = transforms.Resize((target_size, target_size))
        self.to_tensor = transforms.ToTensor()
        
        # Normalization for ImageNet pretrained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Augmentation transforms (if enabled)
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                # Note: For box-aware augmentation, you'd need to implement custom transforms
                # that also transform bounding boxes. For now, we keep it simple.
            ])
        else:
            self.augment_transform = None
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # Load image
        image_path = Path(entry["image_path"])
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        
        # Calculate scale factors
        scale_w = self.target_size / original_width
        scale_h = self.target_size / original_height
        
        # Apply augmentation if enabled
        if self.augment_transform:
            image = self.augment_transform(image)
        
        # Resize and normalize image
        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Convert annotations to FCOS format: (N, 5) where each row is (x1, y1, x2, y2, class_idx)
        # Scale boxes from original image size to target_size
        boxes = []
        for ann in entry["annotations"]:
            x1, y1, x2, y2 = ann["bbox_xyxy"]
            
            # Scale to target size
            x1_scaled = x1 * scale_w
            y1_scaled = y1 * scale_h
            x2_scaled = x2 * scale_w
            y2_scaled = y2 * scale_h
            
            # Convert metadata class_idx (1=fire, 2=smoke) to FCOS class_idx (0=fire, 1=smoke)
            metadata_class = ann["class_idx"]
            fcos_class = metadata_to_fcos_class(metadata_class)
            
            boxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled, fcos_class])
        
        if len(boxes) == 0:
            # No boxes - return empty tensor
            gt_boxes = torch.zeros((0, 5), dtype=torch.float32)
        else:
            gt_boxes = torch.tensor(boxes, dtype=torch.float32)
        
        return image, gt_boxes


def collate_fn(batch):
    """
    Custom collate function to handle variable number of boxes per image.
    
    Pads boxes to the same length in the batch, using -1 to indicate padding.
    FCOS will ignore boxes with class_idx == -1.
    """
    images = []
    gt_boxes_list = []
    
    for image, gt_boxes in batch:
        images.append(image)
        gt_boxes_list.append(gt_boxes)
    
    images = torch.stack(images)
    
    # Find max number of boxes in batch
    max_boxes = max(len(boxes) for boxes in gt_boxes_list) if gt_boxes_list else 0
    
    # Pad all boxes to same length
    if max_boxes > 0:
        padded_boxes = []
        for boxes in gt_boxes_list:
            if len(boxes) < max_boxes:
                # Pad with -1 (background/padding indicator)
                padding = torch.full(
                    (max_boxes - len(boxes), 5),
                    -1.0,
                    dtype=boxes.dtype
                )
                boxes = torch.cat([boxes, padding], dim=0)
            padded_boxes.append(boxes)
        gt_boxes = torch.stack(padded_boxes)
    else:
        # All images have no boxes
        gt_boxes = torch.zeros((len(images), 0, 5), dtype=torch.float32)
    
    return images, gt_boxes


def create_dataloaders(
    metadata_path: Path,
    batch_size: int = 4,
    target_size: int = 800,
    num_workers: int = 4,
    train_augment: bool = True,
):
    """
    Create train and validation dataloaders.
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = FireSmokeDetectionDataset(
        metadata_path,
        split="train",
        target_size=target_size,
        augment=train_augment,
    )
    
    val_dataset = FireSmokeDetectionDataset(
        metadata_path,
        split="validation",
        target_size=target_size,
        augment=False,  # No augmentation for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader

