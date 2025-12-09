"""
FCOS (Fully-Convolutional One-Stage) Object Detector Package.

This package contains the FCOS model implementation and training utilities
for fire/smoke detection.
"""

from .fcos import FCOS
from .fcos_dataset import FireSmokeDetectionDataset, create_dataloaders, collate_fn

__all__ = ["FCOS", "FireSmokeDetectionDataset", "create_dataloaders", "collate_fn"]

