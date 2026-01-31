import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from typing import Generator, Tuple, Optional

def _image_generator(csv_path: Path) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Generator that yields (image, label) tuples on-demand.
    This is memory-efficient as it only loads one image at a time.
    
    Args:
        csv_path: Path to metadata CSV file
        
    Yields:
        Tuple of (image_array, label) where image_array is (256, 256, 3) float32 in [0,1]
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV file not found: {csv_path}")
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Load image from processed_path
            img_path = Path(row['processed_path'])
            if not img_path.exists():
                # Try relative path if absolute doesn't work
                img_path = csv_path.parent / img_path
                if not img_path.exists():
                    continue
            
            # Load image (already processed: RGB, square, saved as PNG)
            with Image.open(img_path) as img:
                # Images are already RGB and square from process_dfire.py
                img_array = np.array(img, dtype=np.float32)
                # Normalize to [0, 1] range
                img_array = img_array / 255.0
            
            # Get label index and convert from 1-based (1,2,3,4) to 0-based (0,1,2,3) for TensorFlow
            label_idx = int(row['label_idx'])
            # Convert: fire=1->0, smoke=2->1, nothing=3->2, fire_and_smoke=4->3
            label_idx_0based = label_idx - 1
            
            yield img_array, label_idx_0based


def create_dataset_from_csv(
    csv_path: Path,
    batch_size: Optional[int] = 32,
    shuffle: bool = True,
    shuffle_buffer_size: Optional[int] = None,
    prefetch: int = tf.data.AUTOTUNE,
    cache: bool = False
) -> tf.data.Dataset:
    """
    Create a memory-efficient tf.data.Dataset from a metadata CSV file.
    Images are loaded on-demand, not all at once.
    
    Args:
        csv_path: Path to metadata CSV file
        batch_size: Batch size for the dataset (None = no batching, batch later)
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling (default: max(1000, 10*batch_size))
                             Use a large buffer (1000+) if CSV is sorted by label
        prefetch: Number of batches to prefetch (default: AUTOTUNE)
        cache: Whether to cache the dataset in memory (only use for small datasets)
        
    Returns:
        tf.data.Dataset yielding (image, label) tuples or batches (if batch_size specified)
    """
    csv_path = Path(csv_path)
    
    # Create dataset from generator
    # Use a factory function to ensure a fresh generator is created for each iteration
    def make_generator():
        """Factory function that creates a fresh generator each time it's called."""
        return _image_generator(csv_path)
    
    dataset = tf.data.Dataset.from_generator(
        make_generator,
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    # Cache if requested (before other transformations for efficiency)
    if cache:
        dataset = dataset.cache()
    
    # Shuffle if requested (after caching for efficiency)
    if shuffle:
        if shuffle_buffer_size is None:
            # Use a larger buffer for better shuffling, especially when CSV is sorted by label
            # For datasets with sorted labels, we need a buffer large enough to mix classes
            if batch_size is not None:
                shuffle_buffer_size = max(1000, 10 * batch_size)  # At least 1000 samples
            else:
                shuffle_buffer_size = 1000  # Default when batch_size is None
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    
    # Batch the dataset (if batch_size is specified)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    
    # Prefetch for better performance (overlaps data preprocessing and model execution)
    dataset = dataset.prefetch(prefetch)
    
    return dataset


def retrieve_data_generator(
    train_path,
    val_path=None,
    test_path=None,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    shuffle_train: bool = True,
    cache_train: bool = False,
    cache_val: bool = False
):
    """
    Memory-efficient data loading using tf.data.Dataset generators (RECOMMENDED).
    All datasets (train, validation, test) use generators with caching for validation/test.
    Images are loaded on-demand, not all at once in memory.
    
    This is the default and recommended method for data loading.
    It is much more memory-efficient than loading all data into memory.
    
    Args:
        train_path: Path to train metadata CSV or directory containing train_metadata.csv
        val_path: Path to validation metadata CSV or directory containing validation_metadata.csv
        test_path: Path to test metadata CSV or directory containing test_metadata.csv
        batch_size: Batch size for training data (None = no batching, batch per experiment)
        val_batch_size: Batch size for validation/test data (default: same as batch_size)
        shuffle_train: Whether to shuffle training data
        cache_train: Whether to cache training data in memory (only for small datasets)
        cache_val: Whether to cache validation data (default: True for validation/test)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        - train_dataset: tf.data.Dataset (memory-efficient generator)
        - val_dataset: tf.data.Dataset (cached generator) or None
        - test_dataset: tf.data.Dataset (cached generator) or None
    """
    if val_batch_size is None:
        val_batch_size = batch_size
    
    # If batch_size is None, use a default for validation/test to ensure they're batched
    if val_batch_size is None:
        val_batch_size = 32  # Default batch size for validation/test
    
    # Helper to find CSV file
    def find_csv_path(path_input):
        path_obj = Path(path_input)
        if path_obj.is_file() and path_obj.suffix == '.csv':
            return path_obj
        elif path_obj.is_dir():
            # Try parent directory first
            csv_file = path_obj.parent / f'{path_obj.name}_metadata.csv'
            if csv_file.exists():
                return csv_file
            # Fallback: look in the directory itself
            csv_file = path_obj / f'{path_obj.name}_metadata.csv'
            if csv_file.exists():
                return csv_file
            # Try common names
            for name in ['train_metadata.csv', 'validation_metadata.csv', 'val_metadata.csv', 'test_metadata.csv']:
                csv_file = path_obj.parent / name
                if csv_file.exists():
                    return csv_file
                csv_file = path_obj / name
                if csv_file.exists():
                    return csv_file
        raise FileNotFoundError(f"Could not find metadata CSV for: {path_input}")
    
    # Create training dataset
    train_csv = find_csv_path(train_path)
    train_dataset = create_dataset_from_csv(
        train_csv,
        batch_size=batch_size,
        shuffle=shuffle_train,
        cache=cache_train
    )
    
    # Create validation dataset
    # Use generator with caching (now that CSV is shuffled, this should work reliably)
    # Note: validation/test datasets should always be batched for evaluation
    val_dataset = None
    if val_path is not None:
        val_csv = find_csv_path(val_path)
        # Ensure validation dataset is batched (required for evaluation)
        val_batch = val_batch_size if val_batch_size is not None else 32
        val_dataset = create_dataset_from_csv(
            val_csv,
            batch_size=val_batch,  # Always batch validation for evaluation
            shuffle=False,  # Don't shuffle validation
            cache=True  # Cache validation dataset to allow multiple iterations
        )
    
    # Create test dataset
    # Use generator with caching (consistent with validation approach)
    # Note: test datasets should always be batched for evaluation
    test_dataset = None
    if test_path is not None:
        test_csv = find_csv_path(test_path)
        # Ensure test dataset is batched (required for evaluation)
        test_batch = val_batch_size if val_batch_size is not None else 32
        test_dataset = create_dataset_from_csv(
            test_csv,
            batch_size=test_batch,  # Always batch test for evaluation
            shuffle=False,  # Don't shuffle test
            cache=True  # Cache test dataset
        )
    
    return train_dataset, val_dataset, test_dataset
