import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

def load_data_from_csv(csv_path):
    """Load images and labels from a metadata CSV file."""
    images = []
    labels = []
    
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
                images.append(img_array)
            
            # Get label index and convert from 1-based (1,2,3) to 0-based (0,1,2) for TensorFlow
            label_idx = int(row['label_idx'])
            # Convert: fire=1->0, smoke=2->1, nothing=3->2
            label_idx_0based = label_idx - 1
            labels.append(label_idx_0based)
    
    X = np.array(images)
    y = np.array(labels)
    # Return as numpy arrays (will convert to tensors later)
    return (X, y)

def retrieve_data(train_path, test_path=None, val_path=None):
    '''
    Retrieves outputs of "process_dfire.py" and returns as numpy arrays for model training.
    Data is kept in CPU memory to prevent GPU OOM errors; TensorFlow handles GPU transfer during training.
    
    Loads validation data from val_path if provided.

    args:
        train_path (string): filepath to the train metadata CSV file, or directory containing train_metadata.csv
        test_path (string): filepath to the test metadata CSV file, or directory containing test_metadata.csv. If None, test process will not run
        val_path (string): filepath to the validation metadata CSV file, or directory containing validation_metadata.csv. If None, val_data will be None.

    returns:
        train_data (tuple): (X_train, y_train) as numpy arrays (TensorFlow will handle GPU transfer during training)
        val_data (tuple): (X_val, y_val) as numpy arrays, or None if val_path is None
        test_data (tuple): (X_test, y_test) as numpy arrays, or None if test_path is None
    '''
    # Load training data from CSV file
    train_path_obj = Path(train_path)
    if train_path_obj.is_file() and train_path_obj.suffix == '.csv':
        train_data = load_data_from_csv(train_path)
    elif train_path_obj.is_dir():
        # Look for metadata CSV in the parent directory (where phase1 writes them)
        csv_file = train_path_obj.parent / 'train_metadata.csv'
        if csv_file.exists():
            train_data = load_data_from_csv(csv_file)
        else:
            # Fallback: look in the directory itself
            csv_file = train_path_obj / 'train_metadata.csv'
            if csv_file.exists():
                train_data = load_data_from_csv(csv_file)
            else:
                raise FileNotFoundError(f"Train metadata CSV not found at {train_path_obj.parent}/train_metadata.csv or {csv_file}")
    else:
        raise FileNotFoundError(f"Train path must be a CSV file or directory: {train_path}")
    
    # Load validation data from val_path if provided
    val_data = None
    if val_path is not None:
        # Load validation data from the provided path
        val_path_obj = Path(val_path)
        if val_path_obj.is_file() and val_path_obj.suffix == '.csv':
            val_data = load_data_from_csv(val_path)
        elif val_path_obj.is_dir():
            # Look for metadata CSV in the parent directory (where phase1 writes them)
            csv_file = val_path_obj.parent / 'validation_metadata.csv'
            if csv_file.exists():
                val_data = load_data_from_csv(csv_file)
            else:
                # Fallback: look in the directory itself
                csv_file = val_path_obj / 'validation_metadata.csv'
                if csv_file.exists():
                    val_data = load_data_from_csv(csv_file)
                else:
                    raise FileNotFoundError(f"Validation metadata CSV not found at {val_path_obj.parent}/validation_metadata.csv or {csv_file}")
        else:
            raise FileNotFoundError(f"Validation path must be a CSV file or directory: {val_path}")
        
        # Keep validation data as numpy arrays (TensorFlow will handle GPU transfer during training)
        # Converting to tensors here causes OOM by trying to allocate all data on GPU at once
        X_val, y_val = val_data
        # Ensure correct dtypes for TensorFlow
        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int32)
        val_data = (X_val, y_val)
    
    # Keep training data as numpy arrays (TensorFlow will handle GPU transfer during training)
    X, y = train_data
    # Ensure correct dtypes for TensorFlow
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    train_data = (X, y)
    
    # Load test data if provided
    test_data = None
    if test_path is not None:
        test_path_obj = Path(test_path)
        if test_path_obj.is_file() and test_path_obj.suffix == '.csv':
            test_data = load_data_from_csv(test_path)
        elif test_path_obj.is_dir():
            # Look for metadata CSV in the parent directory (where phase1 writes them)
            csv_file = test_path_obj.parent / 'test_metadata.csv'
            if csv_file.exists():
                test_data = load_data_from_csv(csv_file)
            else:
                # Fallback: look in the directory itself
                csv_file = test_path_obj / 'test_metadata.csv'
                if csv_file.exists():
                    test_data = load_data_from_csv(csv_file)
                else:
                    raise FileNotFoundError(f"Test metadata CSV not found at {test_path_obj.parent}/test_metadata.csv or {csv_file}")
        else:
            raise FileNotFoundError(f"Test path must be a CSV file or directory: {test_path}")
        
        # Keep test data as numpy arrays (TensorFlow will handle GPU transfer during evaluation)
        X_test, y_test = test_data
        # Ensure correct dtypes for TensorFlow
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)
        test_data = (X_test, y_test)
    
    return train_data, val_data, test_data

