import csv
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

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
            
            # Get label index
            label_idx = int(row['label_idx'])
            labels.append(label_idx)
    
    X = np.array(images)
    y = np.array(labels)
    # Return as numpy arrays (will convert to tensors after splitting if needed)
    return (X, y)

def load_data_from_directory(dir_path):
    """Load images and labels from a directory structure (split/fire/ and split/no_fire/)."""
    images = []
    labels = []
    
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Look for fire and no_fire subdirectories
    for label_name in ['fire', 'no_fire']:
        label_dir = dir_path / label_name
        if not label_dir.exists():
            continue
        
        label_idx = 1 if label_name == 'fire' else 0
        
        # Load all images from this label directory
        # Images are already processed: RGB, square, saved as PNG
        for img_path in sorted(label_dir.glob('*.png')):
            with Image.open(img_path) as img:
                # Images are already RGB and square from process_dfire.py
                img_array = np.array(img, dtype=np.float32)
                # Normalize to [0, 1] range
                img_array = img_array / 255.0
                images.append(img_array)
                labels.append(label_idx)
    
    X = np.array(images)
    y = np.array(labels)
    # Return as numpy arrays (will convert to tensors after splitting if needed)
    return (X, y)

def retrieve_data(train_path, test_path=None, val_size=0.2, random_state=42):
    '''
    Retrieves outputs of "process_dfire.py" and converts into tensors for model training.
    
    Optionally splits training data into train/validation sets.

    args:
        train_path (string): filepath locating the train data (metadata CSV file or directory)
        test_path (string): filepath locating the test data. If this is none, test process will not run
        val_size (float): Proportion of training data to use for validation (default 0.2 = 20%).
                         If None, no split is performed and all training data is returned.
        random_state (int): Random seed for reproducibility

    returns:
        train_data (tuple): (X_train, y_train) as TensorFlow tensors
        val_data (tuple): (X_val, y_val) as TensorFlow tensors, or None if val_size is None
        test_data (tuple): (X_test, y_test) as TensorFlow tensors, or None if test_path is None
    '''
    # Determine if train_path is a CSV file or directory
    train_path_obj = Path(train_path)
    if train_path_obj.is_file() and train_path_obj.suffix == '.csv':
        train_data = load_data_from_csv(train_path)
    elif train_path_obj.is_dir():
        train_data = load_data_from_directory(train_path)
    else:
        # Try to find metadata CSV in the directory
        csv_file = train_path_obj / 'train_metadata.csv'
        if csv_file.exists():
            train_data = load_data_from_csv(csv_file)
        else:
            train_data = load_data_from_directory(train_path)
    
    # Split training data into train/val if requested (before converting to tensors)
    val_data = None
    if val_size is not None and val_size > 0:
        X, y = train_data
        # Use stratified split to maintain class balance
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_size,
            random_state=random_state,
            stratify=y  # Maintains class distribution
        )
        # Convert to tensors after splitting
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
    else:
        # Convert entire training set to tensors
        X, y = train_data
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.int32)
        train_data = (X, y)
    
    # Load test data if provided
    test_data = None
    if test_path is not None:
        test_path_obj = Path(test_path)
        if test_path_obj.is_file() and test_path_obj.suffix == '.csv':
            test_data = load_data_from_csv(test_path)
        elif test_path_obj.is_dir():
            test_data = load_data_from_directory(test_path)
        else:
            # Try to find metadata CSV in the directory
            csv_file = test_path_obj / 'test_metadata.csv'
            if csv_file.exists():
                test_data = load_data_from_csv(csv_file)
            else:
                test_data = load_data_from_directory(test_path)
        
        # Convert test data to tensors
        X_test, y_test = test_data
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
        test_data = (X_test, y_test)
    
    return train_data, val_data, test_data

