# models/utils.py
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_image_data(data_dir='../processed_dfire'):
    """Load actual image data from the processed directories"""
    splits = {}
    data_path = Path(data_dir)
    
    print(f"Looking for images in: {data_path.absolute()}")
    
    for split in ['train', 'validation', 'test']:
        split_path = data_path / split
        if split_path.exists():
            images = []
            labels = []
            
            # Go through each class folder
            for class_name in ['fire', 'smoke', 'nothing']:
                class_path = split_path / class_name
                if class_path.exists():
                    png_files = list(class_path.glob('*.png'))
                    print(f"Found {len(png_files)} images in {class_path}")
                    for img_file in png_files:
                        images.append(str(img_file))
                        labels.append(class_name)
            
            splits[split] = {
                'images': images,
                'labels': labels
            }
            print(f"Loaded {len(images)} images from {split} split")
            print(f"Class distribution in {split}: {np.unique(labels, return_counts=True)}")
        else:
            print(f"Missing: {split_path}")
    
    return splits

def extract_simple_features(image_paths, labels, max_samples=2000):
    """Extract simple features from actual images - with balanced sampling"""
    features = []
    valid_paths = []
    valid_labels = []
    
    print(f"Extracting features from {min(len(image_paths), max_samples)} actual images...")
    
    # Shuffle the data to get balanced classes
    from sklearn.utils import shuffle
    shuffled_paths, shuffled_labels = shuffle(image_paths, labels, random_state=42)
    
    # Count how many samples per class we need
    class_counts = {}
    samples_per_class = max_samples // 3  # Roughly equal for 3 classes
    
    for i, (img_path, label) in enumerate(zip(shuffled_paths, shuffled_labels)):
        if len(features) >= max_samples:
            break
            
        # Limit samples per class to maintain balance
        if class_counts.get(label, 0) >= samples_per_class and len(features) >= samples_per_class * 2:
            continue
            
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Simple RGB statistics from ACTUAL PIXELS
            flattened = img_array.reshape(-1, 3)
            mean_rgb = flattened.mean(axis=0)
            std_rgb = flattened.std(axis=0)
            
            # Additional features from ACTUAL PIXELS
            brightness = img_array.mean()
            contrast = img_array.std()
            
            # Color channel ratios from ACTUAL PIXELS
            r, g, b = mean_rgb
            rg_ratio = r / (g + 1e-8)
            rb_ratio = r / (b + 1e-8)
            
            feature_vector = np.concatenate([
                mean_rgb,           # 3 features from pixels
                std_rgb,            # 3 features from pixels  
                [brightness],       # 1 feature from pixels
                [contrast],         # 1 feature from pixels
                [rg_ratio, rb_ratio] # 2 features from pixels
            ])
            
            features.append(feature_vector)
            valid_paths.append(img_path)
            valid_labels.append(label)
            
            # Update class count
            class_counts[label] = class_counts.get(label, 0) + 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Successfully extracted features from {len(features)} images")
    print(f"Class distribution in extracted features: {np.unique(valid_labels, return_counts=True)}")
    return np.array(features), valid_paths, valid_labels

def get_labels_from_paths(all_images, all_labels, valid_paths):
    """Get labels corresponding to valid image paths"""
    # Create a mapping from path to label
    path_to_label = dict(zip(all_images, all_labels))
    
    # Return labels for valid paths
    labels = []
    for path in valid_paths:
        # Convert to string and normalize path for comparison
        path_str = str(path)
        if path_str in path_to_label:
            labels.append(path_to_label[path_str])
        else:
            # Try with different path separators
            path_alt = path_str.replace('\\', '/')
            for img_path, label in zip(all_images, all_labels):
                if img_path.replace('\\', '/') == path_alt:
                    labels.append(label)
                    break
            else:
                print(f"Warning: Could not find label for {path_str}")
    
    print(f"Unique labels found: {set(labels)}")
    print(f"Label distribution: {np.unique(labels, return_counts=True)}")
    return labels

def save_model(model, scaler, model_name, output_dir='saved_models'):
    """Save trained model and scaler"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model_path = output_path / f'{model_name}.pkl'
    scaler_path = output_path / f'{model_name}_scaler.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Saved {model_name} to {model_path}")
    print(f"Saved scaler to {scaler_path}")

def load_model(model_name, output_dir='saved_models'):
    """Load trained model and scaler"""
    output_path = Path(output_dir)
    
    model_path = output_path / f'{model_name}.pkl'
    scaler_path = output_path / f'{model_name}_scaler.pkl'
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Loaded {model_name} from {model_path}")
    return model, scaler