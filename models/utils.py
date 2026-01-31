import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib
import os
from scipy import ndimage
from scipy.stats import skew, kurtosis

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
            
            for class_name in ['fire', 'smoke', 'nothing', 'fire_and_smoke']: 
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
    """Extract ORIGINAL SIMPLE features (10 features total)"""
    features = []
    valid_paths = []
    valid_labels = []
    
    print(f"Extracting ORIGINAL SIMPLE features from {min(len(image_paths), max_samples)} actual images...")
    
    from sklearn.utils import shuffle
    shuffled_paths, shuffled_labels = shuffle(image_paths, labels, random_state=42)
    
    class_counts = {}
    samples_per_class = max_samples // 4
    
    for i, (img_path, label) in enumerate(zip(shuffled_paths, shuffled_labels)):
        if len(features) >= max_samples:
            break
            
        if class_counts.get(label, 0) >= samples_per_class and len(features) >= samples_per_class * 3:
            continue
            
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Simple RGB statistics from ACTUAL PIXELS
            flattened = img_array.reshape(-1, 3)
            mean_rgb = flattened.mean(axis=0)
            std_rgb = flattened.std(axis=0)
            
            brightness = img_array.mean()
            contrast = img_array.std()
            
            r, g, b = mean_rgb
            rg_ratio = r / (g + 1e-8)
            rb_ratio = r / (b + 1e-8)
            
            feature_vector = np.array([
                mean_rgb[0], mean_rgb[1], mean_rgb[2],  # 3 mean RGB
                std_rgb[0], std_rgb[1], std_rgb[2],     # 3 std RGB
                brightness,                             # 1 brightness
                contrast,                               # 1 contrast
                rg_ratio, rb_ratio                      # 2 ratios
            ])  # TOTAL: 10 features
            
            features.append(feature_vector)
            valid_paths.append(img_path)
            valid_labels.append(label)
            
            # Update class count
            class_counts[label] = class_counts.get(label, 0) + 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Successfully extracted ORIGINAL SIMPLE features from {len(features)} images")
    print(f"Feature dimension: {len(features[0]) if features else 0}")
    print(f"Class distribution: {np.unique(valid_labels, return_counts=True)}")
    return np.array(features), valid_paths, valid_labels

def get_labels_from_paths(all_images, all_labels, valid_paths):
    """Get labels corresponding to valid image paths"""
    path_to_label = dict(zip(all_images, all_labels))
    
    labels = []
    for path in valid_paths:
        path_str = str(path)
        if path_str in path_to_label:
            labels.append(path_to_label[path_str])
        else:
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

def extract_enhanced_features(image_paths, labels, max_samples=2000):
    """Simpler enhanced features that avoid complex calculations"""
    features = []
    valid_paths = []
    valid_labels = []
    
    print(f"Extracting SIMPLE ENHANCED features from {min(len(image_paths), max_samples)} actual images...")
    
    from sklearn.utils import shuffle
    shuffled_paths, shuffled_labels = shuffle(image_paths, labels, random_state=42)
    
    class_counts = {}
    samples_per_class = max_samples // 4
    
    for i, (img_path, label) in enumerate(zip(shuffled_paths, shuffled_labels)):
        if len(features) >= max_samples:
            break
            
        if class_counts.get(label, 0) >= samples_per_class and len(features) >= samples_per_class * 3:
            continue
            
        try:
            img = Image.open(img_path)
            img_array = np.array(img)
            height, width, _ = img_array.shape
            
            flattened = img_array.reshape(-1, 3)
            mean_rgb = flattened.mean(axis=0)
            std_rgb = flattened.std(axis=0)
            
            r, g, b = mean_rgb
            rg_ratio = r / (g + 1e-8)
            rb_ratio = r / (b + 1e-8)
            
            gray = np.mean(img_array, axis=2)
            brightness = gray.mean()
            contrast = gray.std()
            
            rgb_sum = r + g + b + 1e-8
            red_dominance = r / rgb_sum
            green_dominance = g / rgb_sum
            blue_dominance = b / rgb_sum
            
            try:
                diff_h = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
                diff_v = np.abs(gray[1:, :] - gray[:-1, :]).mean()
                edge_score = (diff_h + diff_v) / 2
            except:
                edge_score = 0
            
            r_var = np.var(img_array[:,:,0])
            g_var = np.var(img_array[:,:,1])
            b_var = np.var(img_array[:,:,2])
            avg_var = (r_var + g_var + b_var) / 3
            
            h_mid = height // 2
            w_mid = width // 2
            
            top_left = img_array[:h_mid, :w_mid]
            top_right = img_array[:h_mid, w_mid:]
            bottom_left = img_array[h_mid:, :w_mid]
            bottom_right = img_array[h_mid:, w_mid:]
            
            q1_bright = np.mean(top_left) if top_left.size > 0 else 0
            q2_bright = np.mean(top_right) if top_right.size > 0 else 0
            q3_bright = np.mean(bottom_left) if bottom_left.size > 0 else 0
            q4_bright = np.mean(bottom_right) if bottom_right.size > 0 else 0
            
            spatial_variation = np.std([q1_bright, q2_bright, q3_bright, q4_bright]) if len([q1_bright, q2_bright, q3_bright, q4_bright]) > 1 else 0
            
            rg_diff = np.abs(r - g) / (np.max([r, g]) + 1e-8)
            rb_diff = np.abs(r - b) / (np.max([r, b]) + 1e-8)
            gb_diff = np.abs(g - b) / (np.max([g, b]) + 1e-8)
            
            feature_vector = np.array([
                mean_rgb[0], mean_rgb[1], mean_rgb[2],
                std_rgb[0], std_rgb[1], std_rgb[2],
                
                rg_ratio, rb_ratio,
                
                brightness, contrast,
                
                red_dominance, green_dominance, blue_dominance,
                
                edge_score,
                
                r_var, g_var, b_var, avg_var,
                
                spatial_variation,
                
                q1_bright, q2_bright, q3_bright, q4_bright,
                
                rg_diff, rb_diff, gb_diff
            ])
            
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            
            features.append(feature_vector)
            valid_paths.append(img_path)
            valid_labels.append(label)
            
            class_counts[label] = class_counts.get(label, 0) + 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Successfully extracted SIMPLE ENHANCED features from {len(features)} images")
    print(f"Feature dimension: {len(features[0]) if features else 0}")
    print(f"Class distribution: {np.unique(valid_labels, return_counts=True)}")
    return np.array(features), valid_paths, valid_labels
