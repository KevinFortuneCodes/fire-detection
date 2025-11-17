# models/train_baselines.py
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils import load_image_data, extract_simple_features, get_labels_from_paths, save_model
from sklearn.preprocessing import StandardScaler

def train_knn(X_train, y_train):
    """Train KNN model"""
    from sklearn.neighbors import KNeighborsClassifier
    print("Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """Train Decision Tree model"""
    from sklearn.tree import DecisionTreeClassifier
    print("Training Decision Tree model...")
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    from sklearn.ensemble import RandomForestClassifier
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_all_models(max_samples=2000):
    """Train all baseline models on ACTUAL IMAGE DATA"""
    print("=== TRAINING BASELINE MODELS ON ACTUAL IMAGES ===")
    print(f"Using {max_samples} samples per model")
    
    # Load ACTUAL IMAGE DATA
    splits = load_image_data()
    
    if not splits or 'train' not in splits:
        print("No training images found! Please run preprocessing first.")
        return
    
    # Extract features from ACTUAL TRAINING IMAGES with balanced sampling
    train_images = splits['train']['images']
    train_labels = splits['train']['labels']
    
    print(f"Available training images: {len(train_images)}")
    
    X_train, train_valid_paths, y_train = extract_simple_features(train_images, train_labels, max_samples=max_samples)
    
    print(f"Final training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Class distribution: {np.unique(y_train, return_counts=True)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train models
    models = {}
    
    # KNN
    knn_model = train_knn(X_train_scaled, y_train)
    models['knn'] = knn_model
    
    # Decision Tree
    dt_model = train_decision_tree(X_train_scaled, y_train)
    models['decision_tree'] = dt_model
    
    # Random Forest
    rf_model = train_random_forest(X_train_scaled, y_train)
    models['random_forest'] = rf_model
    
    # Save models and scaler
    for model_name, model in models.items():
        save_model(model, scaler, model_name)
    
    print("\n=== TRAINING COMPLETE ===")
    print("All models saved to 'saved_models' directory")

if __name__ == "__main__":
    train_all_models(max_samples=2000)