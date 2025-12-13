import sys
import numpy as np
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent))

from utils import load_image_data, extract_simple_features, extract_enhanced_features, save_model
from sklearn.preprocessing import StandardScaler

def train_knn_default(X_train, y_train):
    """Train KNN model with default parameters"""
    from sklearn.neighbors import KNeighborsClassifier
    print("Training KNN model with default parameters...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    return model

def train_decision_tree_default(X_train, y_train):
    """Train Decision Tree model with default parameters"""
    from sklearn.tree import DecisionTreeClassifier
    print("Training Decision Tree model with default parameters...")
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest_default(X_train, y_train):
    """Train Random Forest model with default parameters"""
    from sklearn.ensemble import RandomForestClassifier
    print("Training Random Forest model with default parameters...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_all_models(max_samples=4000, feature_type='enhanced', tune_hyperparams=True):
    """Train all baseline models with optional hyperparameter tuning"""
    print(f"=== TRAINING BASELINE MODELS ({feature_type.upper()} FEATURES) ===")
    print(f"Using {max_samples} samples per model")
    print(f"Feature type: {feature_type}")
    print(f"Hyperparameter tuning: {tune_hyperparams}")
    
    splits = load_image_data()
    
    if not splits or 'train' not in splits:
        print("No training images found! Please run preprocessing first.")
        return
    
    train_images = splits['train']['images']
    train_labels = splits['train']['labels']
    
    print(f"Available training images: {len(train_images)}")
    
    if feature_type == 'enhanced':
        X_train, train_valid_paths, y_train = extract_enhanced_features(
            train_images, train_labels, max_samples=max_samples
        )
    else:
        X_train, train_valid_paths, y_train = extract_simple_features(
            train_images, train_labels, max_samples=max_samples
        )
    
    print(f"Final training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Class distribution: {np.unique(y_train, return_counts=True)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {}
    
    if tune_hyperparams:
        print("\n--- Starting Hyperparameter Tuning ---")
        
        try
            from hyperparameter_tuning import tune_knn, tune_decision_tree, tune_random_forest
            
            print("\n" + "="*50)
            knn_model = tune_knn(X_train_scaled, y_train, cv=3, n_iter=20, random_state=42)
            models['knn'] = knn_model
            
            print("\n" + "="*50)
            dt_model = tune_decision_tree(X_train_scaled, y_train, cv=3, n_iter=30, random_state=42)
            models['decision_tree'] = dt_model
            
            print("\n" + "="*50)
            rf_model = tune_random_forest(X_train_scaled, y_train, cv=3, n_iter=30, random_state=42)
            models['random_forest'] = rf_model
            
        except ImportError as e:
            print(f"\nWarning: Could not import hyperparameter_tuning: {e}")
            print("Falling back to default parameters...")
            tune_hyperparams = False
    
    if not tune_hyperparams:
        print("\n--- Training with Default Parameters ---")
        
        knn_model = train_knn_default(X_train_scaled, y_train)
        models['knn'] = knn_model
        
        dt_model = train_decision_tree_default(X_train_scaled, y_train)
        models['decision_tree'] = dt_model
        
        rf_model = train_random_forest_default(X_train_scaled, y_train)
        models['random_forest'] = rf_model
    
    suffix = f"_{feature_type}"
    if tune_hyperparams:
        suffix += "_tuned"
    
    for model_name, model in models.items():
        save_model(model, scaler, f"{model_name}{suffix}")
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"All models saved to 'saved_models' directory")
    print(f"Model suffixes: {suffix}")
    
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train baseline models with optional hyperparameter tuning"
    )
    parser.add_argument("--max-samples", type=int, default=17221, 
                       help="Maximum samples to use for training")
    parser.add_argument("--feature-type", choices=['simple', 'enhanced'], default='enhanced',
                       help="Type of features to use: 'simple' or 'enhanced'")
    parser.add_argument("--no-tuning", action="store_true", 
                       help="Skip hyperparameter tuning (use default parameters)")
    args = parser.parse_args()
    
    train_all_models(
        max_samples=args.max_samples,
        feature_type=args.feature_type,
        tune_hyperparams=not args.no_tuning
    )
