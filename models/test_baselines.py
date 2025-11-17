# models/test_baselines.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_image_data, extract_simple_features, get_labels_from_paths, load_model

def evaluate_model(model, scaler, X_test, y_test, model_name, class_names):
    """Comprehensive model evaluation"""
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics - specify labels to ensure all classes are considered
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0, labels=class_names)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0, labels=class_names)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=class_names)
    
    # Class-wise metrics
    class_report = classification_report(y_test, y_pred, output_dict=True, labels=class_names)
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'predictions': y_pred
    }
    
    return results

def print_results(results, model_name, y_test, class_names):
    """Print formatted results"""
    print(f"\n{'='*60}")
    print(f"RESULTS FOR {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, results['predictions'], target_names=class_names))

def plot_confusion_matrices(all_results, class_names):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx])
        axes[idx].set_title(f'{model_name.upper()} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_all_models(max_test_samples=1000):
    """Test all trained models"""
    print("=== TESTING BASELINE MODELS ===")
    
    # Load test data (ACTUAL IMAGES)
    splits = load_image_data()
    
    if not splits or 'test' not in splits:
        print("No test images found!")
        return
    
    # Extract features from ACTUAL TEST IMAGES with balanced sampling
    test_images = splits['test']['images']
    test_labels = splits['test']['labels']

    print(f"Available test images: {len(test_images)}")

    X_test, test_valid_paths, y_test = extract_simple_features(test_images, test_labels, max_samples=max_test_samples)

    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Test class distribution: {np.unique(y_test, return_counts=True)}")
    
    # Test each model
    model_names = ['knn', 'decision_tree', 'random_forest']
    all_results = {}
    class_names = ['fire', 'smoke', 'nothing']
    
    for model_name in model_names:
        try:
            print(f"\nTesting {model_name}...")
            model, scaler = load_model(model_name)
            results = evaluate_model(model, scaler, X_test, y_test, model_name, class_names)
            all_results[model_name] = results
            print_results(results, model_name, y_test, class_names)
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    # Create comparison table
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append({
            'Model': model_name.upper(),
            'Accuracy': f"{results['accuracy']:.4f}",
            'Precision': f"{results['precision']:.4f}",
            'Recall': f"{results['recall']:.4f}",
            'F1-Score': f"{results['f1_score']:.4f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))
    
    # Save results to CSV
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print(f"\nSaved results to 'model_comparison_results.csv'")
    
    # Plot confusion matrices
    plot_confusion_matrices(all_results, class_names)
    
    return all_results

if __name__ == "__main__":
    test_all_models(max_test_samples=1000)