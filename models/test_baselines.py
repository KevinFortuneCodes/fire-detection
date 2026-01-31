import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from utils import load_image_data, extract_simple_features, extract_enhanced_features, load_model
import argparse

def save_confusion_matrix(cm, class_names, model_name, output_dir='confusion_matrices'):
    """Save a single confusion matrix as PNG and CSV"""
    Path(output_dir).mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    png_path = Path(output_dir) / f'{model_name}_confusion_matrix.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved confusion matrix to: {png_path}")
    
    csv_path = Path(output_dir) / f'{model_name}_confusion_matrix.csv'
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(csv_path)
    print(f"  Saved confusion matrix data to: {csv_path}")
    
    return str(png_path), str(csv_path)

def test_all_models(max_test_samples=1000, feature_type='enhanced', tuned=False):
    """Test all trained models with specified feature type"""
    print(f"=== TESTING BASELINE MODELS ({feature_type.upper()} FEATURES) ===")
    if tuned:
        print("Testing TUNED models")
    
    splits = load_image_data()
    
    if not splits or 'test' not in splits:
        print("No test images found!")
        return
    
    test_images = splits['test']['images']
    test_labels = splits['test']['labels']

    print(f"Available test images: {len(test_images)}")

    if feature_type == 'enhanced':
        X_test, test_valid_paths, y_test = extract_enhanced_features(
            test_images, test_labels, max_samples=max_test_samples
        )
    else:
        X_test, test_valid_paths, y_test = extract_simple_features(
            test_images, test_labels, max_samples=max_test_samples
        )

    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Test class distribution: {np.unique(y_test, return_counts=True)}")
    
    suffix = f"_{feature_type}"
    if tuned:
        suffix += "_tuned"
    
    model_names = ['knn', 'decision_tree', 'random_forest']
    
    all_results = {}
    class_names = ['fire', 'smoke', 'nothing', 'fire_and_smoke']
    
    print(f"\nCreating confusion matrices directory: 'confusion_matrices'")
    
    for model_name in model_names:
        try:
            full_model_name = f"{model_name}{suffix}"
            print(f"\nTesting {full_model_name}...")
            model, scaler = load_model(full_model_name)
            
            X_test_scaled = scaler.transform(X_test)
            
            y_pred = model.predict(X_test_scaled)
            
            cm = confusion_matrix(y_test, y_pred, labels=class_names)
            
            print(f"\nSaving confusion matrix for {full_model_name}...")
            png_path, csv_path = save_confusion_matrix(cm, class_names, full_model_name)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            print(f"\n{'='*60}")
            print(f"RESULTS FOR {full_model_name.upper()}")
            print(f"{'='*60}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names))
            
            all_results[full_model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'confusion_matrix': cm,
                'confusion_matrix_png': png_path,
                'confusion_matrix_csv': csv_path
            }
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")

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
    print(f"MODEL COMPARISON ({feature_type.upper()} FEATURES{' - TUNED' if tuned else ''})")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))

    csv_name = f'model_comparison_{feature_type}'
    if tuned:
        csv_name += '_tuned'
    csv_name += '.csv'
    comparison_df.to_csv(csv_name, index=False)
    print(f"\nSaved comparison results to '{csv_name}'")
    
    create_summary_report(all_results, feature_type, tuned)
    
    return all_results

def create_summary_report(all_results, feature_type, tuned):
    """Create a comprehensive summary report"""
    report_dir = Path('model_reports')
    report_dir.mkdir(exist_ok=True)
    
    report_name = f'model_summary_{feature_type}'
    if tuned:
        report_name += '_tuned'
    report_name += '.txt'
    
    report_path = report_dir / report_name
    
    with open(report_path, 'w') as f:
        f.write(f"MODEL TESTING SUMMARY REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Feature Type: {feature_type.upper()}\n")
        f.write(f"Tuned: {tuned}\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"{'='*60}\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"MODEL: {model_name.upper()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall:    {results['recall']:.4f}\n")
            f.write(f"F1-Score:  {results['f1_score']:.4f}\n")
            f.write(f"\nConfusion Matrix Files:\n")
            f.write(f"  PNG: {results['confusion_matrix_png']}\n")
            f.write(f"  CSV: {results['confusion_matrix_csv']}\n")
    
    print(f"\nSaved detailed summary report to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=4306, help="Maximum test samples to use")
    parser.add_argument("--feature-type", choices=['simple', 'enhanced'], default='enhanced',
                       help="Type of features to use: 'simple' or 'enhanced'")
    parser.add_argument("--tuned", action="store_true", help="Test tuned models instead of default")
    args = parser.parse_args()
    
    test_all_models(
        max_test_samples=args.max_samples,
        feature_type=args.feature_type,
        tuned=args.tuned
    )
