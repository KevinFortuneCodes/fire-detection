from src.experiments.experiments import EXPERIMENTS
import json
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, average_precision_score, f1_score,
    confusion_matrix, classification_report
)

def evaluate_model(model, test_data):
    """
    Evaluate the model on test data for 4-class classification (fire, smoke, nothing, fire_and_smoke).
    
    Args:
        model: Trained Keras model
        test_data: Test data - can be either:
            - (X_test, y_test) tuple of numpy arrays
            - tf.data.Dataset (memory-efficient, loads images on-demand)
    
    Returns:
        Dictionary containing:
        - test_loss: Loss on test set
        - test_accuracy: Accuracy on test set
        - precision_macro: Precision (macro-averaged, treats all classes equally)
        - recall_macro: Recall (macro-averaged)
        - f1_score_macro: F1-score (macro-averaged)
        - precision_weighted: Precision (weighted by class frequency, better for imbalanced classes)
        - recall_weighted: Recall (weighted by class frequency)
        - f1_score_weighted: F1-score (weighted by class frequency)
        - precision, recall, f1_score: Legacy fields (using weighted values)
        - mAP: Mean Average Precision (mean of per-class average precision scores)
        - precision_per_class: Dict with precision for each class (fire, smoke, nothing, fire_and_smoke)
        - recall_per_class: Dict with recall for each class (fire, smoke, nothing, fire_and_smoke)
        - f1_per_class: Dict with F1-score for each class (fire, smoke, nothing, fire_and_smoke)
        - class_support: Dict with number of samples per class
        - confusion_matrix: 4x4 confusion matrix as list of lists
    """
    # Check if test_data is a tf.data.Dataset or numpy arrays
    import tensorflow as tf
    if isinstance(test_data, tf.data.Dataset):
        # For tf.data.Dataset, we need to collect all predictions and labels
        # This is less memory-efficient but necessary for evaluation metrics
        print("Evaluating with tf.data.Dataset - collecting predictions...")
        all_pred_logits = []
        all_labels = []
        
        for batch_images, batch_labels in test_data:
            # Debug: Check image shape
            if len(batch_images.shape) != 4:
                print(f"WARNING: Expected batch_images shape (batch, 256, 256, 3), got {batch_images.shape}")
                # If shape is wrong, try to reshape
                if len(batch_images.shape) == 3:
                    # Might be (batch, height*width, channels) or similar
                    print(f"  Attempting to reshape...")
                    # This shouldn't happen, but let's handle it
                    raise ValueError(f"Unexpected image shape: {batch_images.shape}. Expected (batch, 256, 256, 3)")
            batch_pred = model.predict(batch_images, verbose=0)
            all_pred_logits.append(batch_pred)
            all_labels.append(batch_labels.numpy())
        
        # Concatenate all batches
        y_pred_logits = np.concatenate(all_pred_logits, axis=0)
        y_test_np = np.concatenate(all_labels, axis=0)
        
        # Get loss and accuracy (need to evaluate on dataset)
        eval_results = model.evaluate(test_data, verbose=0, return_dict=True)
        test_loss = eval_results.get('loss', 0.0)
        test_acc = eval_results.get('accuracy', 0.0)
    else:
        # Numpy arrays - original code
        X_test, y_test = test_data
        
        # Convert tensors to numpy if needed
        if isinstance(y_test, tf.Tensor):
            y_test_np = y_test.numpy()
        else:
            y_test_np = y_test
        
        # Get loss and accuracy from model.evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions (logits)
        y_pred_logits = model.predict(X_test, verbose=0)
    
    # Convert logits to probabilities using softmax
    y_pred_probs = tf.nn.softmax(y_pred_logits).numpy()
    
    # Get predicted classes (argmax)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate precision and recall with different averaging strategies
    # Macro: unweighted mean of per-class metrics (treats all classes equally)
    # Weighted: mean weighted by class support (better for imbalanced classes)
    precision_macro = precision_score(y_test_np, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test_np, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test_np, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_test_np, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test_np, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test_np, y_pred, average='weighted', zero_division=0)
    
    # Calculate per-class metrics for detailed analysis (essential for imbalanced classes)
    precision_per_class = precision_score(y_test_np, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test_np, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test_np, y_pred, average=None, zero_division=0)
    
    # Class names for 4-class classification: fire, smoke, nothing, fire_and_smoke
    class_names = ['fire', 'smoke', 'nothing', 'fire_and_smoke']
    
    # Calculate class support (number of samples per class) for context
    from collections import Counter
    class_counts = Counter(y_test_np)
    class_support = {class_names[i]: int(class_counts.get(i, 0)) 
                     for i in range(len(class_names))}
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_np, y_pred)
    
    # Calculate mAP (Mean Average Precision for multi-class)
    # Note: mAP is more commonly used in object detection, but we compute it here
    # as the mean of per-class average precision scores (one-vs-rest approach)
    num_classes = y_pred_probs.shape[1]
    ap_scores = []
    for class_idx in range(num_classes):
        # Create binary labels for this class
        y_true_binary = (y_test_np == class_idx).astype(int)
        y_pred_probs_class = y_pred_probs[:, class_idx]
        if y_true_binary.sum() > 0:  # Only compute if class exists in test set
            ap = average_precision_score(y_true_binary, y_pred_probs_class)
            ap_scores.append(ap)
    mAP = np.mean(ap_scores) if ap_scores else 0.0
    
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        # Macro-averaged metrics (treat all classes equally)
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_score_macro': float(f1_macro),
        # Weighted-averaged metrics (weighted by class frequency - better for imbalanced classes)
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_score_weighted': float(f1_weighted),
        # Legacy fields (keeping for backward compatibility, using weighted as default)
        'precision': float(precision_weighted),
        'recall': float(recall_weighted),
        'f1_score': float(f1_weighted),
        'mAP': float(mAP),
        # Per-class metrics (essential for imbalanced classes)
        'precision_per_class': {class_names[i]: float(precision_per_class[i]) 
                               for i in range(len(class_names))},
        'recall_per_class': {class_names[i]: float(recall_per_class[i]) 
                            for i in range(len(class_names))},
        'f1_per_class': {class_names[i]: float(f1_per_class[i]) 
                        for i in range(len(class_names))},
        # Class support (number of samples per class)
        'class_support': class_support,
        # Confusion matrix (as list of lists for JSON serialization)
        'confusion_matrix': cm.tolist()
    }
    
    return results


def get_hyperparameters(experiment_name):
    """
    Retrieve and serialize hyperparameters from an experiment configuration.
    
    Args:
        experiment_name: Name of experiment from EXPERIMENTS dictionary
    
    Returns:
        Dictionary containing serialized hyperparameters (model_config, compile_config, 
        train_config, description), or None if experiment not found
    """
    if experiment_name not in EXPERIMENTS:
        return None
    
    experiment = EXPERIMENTS[experiment_name]
    # Create a copy to avoid modifying the original
    model_config = experiment.get('model_config', {}).copy()
    compile_config = experiment.get('compile_config', {}).copy()
    train_config = experiment.get('train_config', {}).copy()
    
    # Convert loss function to string for JSON serialization
    if 'loss' in compile_config:
        loss_obj = compile_config['loss']
        if hasattr(loss_obj, '__name__'):
            compile_config['loss'] = loss_obj.__name__
        elif hasattr(loss_obj, '__class__'):
            compile_config['loss'] = loss_obj.__class__.__name__
        else:
            compile_config['loss'] = str(loss_obj)
    
    # Convert optimizer to string if it's an object
    if 'optimizer' in compile_config:
        opt_obj = compile_config['optimizer']
        if isinstance(opt_obj, str):
            pass  # Already a string
        elif hasattr(opt_obj, '__name__'):
            compile_config['optimizer'] = opt_obj.__name__
        elif hasattr(opt_obj, '__class__'):
            compile_config['optimizer'] = opt_obj.__class__.__name__
    
    hyperparameters = {
        'model_config': model_config,
        'compile_config': compile_config,
        'train_config': train_config,
        'description': experiment.get('description', '')
    }
    
    return hyperparameters

def log_results(experiment_name, data_type, hyperparameters, results, log_file="results/experiment_results.json"):
    """
    Log experiment results to a JSON file.
    
    Args:
        experiment_name: Name of the experiment
        data_type: String describing the data type (e.g., "Validation", "Test")
        hyperparameters: Dictionary containing hyperparameters
        results: Dictionary containing evaluation results
        log_file: Path to JSON file where results will be logged
    
    Returns:
        None (prints warning if logging fails)
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'experiment_name': experiment_name,
        'data_type': data_type,
        'hyperparameters': hyperparameters,
        'results': results
    }
    
    # Load existing results if file exists
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_results = []
    else:
        all_results = []
    
    # Append new results
    all_results.append(log_entry)
    
    # Write back to file
    try:
        # Convert any remaining non-serializable objects (like tuples) to lists
        def make_json_serializable(obj):
            if isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_entry = make_json_serializable(log_entry)
        all_results[-1] = serializable_entry  # Replace the last entry with serializable version
        
        with open(log_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults logged to {log_file}")
    except (IOError, TypeError) as e:
        print(f"\nWarning: Could not write to log file {log_file}: {e}")

def plot_training_history(history, experiment_name=None, save_path=None):
    """
    Plot training and validation loss and accuracy over epochs.
    
    Args:
        history: Keras History object returned from model.fit()
        experiment_name: Optional name for the experiment (used in title)
        save_path: Optional path to save the plot. If None, displays the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    if 'accuracy' in history.history:
        ax2.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history.history:
            ax2.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # Add experiment name to title if provided
    if experiment_name:
        fig.suptitle(f'Training History: {experiment_name}', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
