
from experiments import EXPERIMENTS
import json
from datetime import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score

def evaluate_model(model, test_data):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        test_data: Test data (X_test, y_test) tuple
    
    Returns:
        Dictionary containing:
        - test_loss: Loss on test set
        - test_accuracy: Accuracy on test set
        - precision: Precision score
        - recall: Recall score
        - mAP: Mean Average Precision (Average Precision for binary classification)
    """
    X_test, y_test = test_data
    
    # Get loss and accuracy from model.evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions (logits)
    y_pred_logits = model.predict(X_test, verbose=0)
    
    # Convert logits to probabilities using softmax
    y_pred_probs = tf.nn.softmax(y_pred_logits).numpy()
    
    # Get predicted classes (argmax)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Convert tensors to numpy if needed
    if isinstance(y_test, tf.Tensor):
        y_test_np = y_test.numpy()
    else:
        y_test_np = y_test
    
    # Calculate precision and recall
    precision = precision_score(y_test_np, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test_np, y_pred, average='binary', zero_division=0)
    
    # Calculate mAP (Average Precision for binary classification)
    # Use probabilities for the positive class (class 1 = fire or smoke)
    y_pred_probs_positive = y_pred_probs[:, 1]  # Probabilities for fire class
    mAP = average_precision_score(y_test_np, y_pred_probs_positive)
    
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'precision': float(precision),
        'recall': float(recall),
        'mAP': float(mAP)
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

def log_results(experiment_name, data_type, hyperparameters, results, log_file="experiment_results.json"):
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
