from cnn import create_model, compile_model, train_model
from experiments import EXPERIMENTS
from load_data import retrieve_data
from evaluate import get_hyperparameters, log_results, evaluate_model, plot_training_history
from pathlib import Path

def setup_experiment(experiment_name, show_summary=True):
    """
    Create and compile a model for an experiment.
    
    Args:
        experiment_name: Name of experiment from EXPERIMENTS dictionary
        show_summary: Whether to print model summary
    
    Returns:
        Compiled Keras model ready for training
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}. "
                        f"Available: {list(EXPERIMENTS.keys())}")
    
    experiment = EXPERIMENTS[experiment_name]
    
    print(f"\n{'='*60}")
    print(f"Setting up Experiment: {experiment_name}")
    print(f"Description: {experiment['description']}")
    print(f"{'='*60}")
    
    # Create and compile model
    model = create_model(experiment['model_config'])
    model = compile_model(model, experiment['compile_config'])
    
    if show_summary:
        model.summary()
    
    return model

def train_experiment(model, experiment_name, train_data, val_data=None):
    """
    Train a model for an experiment.
    
    Args:
        model: Compiled Keras model
        experiment_name: Name of experiment from EXPERIMENTS dictionary
        train_data: Training data (X_train, y_train) tuple
        val_data: Validation data (X_val, y_val) tuple or None
    
    Returns:
        Trained model object (History from model.fit())
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    experiment = EXPERIMENTS[experiment_name]
    
    print(f"\n{'='*60}")
    print(f"Training Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    trained_model = train_model(
        model,
        train_data,
        val_data=val_data,
        config=experiment['train_config']
    )
    
    # Plot training history if enabled (default: True)
    train_config = experiment.get('train_config', {})
    if train_config.get('plot_history', True):
        save_path = train_config.get('plot_save_path', None)
        # If save_path is a directory or None, generate unique filename with experiment name
        if save_path is None:
            # Default to 'plots' directory
            save_path = Path('plots')
            save_path.mkdir(parents=True, exist_ok=True)
            # Generate unique filename with experiment name and timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = save_path / f'{experiment_name}_training_history_{timestamp}.png'
        else:
            save_path = Path(save_path)
            # Check if it's an existing directory
            if save_path.exists() and save_path.is_dir():
                # It's a directory - generate unique filename
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = save_path / f'{experiment_name}_training_history_{timestamp}.png'
            else:
                # User provided a specific file path - use it as-is
                save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_training_history(trained_model, experiment_name=experiment_name, save_path=str(save_path))
    
    return trained_model

def evaluate_experiment(model, eval_data, data_type="Validation", 
                        experiment_name=None, hyperparameters=None, 
                        log_file="experiment_results.json"):
    """
    Evaluate a trained model on evaluation data.
    
    Args:
        model: Trained Keras model
        eval_data: Evaluation data (X_eval, y_eval) tuple
        data_type: String describing the data type (e.g., "Validation", "Test")
        experiment_name: Name of the experiment being evaluated
        hyperparameters: Dictionary containing hyperparameters (model_config, compile_config, train_config)
        log_file: Path to JSON file where results will be logged
    
    Returns:
        Dictionary with test_loss, test_accuracy, precision, recall, f1_score, and mAP
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Model on {data_type} Data...")
    print(f"{'='*60}")
    
    eval_results = evaluate_model(model, eval_data)
    
    print(f"{data_type} Loss: {eval_results['test_loss']:.4f}")
    print(f"{data_type} Accuracy: {eval_results['test_accuracy']:.4f}")
    
    # Print class distribution for context
    print(f"\nClass Distribution (support):")
    for class_name in ['fire', 'smoke', 'nothing']:
        support = eval_results['class_support'][class_name]
        total = sum(eval_results['class_support'].values())
        pct = (support / total) * 100 if total > 0 else 0
        print(f"  {class_name.capitalize()}: {support} ({pct:.1f}%)")
    
    # Print weighted metrics (better for imbalanced classes)
    print(f"\nOverall Metrics (weighted by class frequency - recommended for imbalanced classes):")
    print(f"  Precision: {eval_results['precision_weighted']:.4f}")
    print(f"  Recall: {eval_results['recall_weighted']:.4f}")
    print(f"  F1-Score: {eval_results['f1_score_weighted']:.4f}")
    
    # Print macro metrics for comparison
    print(f"\nOverall Metrics (macro-averaged - treats all classes equally):")
    print(f"  Precision: {eval_results['precision_macro']:.4f}")
    print(f"  Recall: {eval_results['recall_macro']:.4f}")
    print(f"  F1-Score: {eval_results['f1_score_macro']:.4f}")
    
    print(f"\n  mAP (Mean Average Precision): {eval_results['mAP']:.4f}")
    
    # Print per-class metrics (essential for imbalanced classes)
    print(f"\nPer-Class Metrics (critical for understanding imbalanced class performance):")
    for class_name in ['fire', 'smoke', 'nothing']:
        support = eval_results['class_support'][class_name]
        print(f"  {class_name.capitalize()} (n={support}):")
        print(f"    Precision: {eval_results['precision_per_class'][class_name]:.4f}")
        print(f"    Recall: {eval_results['recall_per_class'][class_name]:.4f}")
        print(f"    F1-Score: {eval_results['f1_per_class'][class_name]:.4f}")
    
    # Print confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              fire smoke nothing")
    cm = eval_results['confusion_matrix']
    class_names = ['fire', 'smoke', 'nothing']
    for i, class_name in enumerate(class_names):
        print(f"{class_name:8} {cm[i]}")
    
    # Log results to file if experiment_name is provided
    if experiment_name is not None:
        log_results(experiment_name, data_type, hyperparameters, eval_results, log_file)
    
    return eval_results

def run_experiment(experiment_name, train_data=None, val_data=None, 
                   test_data=None, show_summary=True):
    """
    Convenience function: Setup, train, and evaluate an experiment in one call.
    
    Args:
        experiment_name: Name of experiment from EXPERIMENTS dictionary
        train_data: Training data (X_train, y_train) tuple or None
        val_data: Validation data (X_val, y_val) tuple or None
        test_data: Test data (X_test, y_test) tuple or None
        show_summary: Whether to print model summary
    
    Returns:
        Tuple of (model, trained_model, eval_results)
        Returns None for trained_model/eval_results if training/evaluation not performed
        Evaluates on test_data if provided, otherwise on val_data if provided
    """
    # Setup model
    model = setup_experiment(experiment_name, show_summary=show_summary)
    
    # Get hyperparameters from experiment config
    hyperparameters = get_hyperparameters(experiment_name)
    
    # Train if data provided
    trained_model = None
    if train_data is not None:
        trained_model = train_experiment(model, experiment_name, train_data, val_data)
    
    # Evaluate on test data if provided, otherwise on validation data
    eval_results = None
    if test_data is not None:
        eval_results = evaluate_experiment(
            model, test_data, 
            data_type="Test",
            experiment_name=experiment_name,
            hyperparameters=hyperparameters
        )
    elif val_data is not None:
        eval_results = evaluate_experiment(
            model, val_data, 
            data_type="Validation",
            experiment_name=experiment_name,
            hyperparameters=hyperparameters
        )
    
    return model, trained_model, eval_results

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors between experiments."""
    import tensorflow as tf
    import gc
    
    # Clear TensorFlow session/Keras backend
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Try to clear GPU memory (if available)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu)
    except:
        pass

def run_all_experiments(train_data=None, val_data=None, test_data=None):
    """
    Run all defined experiments.
    
    Args:
        train_data: Training data (X_train, y_train) tuple or None
        val_data: Validation data (X_val, y_val) tuple or None
        test_data: Test data (X_test, y_test) tuple or None
    
    Returns:
        Dictionary mapping experiment names to results dict with model, trained_model, eval_results
        Evaluates on test_data if provided, otherwise on val_data if provided
    """
    results = {}
    for exp_name in EXPERIMENTS.keys():
        try:
            # Clear GPU memory before each experiment
            clear_gpu_memory()
            
            print(f"\n{'='*60}")
            print(f"Starting experiment: {exp_name}")
            print(f"{'='*60}")
            
            model, trained_model, eval_results = run_experiment(
                exp_name, 
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                show_summary=True
            )
            results[exp_name] = {
                'model': model,
                'trained_model': trained_model,
                'eval_results': eval_results
            }
            print(f"\n✓ Successfully completed experiment: {exp_name}")
            
        except Exception as e:
            error_msg = str(e)
            if "OOM" in error_msg or "ResourceExhausted" in error_msg:
                print(f"\n⚠ Experiment '{exp_name}' failed due to GPU out-of-memory (OOM)")
                print(f"   This experiment requires more GPU memory than available.")
                print(f"   Consider reducing batch size or model size for this experiment.")
                results[exp_name] = {
                    'error': 'OOM',
                    'error_message': error_msg
                }
            else:
                print(f"\n✗ Experiment '{exp_name}' failed with error: {error_msg}")
                results[exp_name] = {
                    'error': 'Training failed',
                    'error_message': error_msg
                }
            
            # Clear memory even after failure
            clear_gpu_memory()
    
    return results

def check_gpu_and_confirm():
    """
    Check GPU availability and prompt user to continue.
    
    Returns:
        bool: True if user confirms to continue, False otherwise
    """
    import tensorflow as tf
    
    # Check GPU availability
    print("\n" + "="*60)
    print("GPU/Device Check")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"\nGPUs available: {len(gpus)}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Get GPU details if possible
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details:
                    print(f"    Details: {gpu_details}")
            except:
                pass
    else:
        print("  No GPUs detected. Training will use CPU.")
    
    print(f"\nCPUs available: {len(cpus)}")
    for i, cpu in enumerate(cpus):
        print(f"  CPU {i}: {cpu.name}")
    
    # Show which device will be used
    if gpus:
        print(f"\n✓ GPU will be used for training")
    else:
        print(f"\n⚠ CPU will be used for training (no GPU detected)")
    
    print("="*60)
    
    # Prompt user to continue
    response = input("\nContinue with experiment? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Experiment cancelled.")
        return False
    
    return True

# ============================================================================

if __name__ == "__main__":
    import sys
    import os
    import random
    import numpy as np
    import tensorflow as tf
    
    # Set random seeds for reproducibility
    SEED = 42
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Configure GPU memory growth to prevent OOM errors
    # This allows TensorFlow to allocate GPU memory incrementally instead of all at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"Warning: Could not set GPU memory growth: {e}")
    
    # Check GPU and get user confirmation
    if not check_gpu_and_confirm():
        sys.exit(0)
    
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    
    #loading the preprocessed data from pre-defined splits
    train_data, val_data, test_data = retrieve_data(
        train_path="processed_dfire/train",
        val_path="processed_dfire/validation",
        test_path=None
    )

    # Run specific experiment or all experiments
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        
        model, trained_model, eval_results = run_experiment(
            experiment_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        
    else:
        # Run all experiments by default
        print("Running all experiments...")
        results = run_all_experiments(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        print(f"\n✓ Completed {len(results)} experiments")