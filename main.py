from cnn import create_model, compile_model, train_model, evaluate_model
from experiments import EXPERIMENTS

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

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
        Training history object
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    experiment = EXPERIMENTS[experiment_name]
    
    print(f"\n{'='*60}")
    print(f"Training Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    history = train_model(
        model,
        train_data,
        val_data=val_data,
        config=experiment['train_config']
    )
    
    return history

def evaluate_experiment(model, test_data):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained Keras model
        test_data: Test data (X_test, y_test) tuple
    
    Returns:
        Dictionary with test_loss and test_accuracy
    """
    print(f"\n{'='*60}")
    print("Evaluating Model...")
    print(f"{'='*60}")
    
    test_loss, test_acc = evaluate_model(model, test_data)
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return test_results

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
        Tuple of (model, training_history, test_results)
        Returns None for history/test_results if training/evaluation not performed
    """
    # Setup model
    model = setup_experiment(experiment_name, show_summary=show_summary)
    
    # Train if data provided
    history = None
    if train_data is not None:
        history = train_experiment(model, experiment_name, train_data, val_data)
    
    # Evaluate if test data provided
    test_results = None
    if test_data is not None:
        test_results = evaluate_experiment(model, test_data)
    
    return model, history, test_results

def run_all_experiments(train_data=None, val_data=None, test_data=None):
    """
    Run all defined experiments.
    
    Args:
        train_data: Training data (X_train, y_train) tuple or None
        val_data: Validation data (X_val, y_val) tuple or None
        test_data: Test data (X_test, y_test) tuple or None
    
    Returns:
        Dictionary mapping experiment names to results dict with model, history, test_results
    """
    results = {}
    for exp_name in EXPERIMENTS.keys():
        model, history, test_results = run_experiment(
            exp_name, 
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            show_summary=True
        )
        results[exp_name] = {
            'model': model,
            'history': history,
            'test_results': test_results
        }
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # TODO: Load your data here
    # X_train, y_train = ...
    # X_val, y_val = ...
    # X_test, y_test = ...
    train_data = None  # (X_train, y_train)
    val_data = None    # (X_val, y_val)
    test_data = None   # (X_test, y_test)
    
    # Run specific experiment or all experiments
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        
        # Example: Use the convenience function (does everything)
        model, history, test_results = run_experiment(
            experiment_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        
        # Alternative: Use separate functions for more control
        # model = setup_experiment(experiment_name)
        # history = train_experiment(model, experiment_name, train_data, val_data)
        # test_results = evaluate_experiment(model, test_data)
        
    else:
        # Run all experiments by default
        print("Running all experiments...")
        results = run_all_experiments(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        print(f"\n✓ Completed {len(results)} experiments")