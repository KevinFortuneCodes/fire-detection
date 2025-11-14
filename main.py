from cnn import create_model, compile_model, train_model, evaluate_model
from experiments import EXPERIMENTS
from load_data import retrieve_data


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
        Trained model object
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
    
    return trained_model

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
        Tuple of (model, trained_model, test_results)
        Returns None for trained_model/test_results if training/evaluation not performed
    """
    # Setup model
    model = setup_experiment(experiment_name, show_summary=show_summary)
    
    # Train if data provided
    trained_model = None
    if train_data is not None:
        trained_model = train_experiment(model, experiment_name, train_data, val_data)
    
    # Evaluate if test data provided
    test_results = None
    if test_data is not None:
        test_results = evaluate_experiment(model, test_data)
    
    return model, trained_model, test_results

def run_all_experiments(train_data=None, val_data=None, test_data=None):
    """
    Run all defined experiments.
    
    Args:
        train_data: Training data (X_train, y_train) tuple or None
        val_data: Validation data (X_val, y_val) tuple or None
        test_data: Test data (X_test, y_test) tuple or None
    
    Returns:
        Dictionary mapping experiment names to results dict with model, trained_model, test_results
    """
    results = {}
    for exp_name in EXPERIMENTS.keys():
        model, trained_model, test_results = run_experiment(
            exp_name, 
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            show_summary=True
        )
        results[exp_name] = {
            'model': model,
            'trained_model': trained_model,
            'test_results': test_results
        }
    return results

# ============================================================================

if __name__ == "__main__":
    import sys
    
    #loading the preprocessed data and making validation split
    train_data, val_data, test_data = retrieve_data(
        train_path="processed_dfire/train",  # TODO: update path after running process_dfire
        test_path="processed_dfire/test",   # TODO: update path after running process_dfire
        val_size=0.2
    )

    # Run specific experiment or all experiments
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        
        model, trained_model, test_results = run_experiment(
            experiment_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        
   # TODO: Add code to log experiment results

    else:
        # Run all experiments by default
        print("Running all experiments...")
        results = run_all_experiments(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )
        print(f"\n✓ Completed {len(results)} experiments")