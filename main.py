from cnn import create_model, compile_model, train_model, evaluate_model
from experiments import EXPERIMENTS
from load_data import retrieve_data
from evaluate import get_hyperparameters, log_results, evaluate_model

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
        Dictionary with test_loss, test_accuracy, precision, recall, and mAP
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Model on {data_type} Data...")
    print(f"{'='*60}")
    
    eval_results = evaluate_model(model, eval_data)
    
    print(f"{data_type} Loss: {eval_results['test_loss']:.4f}")
    print(f"{data_type} Accuracy: {eval_results['test_accuracy']:.4f}")
    print(f"Precision: {eval_results['precision']:.4f}")
    print(f"Recall: {eval_results['recall']:.4f}")
    print(f"mAP (Mean Average Precision): {eval_results['mAP']:.4f}")
    
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