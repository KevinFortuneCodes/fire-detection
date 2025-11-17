from cnn import MODEL_CONFIG, COMPILE_CONFIG, TRAIN_CONFIG

# Experiment configurations for CNN classification
# Note: Images are preprocessed to 256x256 (see phase1_classification_preprocess.py)
# Dataset has 3 classes: fire (1), smoke (2), nothing (3)
# plot loss vs epoch and accuracy vs epoch: good way to visualize the training process
### shape of the curve is good information. looking for consistency

EXPERIMENTS = {
    'baseline': {
        'model_config': MODEL_CONFIG.copy(),
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 16,  # Reduced for GPU memory constraints
            'validation_batch_size': 4  # Very small for validation to prevent OOM
        },
        'description': 'Baseline model with default settings (batch_size=16 for GPU memory)'
    },
    
    'deeper_network': {
        'model_config': {
            'input_shape': (256, 256, 3),  # Matches phase1 preprocessing default image size
            'conv_layers': [
                {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'},  # Increased from 128 to 256
            ],
            'pool_size': (2, 2),
            'dense_units': [256, 128],  # Increased to match final conv layer capacity
            'output_units': 3,  # 3 classes: fire, smoke, nothing
            'output_activation': None,
            'dropout_rate': 0.2,
        },
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 16,  # Reduced from 32 to 16 to fit in GPU memory
            'validation_batch_size': 4  # Very small for validation to prevent OOM
        },
        'description': 'Deeper network with more filters and dense layers (batch_size=16 for GPU memory)'
    },
    
    'more_dropout': {
        'model_config': {
            **MODEL_CONFIG.copy(),
            'dropout_rate': 0.5
        },
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 16,
            'validation_batch_size': 8
        },
        'description': 'Baseline model with dropout regularization (batch_size=16 for GPU memory)'
    },
    
    'lower_learning_rate': {
        'model_config': MODEL_CONFIG.copy(),
        'compile_config': {
            **COMPILE_CONFIG.copy(),
            'learning_rate': 0.0001
        },
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 16,
            'validation_batch_size': 8
        },
        'description': 'Baseline model with lower learning rate (batch_size=16 for GPU memory)'
    },
    'complex_network': {
        'model_config': {
            'input_shape': (256, 256, 3),  # Matches phase1 preprocessing default image size
            'conv_layers': [
                {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'},  # Increased from 128 to 256
            ],
            'pool_size': (2, 2),
            'dense_units': [256, 128],  # Increased to match final conv layer capacity
            'output_units': 3,  # 3 classes: fire, smoke, nothing
            'output_activation': None,
            'dropout_rate': 0.5,
        },
        'compile_config': {
            **COMPILE_CONFIG.copy(),
            'learning_rate': 0.0001
        },
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 16,  # Reduced from 32 to 16 to fit in GPU memory
            'validation_batch_size': 4  # Very small for validation to prevent OOM
        },
        'description': 'Complex network with all parameter adjustments (batch_size=16 for GPU memory)'
    },
}
