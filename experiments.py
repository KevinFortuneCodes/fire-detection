from cnn import MODEL_CONFIG, COMPILE_CONFIG, TRAIN_CONFIG

EXPERIMENTS = {
    'baseline': {
        'model_config': MODEL_CONFIG.copy(),
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 32, 
            'validation_batch_size': 16 
        },
        'description': 'Baseline model with default settings (batch_size=32)'
    },
    
    'deeper_network': {
        'model_config': {
            'input_shape': (256, 256, 3),
            'conv_layers': [
                {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'},
            ],
            'pool_size': (2, 2),
            'dense_units': [256, 128],
            'output_units': 4, 
            'output_activation': None,
            'dropout_rate': 0.2,
        },
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 32, 
            'validation_batch_size': 16  
        },
        'description': 'Deeper network with more filters and dense layers (batch_size=32)'
    },
    
    'more_dropout': {
        'model_config': {
            **MODEL_CONFIG.copy(),
            'dropout_rate': 0.25
        },
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 32,
            'validation_batch_size': 16,
            'epochs': 10
        },
        'description': 'Baseline model with dropout regularization (batch_size=32)'
    },
    'less_dropout': {
        'model_config': {
            **MODEL_CONFIG.copy(),
            'dropout_rate': 0.15
        },
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 32,
            'validation_batch_size': 16,
            'epochs': 10
        },
        'description': 'Baseline model with dropout regularization (batch_size=32)'
    },
    
    'lower_learning_rate': {
        'model_config': MODEL_CONFIG.copy(),
        'compile_config': {
            **COMPILE_CONFIG.copy(),
            'learning_rate': 0.0001
        },
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 32,
            'validation_batch_size': 16
        },
        'description': 'Baseline model with lower learning rate (batch_size=32)'
    },
    'add_regularization': {
        'model_config': { **MODEL_CONFIG.copy(), 'l2_regularization': 0.001},
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': {
            **TRAIN_CONFIG.copy(),
            'batch_size': 32, 
            'validation_batch_size': 16 
        },
        'description': 'Baseline model with L2 regularization'
    },
}
