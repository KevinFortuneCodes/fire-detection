from cnn import MODEL_CONFIG, COMPILE_CONFIG, TRAIN_CONFIG

# filters should get smaller
# 128 is a fine starting point
# plot loss vs epoch and accuracy vs epoch: good way to visualize the training process
### shape of the curve is good information. looking for consistency

#Kanagha will send random sampling splits


# early stopping

EXPERIMENTS = {
    'baseline': {
        'model_config': MODEL_CONFIG.copy(),
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': TRAIN_CONFIG.copy(),
        'description': 'Baseline model with default settings'
    },
    
    'deeper_network': {
        'model_config': {
            'input_shape': (32, 32, 3),
            'conv_layers': [
                {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
                {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
            ],
            'pool_size': (2, 2),
            'dense_units': [128, 64],
            'output_units': 2,
            'output_activation': None,
            'dropout_rate': 0.0,
        },
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': TRAIN_CONFIG.copy(),
        'description': 'Deeper network with more filters and dense layers'
    },
    
    'with_dropout': {
        'model_config': {
            **MODEL_CONFIG.copy(),
            'dropout_rate': 0.5
        },
        'compile_config': COMPILE_CONFIG.copy(),
        'train_config': TRAIN_CONFIG.copy(),
        'description': 'Baseline model with dropout regularization'
    },
    
    'lower_learning_rate': {
        'model_config': MODEL_CONFIG.copy(),
        'compile_config': {
            **COMPILE_CONFIG.copy(),
            'learning_rate': 0.0001
        },
        'train_config': TRAIN_CONFIG.copy(),
        'description': 'Baseline model with lower learning rate'
    },
}
