import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score

MODEL_CONFIG = {
    'input_shape': (256, 256, 3),  # Matches phase1 preprocessing default image size
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},  # Increased from 64 to 128
    ],
    'pool_size': (2, 2),  # Standard 2x2 max pooling (halves spatial dimensions)
    'dense_units': [128],  # Increased from 64 to 128 to match final conv layer
    'output_units': 3,  # 3 classes: fire, smoke, nothing
    'output_activation': None,
    'dropout_rate': 0.2,  # Small default dropout for regularization (131k features → 128 units)
}

COMPILE_CONFIG = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': SparseCategoricalCrossentropy(from_logits=True), # make sure to add new loss classes to the import
    'metrics': ['accuracy'] # must be a list
}

TRAIN_CONFIG = {
    'epochs': 10,
    'batch_size': 16,  # Reduced from 32 to prevent GPU OOM errors
    'validation_batch_size': 4,  # Very small batch size for validation to reduce GPU memory during evaluation
    'verbose': 1,
    'shuffle': True  # Shuffle training data each epoch
}



def create_model(config=None):
    """
    Create a CNN model. See MODEL_CONFIG for parameterization.
    
    Args:
        config: Dictionary with model configuration. Uses MODEL_CONFIG if None.
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = MODEL_CONFIG.copy()
    
    model = models.Sequential()
    
    # Input layer
    first_conv = config['conv_layers'][0]
    model.add(layers.Conv2D(
        first_conv['filters'],
        first_conv['kernel_size'],
        activation=first_conv['activation'],
        input_shape=config['input_shape']
    ))
    model.add(layers.MaxPooling2D(config['pool_size']))
    
    # Additional convolutional layers. Number of layers is determined by the length of the list in the config dictionary.
    for conv_config in config['conv_layers'][1:]:
        model.add(layers.Conv2D(
            conv_config['filters'],
            conv_config['kernel_size'],
            activation=conv_config['activation']
        ))
        model.add(layers.MaxPooling2D(config['pool_size']))
    
    model.add(layers.Flatten())
    
    # Create dense layers. The number of layers is determined by the length of the list in the config dictionary
    for units in config['dense_units']:
        model.add(layers.Dense(units, activation='relu'))
        if config['dropout_rate'] > 0:
            model.add(layers.Dropout(config['dropout_rate']))
    
    # Output layer
    model.add(layers.Dense(
        config['output_units'],
        activation=config['output_activation']
    ))
    
    return model

def compile_model(model, config=COMPILE_CONFIG):
    """
    Compile the model with specified optimizer and learning rate. See COMPILE_CONFIG to change parameters
    """
    if config['optimizer'] == 'adam':
        opt = Adam(learning_rate=config['learning_rate'])
    else:
        opt = config['optimizer']
    
    model.compile(
        optimizer=opt,
        loss=config['loss'],
        metrics=config['metrics']
    )
    return model

def train_model(model, train_data, val_data=None, config=TRAIN_CONFIG):
    """
    Train the model. See TRAIN_CONFIG for parameters
    
    Args:
        model: Compiled Keras model
        train_data: Training data (X_train, y_train) tuple
        val_data: Validation data (X_val, y_val) tuple or None
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Verbosity level
    
    Returns:
        Trained model
    """
    X_train, y_train = train_data
    validation_data = val_data if val_data else None
    
    # Use smaller batch size for validation if specified to reduce GPU memory
    validation_batch_size = config.get('validation_batch_size', config['batch_size'])
    
    # Optionally skip validation during training if memory is too constrained
    # Validation can be done separately after training
    if config.get('skip_validation_during_training', False):
        validation_data = None
        print("Warning: Skipping validation during training to save GPU memory.")
        print("         You can evaluate the model separately after training.")
    
    fit_kwargs = {
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'verbose': config['verbose'],
        'shuffle': config.get('shuffle', True)
    }
    
    if validation_data is not None:
        fit_kwargs['validation_data'] = validation_data
        fit_kwargs['validation_batch_size'] = validation_batch_size
    
    trained_model = model.fit(
        X_train, y_train,
        **fit_kwargs
    )
    
    return trained_model
