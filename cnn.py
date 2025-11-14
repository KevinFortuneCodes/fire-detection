import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import numpy as np

MODEL_CONFIG = {
    'input_shape': (32, 32, 3),
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
    ],
    'pool_size': (2, 2),
    'dense_units': [64],
    'output_units': 2, # number of classifications
    'output_activation': None,
    'dropout_rate': 0.0,
}

COMPILE_CONFIG = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': SparseCategoricalCrossentropy(from_logits=True), # make sure to add new loss classes to the import
    'metrics': ['accuracy'] # must be a list
}

TRAIN_CONFIG = {
    'epochs': 10,
    'batch_size': 32,
    'verbose': 1
}

# need to convert torch tensor
# download the first link in the downloads page
# 

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
    
    trained_model = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=config['verbose']
    )
    
    return trained_model

def evaluate_model(model, test_data):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        test_data: Test data (X_test, y_test) tuple
    
    Returns:
        Test loss and accuracy
    """
    X_test, y_test = test_data
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_loss, test_acc
