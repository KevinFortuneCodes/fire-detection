import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l2 
import numpy as np
from sklearn.metrics import precision_score, recall_score, average_precision_score

MODEL_CONFIG = {
    'input_shape': (256, 256, 3), 
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'}, 
    ],
    'pool_size': (2, 2),  # Standard 2x2 max pooling (halves spatial dimensions)
    'dense_units': [128], 
    'output_units': 4,  # 4 classes: fire, smoke, nothing, fire_and_smoke
    'output_activation': None,
    'dropout_rate': 0.2, 
    'l2_regularization': 0.0, 
}

COMPILE_CONFIG = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': SparseCategoricalCrossentropy(from_logits=True),
    'metrics': ['accuracy']
}

TRAIN_CONFIG = {
    'epochs': 10,
    'batch_size': 32, 
    'validation_batch_size': 16, 
    'verbose': 1,
    'shuffle': True 
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
    
    # Get L2 regularization value, default to 0 if not specified
    l2_reg = config.get('l2_regularization', 0)
    kernel_regularizer = l2(l2_reg) if l2_reg > 0 else None
    
    model = models.Sequential()
    
    # Input layer
    first_conv = config['conv_layers'][0]
    conv_kwargs = {
        'filters': first_conv['filters'],
        'kernel_size': first_conv['kernel_size'],
        'activation': first_conv['activation'],
        'input_shape': config['input_shape']
    }
    if kernel_regularizer is not None:
        conv_kwargs['kernel_regularizer'] = kernel_regularizer
    model.add(layers.Conv2D(**conv_kwargs))
    model.add(layers.MaxPooling2D(config['pool_size']))
    
    # Additional convolutional layers
    for conv_config in config['conv_layers'][1:]:
        conv_kwargs = {
            'filters': conv_config['filters'],
            'kernel_size': conv_config['kernel_size'],
            'activation': conv_config['activation']
        }
        if kernel_regularizer is not None:
            conv_kwargs['kernel_regularizer'] = kernel_regularizer
        model.add(layers.Conv2D(**conv_kwargs))
        model.add(layers.MaxPooling2D(config['pool_size']))
    
    model.add(layers.Flatten())
    
    # Create dense layers
    for units in config['dense_units']:
        dense_kwargs = {
            'units': units,
            'activation': 'relu'
        }
        if kernel_regularizer is not None:
            dense_kwargs['kernel_regularizer'] = kernel_regularizer
        model.add(layers.Dense(**dense_kwargs))
        if config['dropout_rate'] > 0:
            model.add(layers.Dropout(config['dropout_rate']))
    
    # Output layer
    output_kwargs = {
        'units': config['output_units'],
        'activation': config['output_activation']
    }
    if kernel_regularizer is not None:
        output_kwargs['kernel_regularizer'] = kernel_regularizer
    model.add(layers.Dense(**output_kwargs))
    
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
        train_data: Training data - can be either:
            - (X_train, y_train) tuple of numpy arrays
            - tf.data.Dataset (memory-efficient, loads images on-demand)
        val_data: Validation data - can be either:
            - (X_val, y_val) tuple of numpy arrays
            - tf.data.Dataset (memory-efficient, loads images on-demand)
            - None
        config: Training configuration dictionary
    
    Returns:
        Trained model (History object from model.fit())
    """
    # Check if train_data is a tf.data.Dataset or numpy arrays
    if isinstance(train_data, tf.data.Dataset):
        # Dataset already has batching, so we don't specify batch_size
        X_train = train_data
        y_train = None
        use_dataset = True
    else:
        # Numpy arrays - unpack tuple
        X_train, y_train = train_data
        use_dataset = False
    
    # Handle validation data
    validation_data = None
    if val_data is not None:
        if isinstance(val_data, tf.data.Dataset):
            validation_data = val_data
        else:
            validation_data = val_data
    
    # Optionally skip validation during training if memory is too constrained
    # Validation can be done separately after training
    if config.get('skip_validation_during_training', False):
        validation_data = None
        print("Warning: Skipping validation during training to save GPU memory.")
        print("         You can evaluate the model separately after training.")
    
    # Build fit_kwargs
    fit_kwargs = {
        'epochs': config['epochs'],
        'verbose': config['verbose'],
    }
    
    if use_dataset:
        # For tf.data.Dataset, don't specify batch_size or shuffle
        # (those are handled by the dataset)
        if validation_data is not None:
            fit_kwargs['validation_data'] = validation_data
    else:
        # For numpy arrays, specify batch_size and shuffle
        fit_kwargs['batch_size'] = config['batch_size']
        fit_kwargs['shuffle'] = config.get('shuffle', True)
        
        if validation_data is not None:
            fit_kwargs['validation_data'] = validation_data
            # Use smaller batch size for validation if specified to reduce GPU memory
            validation_batch_size = config.get('validation_batch_size', config['batch_size'])
            fit_kwargs['validation_batch_size'] = validation_batch_size
    
    # Train the model
    if use_dataset:
        # For tf.data.Dataset, Keras will handle iteration automatically
        trained_model = model.fit(X_train, **fit_kwargs)
    else:
        trained_model = model.fit(X_train, y_train, **fit_kwargs)
    
    # Check training progress
    if hasattr(trained_model, 'history'):
        history = trained_model.history
        if 'loss' in history:
            losses = history['loss']
            if len(losses) > 1:
                loss_change = losses[0] - losses[-1]
                if abs(loss_change) < 0.001:
                    print(f"\nWarning: Training loss barely changed ({losses[0]:.4f} → {losses[-1]:.4f})")
                else:
                    print(f"\nTraining loss: {losses[0]:.4f} → {losses[-1]:.4f} (Δ={loss_change:.4f})")
    
    return trained_model
