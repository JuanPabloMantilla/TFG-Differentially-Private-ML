# src/utils.py

import tensorflow as tf

def create_keras_model(n_features, n_classes):
    """
    Defines and compiles the standard Keras model architecture for the experiments.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    # The model is returned uncompiled, so it can be compiled with different optimizers.
    return model