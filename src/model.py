"""
Autoencoder Model Architecture for Anomaly Detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class Autoencoder(Model):
    """
    Autoencoder model for anomaly detection in CERN particle physics data.
    
    The model learns to reconstruct Standard Model (SM) data and uses
    reconstruction error to identify Beyond Standard Model (BSM) anomalies.
    """
    
    def __init__(self, input_dim, encoding_dims, latent_dim, 
                 activation='relu', dropout_rate=0.2, output_activation='linear'):
        """
        Initialize the Autoencoder model.
        
        Args:
            input_dim (int): Dimension of input features
            encoding_dims (list): List of dimensions for encoder hidden layers
            latent_dim (int): Dimension of latent space (bottleneck)
            activation (str): Activation function for hidden layers
            dropout_rate (float): Dropout rate for regularization
            output_activation (str): Activation function for output layer
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = latent_dim
        
        # Build Encoder
        self.encoder = self._build_encoder(input_dim, encoding_dims, latent_dim,
                                           activation, dropout_rate)
        
        # Build Decoder
        self.decoder = self._build_decoder(latent_dim, encoding_dims, input_dim,
                                           activation, dropout_rate, output_activation)
    
    def _build_encoder(self, input_dim, encoding_dims, latent_dim, activation, dropout_rate):
        """Build the encoder network"""
        encoder_layers = [layers.InputLayer(input_shape=(input_dim,))]
        
        # Add encoding layers
        for dim in encoding_dims:
            encoder_layers.append(layers.Dense(dim, activation=activation))
            encoder_layers.append(layers.BatchNormalization())
            encoder_layers.append(layers.Dropout(dropout_rate))
        
        # Latent space
        encoder_layers.append(layers.Dense(latent_dim, activation=activation, name='latent'))
        
        return keras.Sequential(encoder_layers, name='encoder')
    
    def _build_decoder(self, latent_dim, encoding_dims, input_dim, 
                       activation, dropout_rate, output_activation):
        """Build the decoder network"""
        decoder_layers = [layers.InputLayer(input_shape=(latent_dim,))]
        
        # Add decoding layers (reverse order of encoder)
        for dim in reversed(encoding_dims):
            decoder_layers.append(layers.Dense(dim, activation=activation))
            decoder_layers.append(layers.BatchNormalization())
            decoder_layers.append(layers.Dropout(dropout_rate))
        
        # Reconstruction layer
        decoder_layers.append(layers.Dense(input_dim, activation=output_activation, 
                                          name='reconstruction'))
        
        return keras.Sequential(decoder_layers, name='decoder')
    
    def call(self, inputs, training=False):
        """Forward pass through the autoencoder"""
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def encode(self, inputs):
        """Encode inputs to latent space"""
        return self.encoder(inputs, training=False)
    
    def decode(self, encoded):
        """Decode from latent space to reconstruction"""
        return self.decoder(encoded, training=False)
    
    def get_config(self):
        """Get model configuration for serialization"""
        return {
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'latent_dim': self.latent_dim
        }


def create_autoencoder(config):
    """
    Create an autoencoder model from configuration.
    
    Args:
        config (dict): Model configuration dictionary
        
    Returns:
        Autoencoder: Compiled autoencoder model
    """
    model_config = config['model']
    
    autoencoder = Autoencoder(
        input_dim=model_config['input_dim'],
        encoding_dims=model_config['encoding_dims'],
        latent_dim=model_config['latent_dim'],
        activation=model_config['activation'],
        dropout_rate=model_config['dropout_rate'],
        output_activation=model_config['output_activation']
    )
    
    # Compile model
    learning_rate = config['training']['learning_rate']
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder


def reconstruction_error(model, data):
    """
    Calculate reconstruction error for anomaly detection.
    
    Args:
        model: Trained autoencoder model
        data: Input data
        
    Returns:
        np.ndarray: Reconstruction errors for each sample
    """
    reconstructions = model.predict(data)
    errors = np.mean(np.square(data - reconstructions), axis=1)
    return errors
