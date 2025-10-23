"""
Training script for Autoencoder anomaly detection model
"""

import os
import yaml
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt

from model import create_autoencoder
from data_utils import DataProcessor, generate_synthetic_data


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history.
    
    Args:
        history: Keras History object
        save_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot MAE
    if 'mae' in history.history:
        axes[1].plot(history.history['mae'], label='Training MAE')
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def train_autoencoder(config, data_path=None, use_synthetic=False):
    """
    Train the autoencoder model on SM data.
    
    Args:
        config (dict): Configuration dictionary
        data_path (str): Path to SM data file (optional if use_synthetic=True)
        use_synthetic (bool): Whether to use synthetic data for testing
        
    Returns:
        model: Trained autoencoder model
        history: Training history
    """
    print("="*50)
    print("Training Autoencoder for Anomaly Detection")
    print("="*50)
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    
    # Load and prepare data
    if use_synthetic:
        print("\nGenerating synthetic data for demonstration...")
        sm_data, _ = generate_synthetic_data(
            n_samples=10000,
            n_features=config['model']['input_dim'],
            random_state=42
        )
        # Save synthetic data
        os.makedirs(config['data']['raw_data_path'], exist_ok=True)
        synthetic_path = os.path.join(config['data']['raw_data_path'], 'sm_data_synthetic.npy')
        np.save(synthetic_path, sm_data)
        print(f"Synthetic SM data saved to {synthetic_path}")
        
        # Split and preprocess
        train_data, val_data, test_data = data_processor.split_data(sm_data)
        train_data = data_processor.preprocess_data(train_data, fit_scaler=True)
        val_data = data_processor.preprocess_data(val_data, fit_scaler=False)
        test_data = data_processor.preprocess_data(test_data, fit_scaler=False)
        
        datasets = {'train': train_data, 'val': val_data, 'test': test_data}
    else:
        if data_path is None:
            data_path = os.path.join(
                config['data']['raw_data_path'],
                config['data']['sm_data_file']
            )
        print(f"\nLoading SM data from {data_path}...")
        datasets = data_processor.prepare_training_data(data_path)
    
    print(f"Training samples: {len(datasets['train'])}")
    print(f"Validation samples: {len(datasets['val'])}")
    print(f"Test samples: {len(datasets['test'])}")
    
    # Create model
    print("\nCreating autoencoder model...")
    model = create_autoencoder(config)
    
    # Build model
    model.build(input_shape=(None, config['model']['input_dim']))
    model.summary()
    
    # Setup callbacks
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'],
            verbose=1,
            min_lr=1e-7
        ),
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'autoencoder_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=config['logging']['tensorboard_dir'],
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        datasets['train'],
        datasets['train'],  # Autoencoder: output = input
        validation_data=(datasets['val'], datasets['val']),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model weights
    model_save_path = config['training']['model_save_path']
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save weights
    weights_path = model_save_path.replace('.keras', '_weights.weights.h5')
    model.save_weights(weights_path)
    print(f"\nModel weights saved to {weights_path}")
    
    # Save model config
    import json
    config_path = model_save_path.replace('.keras', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model config saved to {config_path}")
    
    # Save scaler
    scaler_path = os.path.join(config['data']['processed_data_path'], 'scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    data_processor.save_scaler(scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Plot training history
    plot_training_history(history, 'training_history.png')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_mae = model.evaluate(datasets['test'], datasets['test'], verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    return model, history


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Autoencoder for Anomaly Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to SM data file')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for demonstration')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    model, history = train_autoencoder(config, args.data, args.synthetic)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == '__main__':
    main()
