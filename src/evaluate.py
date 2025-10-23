"""
Evaluation script for anomaly detection using trained autoencoder
"""

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

from model import reconstruction_error
from data_utils import DataProcessor, generate_synthetic_data


def plot_reconstruction_error_distribution(sm_errors, bsm_errors, threshold, save_path='error_distribution.png'):
    """
    Plot reconstruction error distributions for SM and BSM data.
    
    Args:
        sm_errors (np.ndarray): Reconstruction errors for SM data
        bsm_errors (np.ndarray): Reconstruction errors for BSM data
        threshold (float): Anomaly threshold
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(sm_errors, bins=50, alpha=0.7, label='SM (Normal)', color='blue', density=True)
    plt.hist(bsm_errors, bins=50, alpha=0.7, label='BSM (Anomaly)', color='red', density=True)
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Reconstruction Error Distribution: SM vs BSM', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error distribution plot saved to {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path='roc_curve.png'):
    """
    Plot ROC curve.
    
    Args:
        fpr (np.ndarray): False positive rates
        tpr (np.ndarray): True positive rates
        roc_auc (float): Area under ROC curve
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve for Anomaly Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve plot saved to {save_path}")


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal (SM)', 'Anomaly (BSM)'],
                yticklabels=['Normal (SM)', 'Anomaly (BSM)'])
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")


def evaluate_anomaly_detection(config, model_path=None, sm_data_path=None, 
                               bsm_data_path=None, use_synthetic=False):
    """
    Evaluate anomaly detection performance.
    
    Args:
        config (dict): Configuration dictionary
        model_path (str): Path to trained model
        sm_data_path (str): Path to SM test data
        bsm_data_path (str): Path to BSM data
        use_synthetic (bool): Whether to use synthetic data
        
    Returns:
        dict: Evaluation metrics
    """
    print("="*50)
    print("Evaluating Anomaly Detection Performance")
    print("="*50)
    
    # Load model
    if model_path is None:
        model_path = config['training']['model_save_path']
    
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    
    # Load scaler
    scaler_path = os.path.join(config['data']['processed_data_path'], 'scaler.pkl')
    if os.path.exists(scaler_path):
        data_processor.load_scaler(scaler_path)
        print(f"Loaded scaler from {scaler_path}")
    
    # Load data
    if use_synthetic:
        print("\nGenerating synthetic data for demonstration...")
        sm_data, bsm_data = generate_synthetic_data(
            n_samples=10000,
            n_features=config['model']['input_dim'],
            random_state=42
        )
        
        # Use test split of SM data
        _, _, sm_test = data_processor.split_data(sm_data)
        sm_test = data_processor.preprocess_data(sm_test, fit_scaler=False)
        bsm_test = data_processor.preprocess_data(bsm_data, fit_scaler=False)
    else:
        if sm_data_path is None:
            sm_data_path = os.path.join(
                config['data']['raw_data_path'],
                config['data']['sm_data_file']
            )
        if bsm_data_path is None:
            bsm_data_path = os.path.join(
                config['data']['raw_data_path'],
                config['data']['bsm_data_file']
            )
        
        print(f"\nLoading SM data from {sm_data_path}...")
        datasets = data_processor.prepare_training_data(sm_data_path)
        sm_test = datasets['test']
        
        print(f"Loading BSM data from {bsm_data_path}...")
        bsm_test = data_processor.prepare_anomaly_data(bsm_data_path)
    
    print(f"SM test samples: {len(sm_test)}")
    print(f"BSM test samples: {len(bsm_test)}")
    
    # Calculate reconstruction errors
    print("\nCalculating reconstruction errors...")
    sm_errors = reconstruction_error(model, sm_test)
    bsm_errors = reconstruction_error(model, bsm_test)
    
    # Determine threshold
    threshold_percentile = config['evaluation']['anomaly_threshold_percentile']
    threshold = np.percentile(sm_errors, threshold_percentile)
    print(f"\nAnomaly threshold ({threshold_percentile}th percentile): {threshold:.6f}")
    
    # Statistics
    print(f"\nSM reconstruction error - Mean: {np.mean(sm_errors):.6f}, Std: {np.std(sm_errors):.6f}")
    print(f"BSM reconstruction error - Mean: {np.mean(bsm_errors):.6f}, Std: {np.std(bsm_errors):.6f}")
    
    # Create labels and predictions
    y_true = np.concatenate([np.zeros(len(sm_errors)), np.ones(len(bsm_errors))])
    errors = np.concatenate([sm_errors, bsm_errors])
    y_pred = (errors > threshold).astype(int)
    
    # Calculate metrics
    print("\n" + "="*50)
    print("Classification Report:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['SM (Normal)', 'BSM (Anomaly)']))
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot results
    plot_reconstruction_error_distribution(sm_errors, bsm_errors, threshold)
    plot_roc_curve(fpr, tpr, roc_auc)
    plot_confusion_matrix(cm)
    
    # Calculate detection metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    metrics = {
        'roc_auc': roc_auc,
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    print("\n" + "="*50)
    print("Summary Metrics:")
    print("="*50)
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    return metrics


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Autoencoder Anomaly Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model')
    parser.add_argument('--sm_data', type=str, default=None,
                       help='Path to SM test data')
    parser.add_argument('--bsm_data', type=str, default=None,
                       help='Path to BSM data')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for demonstration')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluate model
    metrics = evaluate_anomaly_detection(
        config,
        model_path=args.model,
        sm_data_path=args.sm_data,
        bsm_data_path=args.bsm_data,
        use_synthetic=args.synthetic
    )
    
    print("\n" + "="*50)
    print("Evaluation completed successfully!")
    print("="*50)


if __name__ == '__main__':
    main()
