#!/usr/bin/env python3
"""
Quick demo script to test the autoencoder anomaly detection system
"""

import os
import yaml
import argparse


def run_demo(quick_mode=True):
    """
    Run a complete demo of the autoencoder anomaly detection system.
    
    Args:
        quick_mode (bool): If True, uses reduced epochs for faster testing
    """
    print("="*70)
    print("AUTOENCODER ANOMALY DETECTION DEMO")
    print("Train on Standard Model data, detect Beyond Standard Model anomalies")
    print("="*70)
    
    # Load config
    config_file = 'config.yaml'
    if quick_mode:
        print("\n[QUICK MODE] Using reduced epochs for demonstration...")
        # Create quick config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        config['training']['epochs'] = 5
        config['training']['early_stopping_patience'] = 3
        
        quick_config_file = 'demo_config.yaml'
        with open(quick_config_file, 'w') as f:
            yaml.dump(config, f)
        config_file = quick_config_file
    
    # Step 1: Train the model
    print("\n" + "="*70)
    print("STEP 1: Training Autoencoder on SM Data")
    print("="*70)
    train_cmd = f"python3 src/train.py --config {config_file} --synthetic"
    print(f"Running: {train_cmd}\n")
    exit_code = os.system(train_cmd)
    
    if exit_code != 0:
        print("\n[ERROR] Training failed!")
        return False
    
    print("\n[SUCCESS] Training completed!")
    
    # Step 2: Evaluate anomaly detection
    print("\n" + "="*70)
    print("STEP 2: Evaluating Anomaly Detection on BSM Data")
    print("="*70)
    eval_cmd = f"python3 src/evaluate.py --config {config_file} --synthetic"
    print(f"Running: {eval_cmd}\n")
    exit_code = os.system(eval_cmd)
    
    if exit_code != 0:
        print("\n[ERROR] Evaluation failed!")
        return False
    
    print("\n[SUCCESS] Evaluation completed!")
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated outputs:")
    print("  - models/autoencoder_weights.weights.h5  (trained model weights)")
    print("  - models/autoencoder_config.json         (model configuration)")
    print("  - training_history.png                   (training curves)")
    print("  - error_distribution.png                 (SM vs BSM reconstruction errors)")
    print("  - roc_curve.png                          (ROC curve for anomaly detection)")
    print("  - confusion_matrix.png                   (classification results)")
    print("\nYou can now:")
    print("  1. View the generated plots to see the results")
    print("  2. Use the trained model for anomaly detection on real CERN data")
    print("  3. Adjust hyperparameters in config.yaml and retrain")
    print("  4. Explore notebooks/example_workflow.ipynb for interactive analysis")
    
    # Clean up quick config
    if quick_mode and os.path.exists('demo_config.yaml'):
        os.remove('demo_config.yaml')
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Demo script for autoencoder anomaly detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 demo.py              # Run quick demo (5 epochs)
  python3 demo.py --full       # Run full demo (uses config.yaml epochs)
        """
    )
    parser.add_argument('--full', action='store_true',
                       help='Run full training (not quick mode)')
    
    args = parser.parse_args()
    
    success = run_demo(quick_mode=not args.full)
    
    if not success:
        print("\n[FAILED] Demo encountered errors. Please check the output above.")
        exit(1)


if __name__ == '__main__':
    main()
