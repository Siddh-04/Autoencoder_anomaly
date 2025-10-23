"""
Data loading and preprocessing utilities for CERN particle physics data
"""

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


class DataProcessor:
    """
    Data processor for CERN particle physics data.
    Handles loading, preprocessing, and splitting of SM and BSM data.
    """
    
    def __init__(self, config):
        """
        Initialize data processor.
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def load_h5_data(self, filepath):
        """
        Load data from HDF5 file.
        
        Args:
            filepath (str): Path to HDF5 file
            
        Returns:
            np.ndarray: Loaded data
        """
        with h5py.File(filepath, 'r') as f:
            # Assume data is stored in 'data' key
            # Adjust based on actual HDF5 structure
            if 'data' in f.keys():
                data = f['data'][:]
            else:
                # If no 'data' key, use the first dataset
                key = list(f.keys())[0]
                data = f[key][:]
        return data
    
    def load_csv_data(self, filepath):
        """
        Load data from CSV file.
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            np.ndarray: Loaded data
        """
        df = pd.read_csv(filepath)
        return df.values
    
    def load_data(self, filepath):
        """
        Load data from file (supports .h5, .hdf5, .csv).
        
        Args:
            filepath (str): Path to data file
            
        Returns:
            np.ndarray: Loaded data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext in ['.h5', '.hdf5']:
            return self.load_h5_data(filepath)
        elif file_ext == '.csv':
            return self.load_csv_data(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def preprocess_data(self, data, fit_scaler=True):
        """
        Preprocess data: handle missing values and normalize.
        
        Args:
            data (np.ndarray): Raw data
            fit_scaler (bool): Whether to fit the scaler (True for training data)
            
        Returns:
            np.ndarray: Preprocessed data
        """
        # Handle missing values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize data
        if fit_scaler:
            data = self.scaler.fit_transform(data)
        else:
            data = self.scaler.transform(data)
        
        return data
    
    def split_data(self, data):
        """
        Split data into train, validation, and test sets.
        
        Args:
            data (np.ndarray): Data to split
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        test_split = self.config['data']['test_split']
        
        # First split: separate test set
        train_val_data, test_data = train_test_split(
            data, 
            test_size=test_split, 
            random_state=42
        )
        
        # Second split: separate validation from training
        val_size = val_split / (train_split + val_split)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size,
            random_state=42
        )
        
        return train_data, val_data, test_data
    
    def prepare_training_data(self, sm_data_path):
        """
        Prepare Standard Model data for training.
        
        Args:
            sm_data_path (str): Path to SM data file
            
        Returns:
            dict: Dictionary with 'train', 'val', 'test' datasets
        """
        # Load SM data
        sm_data = self.load_data(sm_data_path)
        
        # Split data
        train_data, val_data, test_data = self.split_data(sm_data)
        
        # Preprocess data
        train_data = self.preprocess_data(train_data, fit_scaler=True)
        val_data = self.preprocess_data(val_data, fit_scaler=False)
        test_data = self.preprocess_data(test_data, fit_scaler=False)
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def prepare_anomaly_data(self, bsm_data_path):
        """
        Prepare Beyond Standard Model (BSM) data for anomaly detection.
        
        Args:
            bsm_data_path (str): Path to BSM data file
            
        Returns:
            np.ndarray: Preprocessed BSM data
        """
        # Load BSM data
        bsm_data = self.load_data(bsm_data_path)
        
        # Preprocess using fitted scaler from training data
        bsm_data = self.preprocess_data(bsm_data, fit_scaler=False)
        
        return bsm_data
    
    def save_scaler(self, filepath):
        """Save the fitted scaler for later use"""
        import joblib
        joblib.dump(self.scaler, filepath)
    
    def load_scaler(self, filepath):
        """Load a previously fitted scaler"""
        import joblib
        self.scaler = joblib.load(filepath)


def generate_synthetic_data(n_samples=10000, n_features=57, anomaly_ratio=0.1, random_state=42):
    """
    Generate synthetic data for testing purposes.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        anomaly_ratio (float): Ratio of anomalies to generate
        random_state (int): Random seed
        
    Returns:
        tuple: (sm_data, bsm_data) Normal and anomaly data
    """
    np.random.seed(random_state)
    
    # Generate SM (normal) data from standard distribution
    sm_data = np.random.randn(n_samples, n_features)
    
    # Generate BSM (anomaly) data with different distribution
    n_anomalies = int(n_samples * anomaly_ratio)
    bsm_data = np.random.randn(n_anomalies, n_features) * 2 + 1  # Different scale and shift
    
    return sm_data, bsm_data
