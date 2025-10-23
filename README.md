# Autoencoder for Anomaly Detection in CERN Data

This repository contains an autoencoder-based anomaly detection system for identifying Beyond Standard Model (BSM) physics signals in CERN particle collision data. The model is trained on Standard Model (SM) background data and detects anomalies such as bulk graviton signals.

## 🎯 Overview

- **Purpose**: Detect BSM physics signals (anomalies) in particle physics data
- **Approach**: Train autoencoder on SM (normal/background) data, detect BSM (anomalies) via reconstruction error
- **Target Signal**: Bulk graviton and other BSM phenomena
- **Framework**: TensorFlow/Keras with Python 3.8+

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Siddh-04/Autoencoder_anomaly.git
cd Autoencoder_anomaly
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
Autoencoder_anomaly/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── model.py              # Autoencoder model architecture
│   ├── data_utils.py         # Data loading and preprocessing
│   ├── train.py              # Training script
│   └── evaluate.py           # Evaluation and anomaly detection
├── notebooks/
│   └── example_workflow.ipynb # Jupyter notebook demonstration
├── data/
│   ├── raw/                  # Raw CERN data files (not included)
│   └── processed/            # Preprocessed data
├── models/                   # Saved trained models
├── checkpoints/              # Training checkpoints
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 💻 Usage

### 1. Training the Model

Train the autoencoder on Standard Model (SM) data:

#### Using Command Line:

```bash
# With your own data
python src/train.py --data data/raw/sm_data.h5

# With synthetic data (for testing)
python src/train.py --synthetic

# With custom configuration
python src/train.py --config custom_config.yaml --data data/raw/sm_data.h5
```

#### Using Python:

```python
import yaml
from src.train import train_autoencoder

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Train model
model, history = train_autoencoder(config, data_path='data/raw/sm_data.h5')
```

### 2. Evaluating Anomaly Detection

Evaluate the model's ability to detect BSM anomalies:

#### Using Command Line:

```bash
# Evaluate with your data
python src/evaluate.py --sm_data data/raw/sm_data.h5 --bsm_data data/raw/bsm_data.h5

# Evaluate with synthetic data (for testing)
python src/evaluate.py --synthetic

# Use specific model
python src/evaluate.py --model models/autoencoder.keras --bsm_data data/raw/bsm_data.h5
```

#### Using Python:

```python
import yaml
from src.evaluate import evaluate_anomaly_detection

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Evaluate
metrics = evaluate_anomaly_detection(
    config,
    sm_data_path='data/raw/sm_data.h5',
    bsm_data_path='data/raw/bsm_data.h5'
)
```

### 3. Using the Jupyter Notebook

For an interactive demonstration:

```bash
jupyter notebook notebooks/example_workflow.ipynb
```

## ⚙️ Configuration

The `config.yaml` file contains all hyperparameters and settings:

```yaml
model:
  input_dim: 57              # Number of input features
  encoding_dims: [32, 16, 8] # Encoder layer sizes
  latent_dim: 4              # Bottleneck dimension
  activation: "relu"         # Activation function
  dropout_rate: 0.2          # Dropout for regularization

training:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15

evaluation:
  anomaly_threshold_percentile: 95  # Threshold for anomaly detection
```

Adjust these parameters based on your specific data and requirements.

## 📊 Data Format

### Input Data

The model expects numerical features from particle collision events. Supported formats:

- **HDF5** (`.h5`, `.hdf5`): Recommended for large datasets
- **CSV** (`.csv`): For smaller datasets

### Data Structure

- **SM Data**: Standard Model background events (for training)
- **BSM Data**: Beyond Standard Model signal events (for evaluation)

Each sample should be a feature vector representing a collision event. Example features:
- Particle momenta (px, py, pz)
- Energy measurements
- Angular distributions
- Derived kinematic variables

### Preparing Your Data

```python
import h5py
import numpy as np

# Example: Save data in HDF5 format
with h5py.File('data/raw/sm_data.h5', 'w') as f:
    f.create_dataset('data', data=your_sm_data)

with h5py.File('data/raw/bsm_data.h5', 'w') as f:
    f.create_dataset('data', data=your_bsm_data)
```

## 🏗️ Model Architecture

The autoencoder consists of:

### Encoder
- Input layer (57 features by default)
- Dense layers: [32, 16, 8] with ReLU activation
- Batch normalization and dropout after each layer
- Latent space: 4 dimensions

### Decoder
- Dense layers: [8, 16, 32] (mirror of encoder)
- Batch normalization and dropout after each layer
- Output layer: 57 features with linear activation

### Loss Function
- Mean Squared Error (MSE) between input and reconstruction

### Anomaly Detection
- Calculate reconstruction error for each sample
- Samples with error above threshold (95th percentile) are classified as anomalies

## 📈 Results

After training and evaluation, the following outputs are generated:

1. **Training History Plot** (`training_history.png`)
   - Training and validation loss curves
   - MAE metrics over epochs

2. **Error Distribution** (`error_distribution.png`)
   - Reconstruction error distributions for SM and BSM data
   - Anomaly threshold visualization

3. **ROC Curve** (`roc_curve.png`)
   - True Positive Rate vs False Positive Rate
   - AUC score

4. **Confusion Matrix** (`confusion_matrix.png`)
   - Classification performance summary

### Example Performance Metrics

```
ROC AUC Score: 0.95
Sensitivity (TPR): 0.89
Specificity (TNR): 0.94
Precision: 0.76
```

## 🔬 Methodology

1. **Training Phase**:
   - Autoencoder learns to reconstruct SM (normal) data
   - Model minimizes reconstruction error on background events
   - Early stopping prevents overfitting

2. **Anomaly Detection**:
   - Calculate reconstruction error for each event
   - SM events: low reconstruction error
   - BSM events: high reconstruction error (anomalies)
   - Threshold-based classification

3. **Evaluation**:
   - ROC curve analysis
   - Precision, recall, F1-score
   - Confusion matrix

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Siddh-04/Autoencoder_anomaly.git
cd Autoencoder_anomaly

# Create development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Make your changes and test
python src/train.py --synthetic
python src/evaluate.py --synthetic
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{autoencoder_anomaly,
  title={Autoencoder for Anomaly Detection in CERN Data},
  author={Your Name},
  year={2025},
  url={https://github.com/Siddh-04/Autoencoder_anomaly}
}
```

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- CERN Open Data Portal for particle physics datasets
- TensorFlow/Keras team for deep learning framework
- High Energy Physics community

## 📧 Contact

For questions or collaborations, please open an issue on GitHub or contact the repository maintainer.

---

**Note**: This repository contains the code framework for anomaly detection. Actual CERN data files are not included due to size constraints. Please download appropriate datasets from the CERN Open Data Portal or use your own experimental data.
