# Copilot Instructions for Autoencoder Anomaly Detection

## Project Overview

This repository implements an autoencoder-based anomaly detection system for identifying Beyond Standard Model (BSM) physics data, specifically for the search of bulk gravitons. The project uses PyTorch to build and train autoencoders on particle physics data.

## Repository Structure

- `/models/` - Contains Python model implementations (model74.py, model76.py, sp_74.py)
- `/images/` - Visualization and result images
- `/scratch/` - Working directory for experiments
- `Autoencoder_model_code.md` - Detailed documentation of the autoencoder implementation
- `datasets.md` - Information about the datasets used
- `Results.md` - Model performance and results
- `autoencoder_model.pth` - Pre-trained model weights

## Technology Stack

- **Primary Language**: Python
- **Deep Learning Framework**: PyTorch
- **Data Processing**: NumPy, pandas, scikit-learn
- **Key Libraries**: 
  - `torch` and `torch.nn` for neural network construction
  - `sklearn.preprocessing.StandardScaler` for data normalization
  - `torch.optim` for optimization algorithms
  - `sklearn.model_selection.train_test_split` for data splitting

## Coding Standards

### Python Style
- Follow PEP 8 style guidelines
- Use meaningful variable names that reflect physics concepts (e.g., `x_bkg` for background data, `x_sig` for signal data)
- Add comments for complex physics-specific operations
- Use type hints where appropriate for better code clarity

### PyTorch Conventions
- Always use `model.train()` before training and `model.eval()` before evaluation
- Use `torch.no_grad()` context manager during inference to reduce memory consumption
- Properly manage random seeds for reproducibility (`torch.manual_seed()`, `np.random.seed()`)
- Save models using `torch.save()` and load with `torch.load()`

### Model Architecture
- The autoencoder follows an encoder-decoder architecture:
  - **Encoder**: Progressively reduces dimensionality (input_dim → 64 → 32 → 16 → latent_dim)
  - **Decoder**: Reconstructs input from latent representation (latent_dim → 32 → 64 → 32 → input_dim)
- Use ReLU activation functions between layers
- Consider L1 regularization on the latent space to encourage sparsity

### Training Practices
- Use MSE (Mean Squared Error) loss as the primary criterion
- Implement learning rate scheduling (e.g., `ReduceLROnPlateau`)
- Monitor loss every 10 epochs and log learning rate changes
- Apply gradient clipping if necessary to prevent exploding gradients
- Use early stopping based on validation loss

### Data Handling
- Always scale/normalize data using `StandardScaler` before training
- Split data into train/test sets with proper shuffling
- Convert data to `torch.float32` tensors for PyTorch compatibility
- Handle background (x_bkg) and signal (x_sig) data separately

## Physics Context

### Features Used
The model works with particle physics features including:
- MET (Missing Transverse Energy): `MET_0_pt`
- FatJet properties: `FatJet_{i}_pt`, `FatJet_{i}_mass`, `FatJet_{i}_tau{1,2,3}`
- Electron properties: `Electron_{i}_pt`
- Muon properties: `Muon_{i}_pt`

### Anomaly Detection Approach
- Train autoencoder on background (Standard Model) data only
- Use reconstruction error to identify anomalies (potential BSM signals)
- Higher reconstruction loss indicates potential signal events

## Best Practices for Code Changes

1. **Preserve Model Architecture**: When modifying the model, ensure changes are backward compatible with saved weights
2. **Document Physics Rationale**: Explain why specific features or architectural choices are made from a physics perspective
3. **Validate on Both Data Types**: Test changes on both background and signal data
4. **Save Intermediate Results**: Use `np.save()` to save encoded representations and reconstructed data for analysis
5. **Maintain Reproducibility**: Keep random seeds fixed for experiments
6. **Performance Monitoring**: Always track and log training metrics (loss, learning rate, etc.)

## Development Notes

- The repository includes code in markdown format due to computational limitations
- Model implementations may reference variables (like `X_train`) that are defined in the preprocessing pipeline
- When making changes, ensure consistency across all model variants (model74.py, model76.py, etc.)

## Testing and Validation

- Verify model loading/saving functionality with `.pth` files
- Test data preprocessing pipeline end-to-end
- Validate reconstruction quality on hold-out test set
- Check that encoded representations have expected dimensionality
- Ensure numerical stability during training (no NaN/Inf values)

## Common Patterns

```python
# Model initialization (X_train must be defined before creating the model)
model = Autoencoder(input_dim=X_train.shape[1], latent_dim=8)

# Training loop with regularization
loss = criterion(output, X_train) + 1e-4 * torch.mean(torch.abs(z))

# Evaluation with no gradient
model.eval()
with torch.no_grad():
    reconstructed = model(X_eval)
    mse_loss = nn.MSELoss(reduction='none')
    losses = mse_loss(reconstructed, X_eval).mean(dim=1).numpy()
```

## Questions to Ask When Reviewing Code

1. Does this change affect model reproducibility?
2. Are gradients properly managed (zeroed, computed, applied)?
3. Is the model in the correct mode (train/eval)?
4. Are tensors on the correct device (CPU/GPU)?
5. Is data properly scaled and normalized?
6. Will this work with the existing saved model weights?
