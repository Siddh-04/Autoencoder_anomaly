# Autoencoder

The Autoencoder for anomaly detection for the search of BSM data of bulk gravitron

The datasets and models have been uploaded to the respective directories
**I've uploaded the codes with respective codes in the markdown format since I don't ave enought computing power to run the model inside the Jupyter Notebook**



# getting the data

 An autoencoder is a type of neural network used to learn efficient codings of input data. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for the purpose of dimensionality reduction.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch

# random seed 
np.random.seed(42)
torch.manual_seed(42)
```

### loading the dataset

```python
data = torch.load("events_data.pt")
x_bkg = data["x_bkg"]
x_sig = data["x_sig"]
var_names = data["var_names"]

```
##### By observing all the features and the their distribution in the background and the signal
I'mm gonna choose following features 
`MET_0_pt, FatJet_0_pt, FatJet_0_mass, FatJet_0_tau1, FatJet_0_tau2, FatJet_0_tau3, FatJet_1_pt,  FatJet_1_mass, FatJet_1_tau1, FatJet_1_tau2, FatJet_1_tau3, FatJet_2_pt, FatJet_2_mass, FatJet_3_pt, Electron_0_pt,Electron_1_pt,Electron_2_pt, Electron_3_pt, Muon_0_pt, Muon_1_pt, Muon_2_pt,  Muon_3_pt`

### scaling
```python

# scale the data
scaler = StandardScaler()
_ = scaler.fit(x_bkg)
x_bkg_scaled = scaler.transform(x_bkg)
x_sig_scaled = scaler.transform(x_sig)
        
# splitting
X_train, X_test = train_test_split(x_bkg_scaled, test_size=0.1, shuffle=True)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.zeros(X_train.shape[0], dtype=torch.float32)  
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.zeros(X_test.shape[0], dtype=torch.float32)  


X_sig = torch.tensor(x_sig_scaled, dtype=torch.float32)
Y_sig = torch.ones(X_sig.shape[0], dtype=torch.float32)  

# concanate
X_eval = torch.cat([X_test, X_sig])
Y_eval = torch.cat([Y_test, Y_sig])

# saving X_eval and Y_eval for later evaluation and comparing reconstruction
torch.save({"X_eval": X_eval, "Y_eval": Y_eval}, "eval_data.pt")

```
### visualizing the datasets
```python
import matplotlib.pyplot as plt
import seaborn as sns
def plot_variables(x_bkg, x_sig, var_names, obj_index=None, ncols=3):
    """
    Plots distributions for variables in x_bkg and x_sig.
    
    Parameters:
        x_bkg, x_sig : np.ndarray
            Background and signal datasets (same var_names order)
        var_names : list
            Variable names like 'FatJet_0_pt', 'Muon_1_eta', etc.
        obj_index : int or None
            If int, plot only that object index (e.g., 0 → '_0_')
            If None, plot *all* objects (all indices)
        ncols : int
            Number of columns in subplot grid
    """
    # Check if input arrays are valid
    if not isinstance(x_bkg, np.ndarray) or not isinstance(x_sig, np.ndarray):
        raise ValueError("Input data must be numpy arrays.")
    if x_bkg.shape[1] != x_sig.shape[1]:
        raise ValueError("Background and signal data must have the same number of features.")
    if len(var_names) != x_bkg.shape[1]:
        raise ValueError("Variable names must match the number of features in the data.")

    # Filter variable names based on obj_index
    if obj_index is not None:
        tag = f"_{obj_index}_"
        selected = [name for name in var_names if tag in name]
    else:
        selected = var_names[:]  # all variables
    
    n_vars = len(selected)
    nrows = int(np.ceil(n_vars / ncols))
    
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
    axes = axes.flatten()
    
    for plot_idx, name in enumerate(selected):
        ax = axes[plot_idx]
        i = var_names.index(name)
        
        h_bkg = ax.hist(x_bkg[:, i], bins=100, density=True, log=True, 
                        label="Bkg", color="royalblue", alpha=0.7)
        ax.hist(x_sig[:, i], bins=h_bkg[1], density=True, log=True, 
                histtype="step", color="darkorange", label="Sig", linewidth=1.3)
        ax.set_xlabel(name)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for j in range(len(selected), len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(f"variable_plots_obj{obj_index if obj_index is not None else 'all'}.png")

# Plot all variables
plot_variables(x_bkg.numpy(), x_sig.numpy(), var_names, obj_index=None, ncols=3)
```



![[[image/variable_plots_objall.png]]](https://github.com/Siddh-04/Autoencoder_anomaly/blob/de720cfb69399695df6359e5f372cef4d10d02a1/images/variable_plots_objall.png)




## Model
After trying multiple architectures, activations functions and latent dimensions, I found the following architecture to be the best performing one.

Perhaps due to the small size of the dataset and the simplicity of the features, a simple architecture with one hidden layer and sigmoid activation function worked the best.

```python

class Autoencoder(nn.Module):
    def __init__(self, input_dim=X_train.shape[1], latent_dim=18):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
```

This is the simplest and the most successful AE arcitecture 
$$
22 \rightarrow\text{ [sigmoid]} \implies  18 \implies 22
$$

The Arcitectures and model I've tries listed it here 
| Architecture | Activation | Latent Dim | Notes |
|--------------|------------|------------|-------|
|  |  |  |  |


### Training 
```python
model = Autoencoder()
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',       # we want to minimize loss
    factor=0.8,       # reduce LR by a factor of 8/10
    patience=10,      # wait 10 epochs before reducing
    min_lr=1e-8,      
    # verbose=True
)


# Training 
n_epochs =400 
for epoch in range(n_epochs):
    optimizer.zero_grad()
    z = model.encoder(X_train)
    model.train()
    output = model.decoder(z)
    loss = criterion(output, X_train) + 1e-4 * torch.mean(torch.abs(z)) # L1 regularization 
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())
    if (epoch+1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {loss.item():.4f} Learning Rate: {current_lr:.6f}")

```

### Evaluating & saving the model 
```python
model.eval()
torch.save(model.state_dict(), "autoencoder_model.pth")

with torch.no_grad(): 
    reconstructed = model(X_eval)
    np.save("reconstructed_data.npy", reconstructed.numpy())
    # wmse_loss = weighted_mse_loss(reconstructed, X_eval, weights)   
    mse_loss = nn.MSELoss(reduction='none')
    
    losses = mse_loss(reconstructed, X_eval).mean(dim=1).numpy() 
    a = model.encoder(X_eval).detach().numpy()
    b = model.encoder(X_sig).detach().numpy()

# saving a
    np.save("encoded_eval.npy", a) 
    np.save("encoded_sig.npy", b)

Y_eval_np = Y_eval.numpy()
fpr , tpr, thresholds = roc_curve(Y_eval_np, losses)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)


```

