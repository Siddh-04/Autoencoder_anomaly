# getting the data and getting their standard deviation, mean, variance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

data = torch.load("events_data.pt")
x_bkg = data["x_bkg"]
x_sig = data["x_sig"]   
var_names = data["var_names"]
# scaling wrt electron_pt in the bkg
electron_pt_bkg = x_bkg[:, var_names.index("Electron_0_pt")]
scaler = StandardScaler()
_ = scaler.fit(x_bkg)
x_bkg_scaled = scaler.transform(x_bkg)
x_sig_scaled = scaler.transform(x_sig)  
# saving scaled data
torch.save({"x_bkg_scaled": x_bkg_scaled, "x_sig_scaled": x_sig_scaled, "var_names": var_names}, "scaled_ept_events_data.pt")

# NOw plotting the scaled distributions of each variable for bkg and signal
# num_vars = len(var_names)
# num_cols = 4
# num_rows = (num_vars + num_cols - 1) // num_cols  # ceiling division
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
# axes = axes.flatten()
# for i, var in enumerate(var_names):
#     axes[i].hist(x_bkg_scaled[:, i], bins=50, alpha=0.5, label='Background', color='blue', density=True)
#     axes[i].hist(x_sig_scaled[:, i], bins=50, alpha=0.5, label='Signal', color='red', density=True)
#     axes[i].set_title(var)
#     axes[i].legend()
# # Remove any unused subplots
# # unused means if num_vars is not a perfect square
# # for j in range(i + 1, len(axes)):
# #     fig.delaxes(axes[j])
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

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

# Example usage:
plot_variables(x_bkg_scaled, x_sig_scaled, var_names, obj_index=None, ncols=4)

# calculating and printing mean, stddev, variance for each variable
mean_bkg = np.mean(x_bkg_scaled, axis=0)
stddev_bkg = np.std(x_bkg_scaled, axis=0)
var_bkg = np.var(x_bkg_scaled, axis=0)  
mean_sig = np.mean(x_sig_scaled, axis=0)
stddev_sig = np.std(x_sig_scaled, axis=0)
var_sig = np.var(x_sig_scaled, axis=0)

for i, var in enumerate(var_names):
    print(f"Variable: {var}")
    print(f"  Background - Mean: {mean_bkg[i]:.4f}, StdDev: {stddev_bkg[i]:.4f}, Variance: {var_bkg[i]:.4f}")
    print(f"  Signal     - Mean: {mean_sig[i]:.4f}, StdDev: {stddev_sig[i]:.4f}, Variance: {var_sig[i]:.4f}")
    print()

# plotting mean, stddev, variance for each variable
x = np.arange(len(var_names))
width = 0.25  # width of the bars
fig, ax = plt.subplots(3, 1, figsize=(12, 18))
# Mean
ax[0].bar(x - width, mean_bkg, width, label='Background')
ax[0].bar(x, mean_sig, width, label='Signal')
ax[0].set_ylabel('Mean')
ax[0].set_title('Mean of Variables')
ax[0].set_xticks(x)
ax[0].set_xticklabels(var_names, rotation=45)
ax[0].legend()
# StdDev
ax[1].bar(x - width, stddev_bkg, width, label='Background')
ax[1].bar(x, stddev_sig, width, label='Signal')
ax[1].set_ylabel('Standard Deviation')
ax[1].set_title('Standard Deviation of Variables')
ax[1].set_xticks(x)
ax[1].set_xticklabels(var_names, rotation=45)
ax[1].legend()
# Variance
ax[2].bar(x - width, var_bkg, width, label='Background')
ax[2].bar(x, var_sig, width, label='Signal')
ax[2].set_ylabel('Variance')
ax[2].set_title('Variance of Variables')
ax[2].set_xticks(x)
ax[2].set_xticklabels(var_names, rotation=45)
ax[2].legend()
plt.tight_layout()
plt.show()