# Visualizing the datasets
### code

```python
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
```

![[images/variable_plots_objall.png]]

These plots show the distributions of all variables in the background and signal datasets after scaling. The background distributions are shown in blue, while the signal distributions are overlaid in orange. The y-axis is on a logarithmic scale to better visualize differences in the tails of the distributions.

## Stats
```python
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
plt.savefig("variable_statistics.png")
```
![[images/variable_statistics.png]]

    The above code calculates and prints the mean, standard deviation, and variance for each variable in both the background and signal datasets after scaling. It also creates bar plots to visualize these statistics for easier comparison between the two datasets.

**As we can see, since we've scaled the data on the background statistics, the background mean is 0, stddev close to 1 and variance is 1 for most variables. The signal statistics deviate from these values, indicating differences in the distributions between background and signal events.**