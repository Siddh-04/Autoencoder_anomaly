# here im gonna compare reconstructions on eval data
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

plot_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# load eval data
data = torch.load("eval_data.pt")
X_eval = data["X_eval"]
Y_eval = data["Y_eval"]

# load the reconstructed_data.npy
recon_data = np.load("reconstructed_data.npy")
print("Loaded reconstructed data of shape:", recon_data.shape)
print("Eval data shape:", X_eval.shape)
print("Eval labels shape:", Y_eval.shape)


X_eval_bkg = X_eval[Y_eval == 0]
recon_data_bkg = recon_data[Y_eval.numpy() == 0]
X_eval_sig = X_eval[Y_eval == 1]
recon_data_sig = recon_data[Y_eval.numpy() == 1]
Y_eval_bkg = Y_eval[Y_eval == 0]
Y_eval_sig = Y_eval[Y_eval == 1]

# counting number of bkg and sig events
# num_bkg = X_eval_bkg.shape[0]
# num_sig = X_eval_sig.shape[0]
# print(f"Number of background events: {num_bkg}")
# print(f"Number of signal events: {num_sig}")
# couting number of bkg and sig events in recon_data
# num_bkg_recon = recon_data_bkg.shape[0]
# num_sig_recon = recon_data_sig.shape[0]
# print(f"Number of background events in recon_data: {num_bkg_recon}")
# print(f"Number of signal events in recon_data: {num_sig_recon}")


"""
So now the plan is we have saved the weights based on variance from train.py
so we will load those weights here and use them to compute weighted mse for each feature for both bkg and signal eval data.
"""

# Load weights
# weights = torch.load("feature_weights.pt")
# print("Loaded feature weights of shape:", weights.shape)

# # Compute weighted MSE for background and signal
# def compute_weighted_mse(X, reconstructed, weights):
#     mse = (X - reconstructed) ** 2
#     weighted_mse = mse * weights
#     return weighted_mse.mean(dim=0)

# weighted_mse_bkg = compute_weighted_mse(X_eval_bkg, recon_data_bkg, weights)
# weighted_mse_sig = compute_weighted_mse(X_eval_sig, recon_data_sig, weights)
# print("Weighted MSE per feature (background):\n", weighted_mse_bkg.numpy())
# print("Weighted MSE per feature (signal):\n", weighted_mse_sig.numpy())

# # now declaring the anomaly score as the sum of weighted mse over all features
# def anomaly_score(X, reconstructed, weights):
#     mse = (X - reconstructed) ** 2
#     weighted_mse = mse * weights
#     score = weighted_mse.sum(dim=1)  # sum over features
#     return score
# anomaly_scores_bkg = anomaly_score(X_eval_bkg, recon_data_bkg, weights)
# anomaly_scores_sig = anomaly_score(X_eval_sig, recon_data_sig, weights)
# print("Example anomaly scores for background events:", anomaly_scores_bkg[:5].numpy())
# print("Example anomaly scores for signal events:", anomaly_scores_sig[:5].numpy())

# print("shape of anomaly scores bkg:", anomaly_scores_bkg.shape)
# print("shape of anomaly scores sig:", anomaly_scores_sig.shape)














# print(weights.numpy())
# plot some example reconstructions along 
num_examples = 10
fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples * 3))
for i in range(num_examples):
    j = i + plot_index  
    axes[i, 0].bar(range(X_eval.shape[1]), X_eval_bkg[j].numpy())
    axes[i, 0].set_title(f"data {j} (Label: {Y_eval_bkg[j].item()})")
    axes[i, 1].bar(range(X_eval.shape[1]), recon_data_bkg[j])
    axes[i, 1].set_title(f"Reconstructed Event {j} recon (Label: {Y_eval_bkg[j].item()})")
plt.title("Background Event Reconstructions")
plt.tight_layout()
plt.savefig("compare_reconstructions_bkg.png")

fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples * 3))
for i in range(num_examples):
    j = i + plot_index  
    axes[i, 0].bar(range(X_eval.shape[1]), X_eval_sig[j].numpy())
    axes[i, 0].set_title(f"data {j} (Label: {Y_eval_sig[j].item()})")
    axes[i, 1].bar(range(X_eval.shape[1]), recon_data_sig[j])
    axes[i, 1].set_title(f"Reconstructed Event {j} recon (Label: {Y_eval_sig[j].item()})")
plt.tight_layout()
plt.title("Signal Event Reconstructions")
plt.savefig("compare_reconstructions_sig.png")

"""





# Now getting the weighted mse for bkg only

# mse before weighting
mse_bkg = np.mean((X_eval_bkg.numpy() - recon_data_bkg)**2, axis=0)
mse_sig = np.mean((X_eval_sig.numpy() - recon_data_sig)**2, axis=0)
print("MSE per feature (background):\n", mse_bkg)
print("MSE per feature (signal):\n", mse_sig)



import torch
import torch.nn.functional as F

# --- Convert to torch tensors if they are NumPy arrays ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_bkg = torch.tensor(X_eval_bkg, dtype=torch.float32, device=device)
reconstructed_bkg = torch.tensor(recon_data_bkg, dtype=torch.float32, device=device)

X_sig = torch.tensor(X_eval_sig, dtype=torch.float32, device=device)
reconstructed_sig = torch.tensor(recon_data_sig, dtype=torch.float32, device=device)

# --- Define weights ---
n_features = X_bkg.shape[1]
weights = torch.nn.Parameter(torch.ones(n_features, device=device, requires_grad=True))


optimizer = torch.optim.Adam([weights], lr=1e-2)
n_epochs = 1000

# Compute per-feature MSE for background
mse_bkg = torch.mean((X_bkg - reconstructed_bkg)**2, dim=0)

# --- Step 2: Train weights on background data ---
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # Enforce positivity via softplus
    a = F.softplus(weights)
    # Normalize weights to sum to 1
    a = a / torch.sum(a)
    
    # Weighted total MSE (objective)
    mse_net = torch.sum(a * mse_bkg)
    
    mse_net.backward()
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:03d} | MSE(net) = {mse_net.item():.6f}")

# --- Step 3: Get final positive weights ---
a_final = F.softplus(weights).detach()
print("\nFinal positive weights:\n", a_final.cpu().numpy())

# Now calcuate weighted mse for signal data
# mse_bkg = torch.mean((X_bkg - reconstructed_bkg)**2, dim=0)
mse_bkg_weighted = torch.mean(a_final * mse_bkg)
print(f"Weighted MSE for background data: {mse_bkg_weighted.item():.6f}")
mse_sig = torch.mean((X_sig - reconstructed_sig)**2, dim=0)
mse_sig_weighted = torch.mean(a_final * mse_sig)
print(f"\nWeighted MSE for signal data: {mse_sig_weighted.item():.6f}")

def weighted_mse_per_event(X, reconstructed, a):
    diff_sq = (X - reconstructed) ** 2
    wmse = torch.matmul(diff_sq, a)
    return wmse.detach().cpu().numpy()

wmse_bkg = weighted_mse_per_event(X_bkg, reconstructed_bkg, a_final)
wmse_sig = weighted_mse_per_event(X_sig, reconstructed_sig, a_final)
print("\nExample weighted MSE values for background events:", wmse_bkg[:5])
print("Example weighted MSE values for signal events:", wmse_sig[:5])

plt.figure(figsize=(7,5))
plt.hist(wmse_bkg, bins=50, alpha=0.6, label='Background', density=True)
plt.hist(wmse_sig, bins=50, alpha=0.6, label='Signal', density=True)
plt.xlabel("Weighted MSE")
plt.ylabel("Density")
plt.legend()
plt.title("Weighted MSE Separation between Signal and Background")
plt.show()
"""