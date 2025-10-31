import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


# getting data from events_data.pt
data = torch.load("events_data.pt")
x_bkg = data["x_bkg"]
x_sig = data["x_sig"]
var_names = data["var_names"]

# imp_var = ["MET_0_pt",
#            "FatJet_0_pt", "FatJet_0_mass", "FatJet_0_tau1", "FatJet_0_tau2", "FatJet_0_tau3",
#            "FatJet_1_pt", "FatJet_1_mass", "FatJet_1_tau1", "FatJet_1_tau2", "FatJet_1_tau3",
#            "FatJet_2_pt", "FatJet_2_mass",
#            "FatJet_3_pt",
#            "Electron_0_pt",
#            "Electron_1_pt",
#            "Electron_2_pt",
#            "Electron_3_pt",
#            "Muon_0_pt",
#            "Muon_1_pt",
#            "Muon_2_pt",
#            "Muon_3_pt"]
# imp_indices = [var_names.index(var) for var in imp_var]
# x_bkg = x_bkg[:, imp_indices]
# x_sig = x_sig[:, imp_indices]
# var_names = imp_var

# removing all variables related to phi
# non_phi_indices = [i for i, name in enumerate(var_names) if "phi" not in name]
# x_bkg = x_bkg[:, non_phi_indices]
# x_sig = x_sig[:, non_phi_indices]
# var_names = [name for name in var_names if "phi" not in name]


# variances = x_bkg.var(dim=0)
# weights = 1.0 / variances
# weights = weights / weights.sum()  # normalize to sum=1
# def weighted_mse_loss(x, x_hat, weights):
#     diff = (x - x_hat) ** 2
#     weighted = diff * weights
#     return weighted.mean()
"""
Here we are defining the weights based on the variance of each feature in the background     data.
Features with higher variance get higher weights, emphasizing their importance in the loss calculation.
This helps the autoencoder focus on accurately reconstructing features that are more consistent in the background,
potentially improving its ability to detect anomalies.
"""
# saving weights for later use in eval.py
# torch.save(weights, "feature_weights.pt")


# scale the data
scaler = StandardScaler()
_ = scaler.fit(x_bkg)
x_bkg_scaled = scaler.transform(x_bkg)
x_sig_scaled = scaler.transform(x_sig)
    
# random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
        
# define training, test and validation datasets
X_train, X_test = train_test_split(x_bkg_scaled, test_size=0.1, shuffle=True)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.zeros(X_train.shape[0], dtype=torch.float32)  # background labels
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.zeros(X_test.shape[0], dtype=torch.float32)  # background labels
X_sig = torch.tensor(x_sig_scaled, dtype=torch.float32)
Y_sig = torch.ones(X_sig.shape[0], dtype=torch.float32)  # signal labels

# Combine test background and signal
X_eval = torch.cat([X_test, X_sig])
Y_eval = torch.cat([Y_test, Y_sig])

# saving X_eval and Y_eval for later use in eval.py
torch.save({"X_eval": X_eval, "Y_eval": Y_eval}, "eval_data.pt")




# ---------------------
# 2. Define Autoencoder with randomized weights
# ---------------------


class Autoencoder(nn.Module):
    def __init__(self, input_dim=X_train.shape[1], latent_dim=18):

        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 22),
            # nn.LeakyReLU(0.5)
            # nn.Linear(64,32),
            nn.Sigmoid(),
            # nn.ReLU(),
            # nn.Linear(32, 16),  
            # nn.BatchNorm1d(input_dim),
            # nn.ReLU(),
            nn.Linear(22, latent_dim),
            # nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            # nn.Sigmoid(),
            # nn.ReLU(),
            # nn.Linear(64, 16),
            # nn.ReLU(),
            # nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, input_dim),
            # nn.ReLU()
        )
        # self.logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# ---------------------
# 3. Train Autoencoder
# ---------------------
model = Autoencoder()
# summarize the model
print(model)


# criterion = lambda x, x_hat: weighted_mse_loss(x, x_hat, weights)
# criterion = model.loss
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',       # we want to minimize loss
    factor=0.8,       # reduce LR by a factor of 8/10
    patience=10,      # wait 10 epochs before reducing
    min_lr=1e-8,      # don't go below this
    # verbose=True
)


# Training loop
n_epochs =400 
for epoch in range(n_epochs):
    optimizer.zero_grad()
    z = model.encoder(X_train)
    model.train()
    output = model.decoder(z)
    loss = criterion(output, X_train) + 1e-4 * torch.mean(torch.abs(z)) # L1 regularization on latent space
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())
    if (epoch+1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {loss.item():.4f} Learning Rate: {current_lr:.6f}")
    # checking for early stopping after 10 epochs if min_lr is reached
    # if optimizer.param_groups[0]['lr'] <= 1e-6:
    #     counter = 0
    #     patience = 20
    # else:
    #     counter = -1  # disable early stopping if min_lr not reached



    # # early stopping check
    # if loss.item() < best_loss and optimizer.param_groups[0]['lr'] <= 1e-6:
    #     best_loss = loss.item()
    #     counter = 0
    # else:
    #     counter += 1
    # if counter >= patience:
    #     print(f"Early stopping at epoch {epoch+1}")
    #     break


model.eval()
# save the model
torch.save(model.state_dict(), "autoencoder_model.pth")
# recon = model(X_train).detach().numpy()
# np.save("reconstructed_data_train.npy", recon)
# recon_test = model(X_test).detach().numpy()
# np.save("reconstructed_data_test.npy", recon_test)
# recon_sig = model(X_sig).detach().numpy()
# np.save("reconstructed_data_sig.npy", recon_sig)



with torch.no_grad(): # torch.no_grad() is used to disable gradient calculation, which reduces memory consumption and speeds up computations during inference.
    reconstructed = model(X_eval)
    np.save("reconstructed_data.npy", reconstructed.numpy())
    # wmse_loss = weighted_mse_loss(reconstructed, X_eval, weights)   
    mse_loss = nn.MSELoss(reduction='none')
    
    losses = mse_loss(reconstructed, X_eval).mean(dim=1).numpy()
    # mse = torch.mean((X_eval - reconstructed) ** 2, dim=1).numpy() # calculating mse 
    a = model.encoder(X_eval).detach().numpy()
    b = model.encoder(X_sig).detach().numpy()

# saving a
    np.save("encoded_eval.npy", a) 
    np.save("encoded_sig.npy", b)

Y_eval_np = Y_eval.numpy()
fpr , tpr, thresholds = roc_curve(Y_eval_np, losses)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)

# plotting reconstruction error histograms
plt.figure(figsize=(8,6))
plt.hist(losses[Y_eval_np == 0], bins=100, alpha=0.6, label='Background', color='blue', density=True)
plt.hist(losses[Y_eval_np == 1], bins=100, alpha=0.6, label='Signal', color='red', density=True)
plt.xlabel('Reconstruction Loss (MSE)')
plt.ylabel('Density')
plt.title('Reconstruction Loss Distribution')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("reconstruction_loss_histogram.png")




# # Optionally: print optimal threshold (Youden's J statistic)
# youden_index = np.argmax(tpr - fpr)
# optimal_threshold = thresholds[youden_index]
# print(f"Optimal Threshold (Youden’s J): {optimal_threshold:.6f}")


# # Predict anomalies
# y_pred = (torch.tensor(losses) > optimal_threshold).float()





# # classification report
# from sklearn.metrics import classification_report
# print(classification_report(Y_eval, y_pred))
# plt.hist(losses[Y_eval == 0], bins=100, alpha=0.5, label='Background')
# plt.hist(losses[Y_eval == 1], bins=100, alpha=0.5, label='Signal')
# # plt.axvline(threshold.item(), color='r', linestyle='--', label='Threshold')
# plt.legend()
# plt.xlabel("Reconstruction Error (MSE)")
# plt.ylabel("Count")
# plt.show()
