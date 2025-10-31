import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from train import Autoencoder


# loading eval data
X_eval = torch.load("eval_data.pt")["X_eval"]
Y_eval = torch.load("eval_data.pt")["Y_eval"]


class Autoencoder(nn.Module):
    def __init__(self, input_dim=X_eval.shape[1], latent_dim=16):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Dropout(0.1),
            nn.Linear(64, input_dim),
            nn.ReLU()
        )
        # self.logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# Load the trained model
model = Autoencoder()
model.load_state_dict(torch.load("autoencoder_model.pth"))
model.eval()
# Evaluate the model and save the mse for each event each feature
losses = []
with torch.no_grad():
    reconstructed = model(X_eval)
    



# saving recontruted data for X_eval
reconstructed_data = reconstructed.numpy()
np.save("reconstructed_data_eval.npy", reconstructed_data)


# Compute ROC curve and AUC
Y_eval_np = Y_eval.numpy()
fpr, tpr, thresholds = roc_curve(Y_eval_np, losses)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

# overall mse on eval set
overall_mse = np.mean(losses)
print("Overall MSE on eval set:", overall_mse)
# mse for bkg and sig separately
mse_bkg = np.mean(losses[Y_eval_np == 0])
mse_sig = np.mean(losses[Y_eval_np == 1])
print("MSE for background events:", mse_bkg)
print("MSE for signal events:", mse_sig)

# Plot histogram of reconstruction losses
plt.figure(figsize=(8, 6))
plt.hist(losses[Y_eval_np == 0], bins=50, alpha=0.6, label='Background', color='blue', density=True)
plt.hist(losses[Y_eval_np == 1], bins=50, alpha=0.6, label='Signal', color='red', density=True)
plt.xlabel('Reconstruction Loss (MSE)')
plt.ylabel('Density')
plt.title('Histogram of Reconstruction Losses')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()



# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Autoencoder Anomaly Detection')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


