import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import h5py


# Normal 
normal_data = np.random.normal(0, 1, (1000, 10))

# Anomalous data: Gaussian centered at 5
anomalous_data = np.random.normal(5, 1, (200, 10))

# Combine
X = np.vstack([normal_data, anomalous_data])
y = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomalous_data))])  # 0=normal, 1=anomaly

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (train only on normal data!)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_train = X_train[y_train == 0]  # only normal samples

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ---------------------
# 2. Define Autoencoder
# ---------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=10, latent_dim=4):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# ---------------------
# 3. Train Autoencoder
# ---------------------
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

n_epochs = 150
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, X_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {loss.item():.4f}")

# ---------------------
# 4. Detect anomalies
# ---------------------
model.eval()
with torch.no_grad():
    reconstructed = model(X_test)
    mse = torch.mean((X_test - reconstructed) ** 2, dim=1)

# Choose threshold (e.g., 95th percentile of normal reconstruction error)
threshold = torch.quantile(mse[y_test == 0], 0.95)
print(f"Threshold: {threshold.item():.4f}")

# Predict anomalies
y_pred = (mse > threshold).float()

# ---------------------
# 5. Evaluate
# ---------------------
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
