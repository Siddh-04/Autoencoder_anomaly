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

class Autoencoder(nn.Module):
    def __init__(self, input_dim=X_train.shape[1], latent_dim=4):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            # nn.BatchNorm1d(44),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim),
            
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Linear(32, input_dim),
            
        )
        # self.logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
model = Autoencoder(input_dim=X_train.shape[1], latent_dim=4)
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
n_epochs = 600
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
