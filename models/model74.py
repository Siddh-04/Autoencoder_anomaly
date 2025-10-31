import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, input_dim=X_train.shape[1], latent_dim=8):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            # nn.BatchNorm1d(44),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            # nn.ReLU()
        )
        # self.logits = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
model = Autoencoder()

criterion = lambda x, x_hat: weighted_mse_loss(x, x_hat, weights)
# criterion = model.loss
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',       # we want to minimize loss
    factor=0.5,       # reduce LR by a factor of 2
    patience=10,      # wait 10 epochs before reducing
    min_lr=1e-8,      # don't go below this
    # verbose=True
)


# Training loop
n_epochs = 1000
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

model.eval()
with torch.no_grad(): # torch.no_grad() is used to disable gradient calculation, which reduces memory consumption and speeds up computations during inference.
    reconstructed = model(X_eval)
    np.save("reconstructed_data.npy", reconstructed.numpy())
    # wmse_loss = weighted_mse_loss(reconstructed, X_eval, weights)   
    mse_loss = nn.MSELoss(reduction='none')
    
    losses = mse_loss(reconstructed, X_eval).mean(dim=1).numpy()
    # mse = torch.mean((X_eval - reconstructed) ** 2, dim=1).numpy() # calculating mse 
    a = model.encoder(X_eval).detach().numpy()
    b = model.encoder(X_sig).detach().numpy()
    np.save("encoded_eval.npy", a) 
    np.save("encoded_sig.npy", b)
