## Results of the best model (by AUC)

| **ROC curve**| **Reconstruction Loss** |
| --- | --- |
| ![[/imagesroc_curve_eval.png]] | ![[images/reconstruction_loss_histogram_eval.png]] |
| **AUC: 0.796** |  |

| **First 10 bkg and its reconstruction** | **10 signal events and their reconstruction** |
| --------------------------------------- | --------------------------------------------- |
| ![[compare_reconstructions_bkg.png]]    | ![[compare_reconstructions_sig.png]]          |
|                                         |                                               |

## Results of the my personal favorite model (by AUC)
| **ROC curve**              | **Reconstruction Loss**               |
| -------------------------- | ------------------------------------- |
| ![[74_roc_curve_eval.png]] | ![[74_recon_loss_histogram_eval.png]] |
| **AUC: 0.749** |  |

| **First 10 bkg and its reconstruction** | **10 signal events and their reconstruction** |
| --------------------------------------- | --------------------------------------------- |
| ![[74_compare_recon_bkg.png]]    | ![[74_compare_recon_sig.png]]          |

although its AUC is lower, I personally prefer this model since it produces better reconstructions visually.




#### Code for training and evaluating the model
### Model
```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
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
### roc and reconstruction loss histogram
```python
# loading eval data
eval_data = torch.load("eval_data.pt")
X_eval = eval_data["X_eval"]
Y_eval = eval_data["Y_eval"]

# getting model
model = Autoencoder(input_dim=X_eval.shape[1], latent_dim=18)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

with torch.no_grad():
    X_eval_recon = model(X_eval)

# loss
recon_loss = torch.mean((X_eval - X_eval_recon)**2, dim=1).numpy()

fpr, tpr, thresholds = roc_curve(Y_eval.numpy(), recon_loss)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")
```

