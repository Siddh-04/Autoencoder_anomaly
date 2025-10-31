import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = torch.load("events_data.pt")
x_bkg = data["x_bkg"]
x_sig = data["x_sig"]
var_names = data["var_names"]

# # removing all variables related to phi
# non_phi_indices = [i for i, name in enumerate(var_names) if "phi" not in name]
# x_bkg = x_bkg[:, non_phi_indices]
# x_sig = x_sig[:, non_phi_indices]
# var_names = [name for name in var_names if "phi" not in name]

# print("After removing 'phi' variables:")
# print(f"Background data shape: {x_bkg.shape}")
# print(f"Signal data shape: {x_sig.shape}")


"""
# Not doing this part now

# print(var_names)
['MET_0_pt', 'MET_0_phi', 'FatJet_0_pt', 'FatJet_0_eta', 'FatJet_0_phi', 'FatJet_0_mass', 'FatJet_0_tau1', 'FatJet_0_tau2', 'FatJet_0_tau3', 'FatJet_1_pt', 'FatJet_1_eta', 'FatJet_1_phi', 'FatJet_1_mass', 'FatJet_1_tau1', 'FatJet_1_tau2', 'FatJet_1_tau3', 'FatJet_2_pt', 'FatJet_2_eta', 'FatJet_2_phi', 'FatJet_2_mass', 'FatJet_2_tau1', 'FatJet_2_tau2', 'FatJet_2_tau3', 'FatJet_3_pt', 'FatJet_3_eta', 'FatJet_3_phi', 'FatJet_3_mass', 'FatJet_3_tau1', 'FatJet_3_tau2', 'FatJet_3_tau3', 'FatJet_4_pt', 'FatJet_4_eta', 'FatJet_4_phi', 'FatJet_4_mass', 'FatJet_4_tau1', 'FatJet_4_tau2', 'FatJet_4_tau3', 'FatJet_5_pt', 'FatJet_5_eta', 'FatJet_5_phi', 'FatJet_5_mass', 'FatJet_5_tau1', 'FatJet_5_tau2', 'FatJet_5_tau3', 'Electron_0_pt', 'Electron_0_eta', 'Electron_0_phi', 'Electron_1_pt', 'Electron_1_eta', 'Electron_1_phi', 'Electron_2_pt', 'Electron_2_eta', 'Electron_2_phi', 'Electron_3_pt', 'Electron_3_eta', 'Electron_3_phi', 'Muon_0_pt', 'Muon_0_eta', 'Muon_0_phi', 'Muon_1_pt', 'Muon_1_eta', 'Muon_1_phi', 'Muon_2_pt', 'Muon_2_eta', 'Muon_2_phi', 'Muon_3_pt', 'Muon_3_eta', 'Muon_3_phi']

# considering following variables for analysis:
MET_0_pt 
FatJet_0_pt, FatJet_0_mass, FatJet_0_tau1, FatJet_0_tau2, FatJet_0_tau3
FatJet_1_pt, FatJet_1_mass, FatJet_1_tau1, FatJet_1_tau2, FatJet_1_tau3
FatJet_2_pt, FatJet_2_mass
FatJet_3_pt,
Electron_0_pt,
Electron_1_pt,
Electron_2_pt,
Electron_3_pt,
Muon_0_pt,
Muon_1_pt,
Muon_2_pt,
Muon_3_pt
""" 
# selecting only the important variables
imp_var = ["MET_0_pt",
           "FatJet_0_pt", "FatJet_0_mass", "FatJet_0_tau1", "FatJet_0_tau2", "FatJet_0_tau3",
           "FatJet_1_pt", "FatJet_1_mass", "FatJet_1_tau1", "FatJet_1_tau2", "FatJet_1_tau3",
           "FatJet_2_pt", "FatJet_2_mass",
           "FatJet_3_pt",
           "Electron_0_pt",
           "Electron_1_pt",
           "Electron_2_pt",
           "Electron_3_pt",
           "Muon_0_pt",
           "Muon_1_pt",
           "Muon_2_pt",
           "Muon_3_pt"]
imp_indices = [var_names.index(var) for var in imp_var]
x_bkg = x_bkg[:, imp_indices]
x_sig = x_sig[:, imp_indices]
var_names = imp_var

scaler = StandardScaler()
_ = scaler.fit(x_bkg)
x_bkg_scaled = scaler.transform(x_bkg)
x_sig_scaled = scaler.transform(x_sig)

# splitting
X_train, X_test = train_test_split(x_bkg_scaled, test_size=0.2, shuffle=True)
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Signal data shape: {x_sig_scaled.shape}")
# saving the selected data for future use
# selected_data = {
#     "x_bkg": x_bkg_scaled,
#     "x_sig": x_sig_scaled,
#     "var_names": var_names
# }
# torch.save(selected_data, "imp_events_data.pt")



# # only taking variables with name phi
# phi_indices = [i for i, name in enumerate(var_names) if "phi" in name]
# x_bkg_scaled = x_bkg_scaled[:, phi_indices]
# x_sig_scaled = x_sig_scaled[:, phi_indices]
# var_names = [name for name in var_names if "phi" in name]

# print(f"Selected {len(var_names)} 'phi' variables for analysis.")
# print("Variable Names:", var_names)

# # getting mean and std for each phi variable
# bkg_mean = torch.mean(torch.tensor(x_bkg_scaled), dim=0)
# bkg_std = torch.std(torch.tensor(x_bkg_scaled), dim=0)
# sig_mean = torch.mean(torch.tensor(x_sig_scaled), dim=0)
# sig_std = torch.std(torch.tensor(x_sig_scaled), dim=0)
# print("Background Data Mean for phi variables:", bkg_mean)
# print("Background Data Std Dev for phi variables:", bkg_std)
# print("Signal Data Mean for phi variables:", sig_mean)
# print("Signal Data Std Dev for phi variables:", sig_std)    


# print(x_bkg.shape, x_sig.shape)
# # print(x_bkg_scaled)
# print("---------------------")

# # now plotting histograms of phi variables for signal and background
# for i, name in enumerate(var_names):
#     print(f"Plotting histogram for variable: {name}")
#     plt.figure(figsize=(8, 5))
#     plt.hist(x_bkg_scaled[:, i], bins=100,  label='Background', density=True)
#     plt.hist(x_sig_scaled[:, i], bins=100,  label='Signal', density=True, linewidth=1.5, histtype='step')
#     plt.title(f'Histogram of {name}')
#     plt.xlabel(name)
#     plt.ylabel('Density')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'hist_{name}.png')
#     plt.close()




# # sig_mean = torch.mean(x_sig, dim=0)
# sig_std = torch.std(x_sig, dim=0)
# bkg_mean = torch.mean(x_bkg, dim=0)
# bkg_std = torch.std(x_bkg, dim=0)

# # print("Background Data Mean:", bkg_mean)
# # print("Background Data Std Dev:", bkg_std)
# # print("Signal Data Mean:", sig_mean)
# # print("Signal Data Std Dev:", sig_std)

# # calculating the discriminative power of each feature

# # discriminative power formula: |mean_signal - mean_background|**2 / (std_signal**2 + std_background**2)
# discriminative_power = torch.abs(sig_mean - bkg_mean) ** 2 / (sig_std ** 2 + bkg_std ** 2)

# print("Feature Discriminative Power:")
# for name, power in zip(var_names, discriminative_power):
#     print(f"{name}: {power.item():.8f}")

