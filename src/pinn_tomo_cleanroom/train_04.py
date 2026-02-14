from .initialize_network_03 import TTPinn, VPinn, compute_data_loss, boundary_condition_loss, eikonal_loss
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
import wandb
import pandas as pd

# Training setup
training_percent = 0.8
batch_size = 64
epochs = 5000
device = torch.device("cpu")

data = pd.read_csv("data/marmousi_training_picks.csv")
Xs = torch.from_numpy(data.sx.to_numpy())
Zs = torch.from_numpy(data.sz.to_numpy())
Xr = torch.from_numpy(data.rx.to_numpy())
Zr = torch.from_numpy(data.rz.to_numpy())
T = torch.from_numpy(data.t_obs.to_numpy())

tt_list = [Xr, Zr, Xs, Zs]
v_list = [Xr, Zr]

tt_tensor = torch.stack(tt_list, dim=0)
v_tensor = torch.stack(v_list, dim=0)

tt_dataset = TensorDataset(tt_tensor, T)
v_dataset = TensorDataset(v_tensor, T)

training_size = int(len(tt_dataset) * training_percent)
test_size = len(tt_dataset) - training_size

tt_train, tt_test = random_split(tt_dataset, [training_size, test_size])
v_train, v_test = random_split(v_dataset, [training_size, test_size])

tt_train_loader = DataLoader(tt_train, batch_size=batch_size, shuffle=True)
tt_test_loader = DataLoader(tt_test, batch_size=batch_size, shuffle=False)
v_train_loader = DataLoader(v_train, batch_size=batch_size, shuffle=True)
v_test_loader = DataLoader(v_test, batch_size=batch_size, shuffle=False)

# Initialize model and loss
tt_model = TTPinn().to(device)
v_model = VPinn().to(device)
optimizer = optim.Adam()

# Initialize WandB
wandb.init(project="ant-pinn-inversion")

for epoch in range(epochs):
    optim.zero



