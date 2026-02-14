from .initialize_network_03 import TTPinn, VPinn, compute_data_loss, boundary_condition_loss, eikonal_loss
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
import wandb
import pandas as pd

# Training setup
training_percent = 0.8
batch_size = 64
epochs = 500
lbfgs_max_iter = 100
device = torch.device("cpu")

data = pd.read_csv("data/marmousi_training_picks.csv")
Xs = torch.from_numpy(data.sx.to_numpy()) / 4000.0 # normalize
Zs = torch.from_numpy(data.sz.to_numpy()) / 4000.0 # normalize
Xr = torch.from_numpy(data.rx.to_numpy()) / 4000.0 # normalize
Zr = torch.from_numpy(data.rz.to_numpy()) / 4000.0 # normalize
T = torch.from_numpy(data.t_obs.to_numpy()).float().view(-1,1) # shape [N, 1]

tt_list = [Xr, Zr, Xs, Zs]
v_list = [Xr, Zr]

tt_tensor = torch.stack(tt_list, dim=1)
v_tensor = torch.stack(v_list, dim=1)

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

all_params = list(tt_model.parameters()) + list(v_model.parameters())
criterion = torch.nn.CrossEntropyLoss()
optimizer1 = optim.Adam(all_params, lr=0.001)
optimizer2 = optim.LBFGS(all_params, lr=1.0, max_iter=20, history_size=100)

# Initialize WandB
wandb.init(project="tomo-cleanroom")
print("starting phase 1: adam optimization")
for epoch in range(epochs):
    running_loss = 0.0

    for batch_idx, (tt_data, v_data) in enumerate(zip(tt_train_loader, v_train_loader)):
        tt_inputs, tt_outputs = tt_data
        v_inputs, v_outputs = v_data

        rx = tt_inputs[:, 0:1] # Keep the 2nd dim for matrix math!
        rz = tt_inputs[:, 1:2]
        sx = tt_inputs[:, 2:3]
        sz = tt_inputs[:, 3:4]
        pde_x = torch.rand(batch_size, 1, requires_grad=True, device=device)
        pde_z = torch.rand(batch_size, 1, requires_grad=True, device=device)

        # zero gradients
        optimizer1.zero_grad()
        
        # Forward pass
        tt_pred = tt_model(rx, rz, sx, sz)
        v_pred = v_model(rx, rz)

        # Calculate losses and cumulative loss
        data_loss = compute_data_loss(tt_pred, tt_outputs)
        pde_loss = boundary_condition_loss(tt_pred, tt_outputs)
        eik_loss = eikonal_loss(tt_pred, tt_outputs, v_pred, v_outputs)
        total_loss = criterion(data_loss + pde_loss + eik_loss)

        # Backwards pass
        total_loss.backward()
        optimizer1.step()

        running_loss += total_loss.item()

    print(f"Epoch [{epoch+1}/{epoch}], Loss: {running_loss/len(tt_train_loader):.4f}")
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss.item():.5f} | Data: {data_loss.item():.5f} | Phys: {eik_loss.item():.5f}")
        wandb.log({
            "total_loss": total_loss.item(),
            "eik_loss": eik_loss.item(),
            "data_loss": data_loss.item(),
            "pde_loss": pde_loss().item()
        })

print("starting phase 2: lbfgs fine tuning")
# L-BFGS needs a "closure" function that re-evaluates the loss
def closure():
    optimizer2.zero_grad()
    
    # For L-BFGS, we often use full-batch or large-batch
    # Here we just grab one batch from the iterator for simplicity, 
    # but ideally, you'd iterate over the whole dataset or a large chunk.
    try:
        inputs, targets = next(iter(tt_train_loader))
    except StopIteration:
        # Restart loader if exhausted
        inputs, targets = next(iter(DataLoader(tt_train, batch_size=len(tt_train))))
        
    inputs, targets = inputs.to(device), targets.to(device)
    
    rx, rz = inputs[:, 0:1], inputs[:, 1:2]
    sx, sz = inputs[:, 2:3], inputs[:, 3:4]
    
    pde_x = torch.rand(inputs.shape[0], 1, requires_grad=True, device=device)
    pde_z = torch.rand(inputs.shape[0], 1, requires_grad=True, device=device)
    
    tt_pred = tt_model(rx, rz, sx, sz)
    loss_data = compute_data_loss(tt_pred, targets)
    loss_eik = eikonal_loss(tt_model, v_model, pde_x, pde_z, sx, sz)
    
    total_loss = loss_data + (0.1 * loss_eik)
    total_loss.backward()
    return total_loss

# Run L-BFGS Steps
for i in range(lbfgs_max_iter):
    loss = optimizer2.step(closure)
    if i % 10 == 0:
        print(f"L-BFGS Step {i} | Loss: {loss.item():.5f}")
        wandb.log({"lbfgs_loss": loss.item()})

# Save Models
torch.save(tt_model.state_dict(), "data/tt_model.pt")
torch.save(v_model.state_dict(), "data/v_model.pt")
print("Training Complete.")