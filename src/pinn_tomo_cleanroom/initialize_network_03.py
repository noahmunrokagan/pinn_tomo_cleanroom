import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

# Initialize WandB
wandb.init(project="ant-pinn-inversion")

class AdaptiveELU(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        # Initialize the slope parameter 'a' to 1.0
        # This becomes a learnable parameter for the optimizer
        self.a = nn.Parameter(torch.ones(1)) 
        self.elu = nn.ELU()

    def forward(self, x):
        # The 'adaptive' part: scale the input by 'a' before applying ELU
        return self.elu(self.a * x)

class TTPinn(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_features = 4 # [rx, rz, sx, sz]

        # Build 10 layers
        for _ in range(10):
            layers.append(nn.Linear(in_features, 20))
            layers.append(AdaptiveELU(20)) # Use our custom layer
            in_features = 20
            
        layers.append(nn.Linear(20, 1))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
    
    def forward(self, rx, rz, sx, sz):

        # T0 = distance / 2000
        dist = torch.sqrt((rx - sx)**2 + (rz - sz)**2 + 1e-6)
        t0 = dist / 2000.0
        
        inputs = torch.cat([rx, rz, sx, sz], dim=1)
        tau = self.net(inputs)
        
        # Predicted T = T0 * tau
        return t0 * tau


class VPinn(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_features = 2

        # Build 10 layers
        for _ in range(10):
            layers.append(nn.Linear(in_features, 10))
            layers.append(AdaptiveELU(10)) # Use our custom layer
            in_features = 10
            
        layers.append(nn.Linear(10, 1))
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)
        
    
    def forward(self, rx, rz):
        inputs = torch.cat([rx, rz], dim=1)
        # Scale Sigmoid (0 to 1) to (1500 to 5000) m/s
        v_min, v_max = 1500.0, 5000.0
        v = v_min + (v_max - v_min) * self.net(inputs)
        return v
    
def compute_data_loss(tt_model, batch):
    """
    batch: a dictionary or list containing (sx, sz, rx, rz, t_obs)
    """
    # 1. Unpack and Normalize (Assume model_max_dist = 4000.0)
    sx, sz = batch['sx'] / 4000.0, batch['sz'] / 4000.0
    rx, rz = batch['rx'] / 4000.0, batch['rz'] / 4000.0
    t_obs = batch['t_obs']

    # 2. Predict Traveltime
    t_pred = tt_model(rx, rz, sx, sz)

    # 3. Mean Squared Error
    return torch.mean((t_pred - t_obs)**2)

def boundary_condition_loss(tt_model, v_model, collocation_points, source_locations):
    """
    collocation_points: Tensor [N, 2] representing (x, z)
    source_locations: Tensor [N, 2] representing (sx, sz)
    """
    # 1. Enable gradient tracking for coordinates
    coords = collocation_points.clone().detach().requires_grad_(True)
    x = coords[:, 0:1]
    z = coords[:, 1:2]
    
    # Sources (usually fixed for the batch)
    sx = source_locations[:, 0:1]
    sz = source_locations[:, 1:2]

    # 2. Get Predicted Traveltime
    # We treat the collocation point as a 'virtual receiver'
    T = tt_model(x, z, sx, sz)

    # 3. Calculate Gradients dT/dx and dT/dz
    grad_outputs = torch.ones_like(T)
    gradients = torch.autograd.grad(
        outputs=T,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True, # Critical for allowing backprop through the gradient
        retain_graph=True
    )[0]

    dT_dx = gradients[:, 0:1]
    dT_dz = gradients[:, 1:2]

    # 4. Get Predicted Velocity at these points
    V_pred = v_model(x, z)

    # 5. Eikonal Equation: (dT/dx)^2 + (dT/dz)^2 = 1 / V^2
    lhs = dT_dx**2 + dT_dz**2
    rhs = 1.0 / (V_pred**2 + 1e-6) # add epsilon to avoid div by zero

    return torch.mean((lhs - rhs)**2)

def eikonal_loss(model, x, y):
    """
    The Physics Loss: |grad(T)| - 1/v = 0
    """
    x.requires_grad = True
    y.requires_grad = True
    
    T_pred, V_pred = model(x, y)
    grad_outputs = torch.ones_like(T_pred)
    
    # 1. Compute gradients of T with respect to x and y
    dT_dx = torch.autograd.grad(outputs=T_pred, 
                                inputs=x, 
                                grad_outputs=grad_outputs, 
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True
                                )[0]
    dT_dy = torch.autograd.grad(outputs=T_pred, 
                                inputs=y,
                                grad_outputs=grad_outputs,
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True
                                )[0]
    
    # 2. Compute the Eikonal Residual
    # Equation: (dT/dx)^2 + (dT/dy)^2 = 1 / V^2
    # So, Residual = (dT_dx**2 + dT_dy**2) - (1 / V_pred**2)
    
    eikonal_residual = (dT_dx ** 2 + dT_dy ** 2) - (1 / V_pred ** 2)
    
    return torch.mean(eikonal_residual**2)