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
        super.__init__()
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
        super.__init__()
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
    

    
