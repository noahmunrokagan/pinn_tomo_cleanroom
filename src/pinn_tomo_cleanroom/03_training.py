import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

# Initialize WandB
wandb.init(project="ant-pinn-inversion")

class TTPinn(nn.Module):
    def __init__(self):
        super.__init__()

        self.net = nn.Sequential(
            nn.Linear(4,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20,20),
            nn.ELU(),
            nn.Linear(20, 1),
            nn.Softplus() # ensures it's positive
        )
    
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

        self.net = nn.Sequential(
            nn.Linear(2,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10,10),
            nn.ELU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rx, rz):
        inputs = torch.cat([rx, rz], dim=1)
        # Scale Sigmoid (0 to 1) to (1500 to 5000) m/s
        v_min, v_max = 1500.0, 5000.0
        v = v_min + (v_max - v_min) * self.net(inputs)
        return v
    

    
