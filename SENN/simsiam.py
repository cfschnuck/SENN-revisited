import torch.nn as nn
import torch
import torch.nn.functional as F 

from .losses import kl_div


## Inspired by https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py

def D(p, z, version='simplified'): 
    """Negative cosine distance

    Args:
        p: vector
        z: vector. will be detached
        version:. Defaults to 'simplified'.

    Returns:
        negative cosine distance
    """
    if version == 'original':
        z = z.detach()
        p = F.normalize(p, dim=1) 
        z = F.normalize(z, dim=1) 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 
class GaussTripletSimSiam(nn.Module):
    """Conceptizer for VSiamSenn

    Args:
        z_dim: number of concepts
        n_channels: number of input channels
    
    Output:
        concepts
    """
    def __init__(self, z_dim, n_channels):
        super().__init__()
        self.z_dim = z_dim
        self.n_channels = n_channels

        self.backbone = nn.Sequential(
            nn.Conv2d(self.n_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.mean = nn.Linear(500, self.z_dim)
        self.log_var = nn.Linear(500, self.z_dim)

        self.predictor_mean = prediction_MLP(in_dim=self.z_dim, out_dim=self.z_dim)
    
    def forward_training(self, x1, x2, x3):
        f, h, m, l = self.backbone, self.predictor_mean, self.mean, self.log_var
        z1, z2, z3 = f(x1), f(x2), f(x3)
        m1, m2, m3 = m(z1), m(z2), m(z3)
        l1, l2, l3 = l(z1), l(z2), l(z3)
        s1 = m1 + torch.exp(0.5 * l1) * torch.randn_like(m1)
        s2 = m2 + torch.exp(0.5 * l2) * torch.randn_like(m1)
        s3 = m3 + torch.exp(0.5 * l3) * torch.randn_like(m1)
        pm1, pm2, pm3 = h(m1), h(m2), h(m3)
        L1, L2, KL = D(pm1, s2) / 2 + D(pm2, s1) / 2, torch.abs(D(pm1, s3)/2) + torch.abs(D(pm3, s1)/2), kl_div(m1, l1)/3 + kl_div(m2, l2)/3 + kl_div(m3, l3)/3 
        return m1, (L1, L2, KL)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.mean(x)
        return x


