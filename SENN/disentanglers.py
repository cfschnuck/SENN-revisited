import torch
import torch.nn as nn

class Disentangler(nn.Module):
    """Disentangler module for invarsenn

    Args:
        in_dim: dimension of input
        out_dim: dimension of output
    Returns:
        
    """
    def __init__(self, in_dim, out_dim):
        super(Disentangler, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            # nn.LeakyRelu()
            # nn.Linear(64, self.out_dim)
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
