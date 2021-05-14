import numpy as np

# torch
import torch.nn as nn

## locals
from .utils import View

class SENNDecoder(nn.Module):
    """ Decoder (SENN architecture)

    Args:
        n_concepts: number of concepts
        n_channels: number of output channels

    Input:
        z: concept vector (b, n_concepts)

    Returns:
        reconstructed image (b, n_channels, h, w)
    """
    def __init__(self, n_concepts, n_channels):
        super(SENNDecoder, self).__init__()
        self.n_concepts = n_concepts
        self.n_channels = n_channels
        self.dout = int(np.sqrt(32**2)//4 - 3*(5-1)//4) ## For kernel = 5 in both, and maxppol stride = 2 in both
        self.decoder = nn.Sequential(
            nn.Linear(self.n_concepts, int(20*self.dout**2)),
            View(-1, 20, self.dout, self.dout),
            nn.ConvTranspose2d(20, 16, 5, stride = 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 5),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, self.n_channels, 2, stride=2, padding=1)
        )
    
    def forward(self, z):
        x = self.decoder(z)
        return x
