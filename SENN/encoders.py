# torch
import torch.nn as nn

class SENNEncoder(nn.Module):
    """Encoder (SENN architecture)

    Args:
        n_concepts: number of concepts
        n_channels: number of output channels
    
    Input:
        x: image (b, n_channels, h, w)
    
    Output:
        concept vector (b, n_concepts)

    """
    def __init__(self, n_concepts, n_channels):
        super(SENNEncoder, self).__init__()
        self.n_concepts = n_concepts
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(500, self.n_concepts),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class StyleEncoder(nn.Module):
    def __init__(self, n_styles, n_channels):
        super(StyleEncoder, self).__init__()
        self.n_styles = n_styles
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(500, self.n_styles)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class VAEEncoder(nn.Module):
    def __init__(self, z_dim, n_channels):
        super(VAEEncoder, self).__init__()
        self.z_dim = z_dim
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.mean_layer = nn.Linear(500, self.z_dim)
        self.log_var_layer = nn.Linear(500, self.z_dim)
    
    def forward(self, x):
        encoded = self.encoder(x)
        mean = self.mean_layer(encoded)
        log_var = self.log_var_layer(encoded)
        return mean, log_var