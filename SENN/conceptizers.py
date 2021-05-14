# torch
import torch
import torch.nn as nn

# locals
from .utils import OneHotEncode
from .encoders import SENNEncoder, StyleEncoder, VAEEncoder
from .decoders import SENNDecoder

class SENNConceptizer(nn.Module):
    """Class to reproduce Senn conceptizer architecture

    Args:
        n_concepts: number of concepts
        dataset: MNIST or CIFAR10. Defaults to "MNIST".
    
    Inout:
        x: image (b, n_channels, h, w)
    
    Output:
        z: vector of concepts (b, n_concepts)
        x_tilde: reconstructed image (b, n_channels, h, w)
    """
    def __init__(self, n_concepts, dataset = "MNIST"):
        super(SENNConceptizer, self).__init__()
        self.n_concepts = n_concepts
        self.n_channels = 3 if dataset == "CIFAR10" else 1
        self.encoder = SENNEncoder(self.n_concepts, self.n_channels)
        self.decoder = SENNDecoder(self.n_concepts, self.n_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_tilde = self.decoder(z)
        return z, x_tilde.view_as(x)
class VAEConceptizer(nn.Module):
    """Conzeptizer for vaesenn

    Args:
        n_concepts: number of concepts
        n_styles: number of styles
        n_classes: number of classes for classification task. Defaults to 10.
        dataset: dataset. Defaults to MNIST.
    Returns
        vaesenn conceptizer module
    """
    def __init__(self, n_concepts, n_styles, n_classes = 10, dataset = "MNIST"):
        super(VAEConceptizer, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_styles = n_styles
        self.n_channels = 3 if dataset == "CIFAR10" else 1
        self.encoder_concepts = VAEEncoder(self.n_concepts, self.n_channels)
        self.decoder_concepts = SENNDecoder(self.n_concepts+self.n_styles, self.n_channels)
        self.encoder_styles = VAEEncoder(self.n_styles, self.n_channels)
        self.decoder_styles = SENNDecoder(self.n_classes+self.n_styles, self.n_channels)
    
    def forward_styles(self, x, targets):
        one_hot = OneHotEncode(self.n_classes)(targets)
        mean, log_var = self.encoder_styles(x)
        if self.training:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        x_decoded = self.decoder_styles(torch.cat([z, one_hot], axis=-1))
        return z, mean, log_var, x_decoded.view_as(x)
    
    def forward(self, x):
        mean, log_var = self.encoder_concepts(x)
        mean_styles, _ = self.encoder_styles(x)
        if self.training:
            std = torch.exp(0.5 * log_var)
            epsilon = torch.randn_like(std)
            z = mean + std * epsilon
        else:
            z = mean
        x_decoded = self.decoder_concepts(torch.cat([z, mean_styles], axis=-1))
        return z, mean, log_var, x_decoded.view_as(x)
class InvarConceptizer(SENNConceptizer):
    """Conceptizer for invarsenn

    Args:
        n_concepts: number of concepts
        n_e2: number of noise variables
        dataset: datset
        dropout_rate: dropout rate
    Returns:
        conceptizer module for invarseen
    """
    def __init__(self, n_concepts, n_e2, dataset, dropout_rate = 0.5):
        super(InvarConceptizer, self).__init__(n_concepts + n_e2, dataset)
        self.n_e2 = n_e2
        self.noise = nn.Dropout(p=dropout_rate)
        self.fc_e1 = nn.Linear(n_concepts+n_e2, n_concepts)
        self.fc_e2 = nn.Linear(n_concepts+n_e2, n_e2)
    
    def forward(self, x):
        out = self.encoder(x)
        concepts = self.fc_e1(out)
        e2 = self.fc_e2(out)
        concepts_noisy = self.noise(concepts)
        reconstructed_x = self.decoder(torch.cat((concepts_noisy, e2), axis=-1))
        return concepts, e2, reconstructed_x.view_as(x)

if __name__ == "__main__":
    pass
    
