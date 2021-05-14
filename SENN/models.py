# torch
import torch.nn as nn

class Senn(nn.Module):
    """Self-Explaining Neural Network (SENN)

    Args:
        conceptizer: conceptizer architecture
        parametrizer: parametrizer architecture
        aggregator: aggregator architecture
    
    Inputs:
        x: image (b, n_channels, h, w)

    Returns:
        pred: vector of class probabilities (b, n_classes)
        concepts: concept vector (b, n_concepts)
        relevances: vector of concept relevances (b, n_concepts, n_classes)
        x_reconstructed: reconstructed image (b, n_channels, h, w)
    """
    def __init__(self, conceptizer, parametrizer, aggregator):
        super(Senn, self).__init__()
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
    
    def forward(self, x):
        concepts, x_reconstructed = self.conceptizer(x)
        relevances = self.parametrizer(x)
        pred = self.aggregator(concepts, relevances)
        return pred, (concepts, relevances), x_reconstructed
class VAESenn(nn.Module):
    """VaeSENN

    Args:
        conceptizer: conceptizer architecture
        parametrizer: parametrizer architecture
        aggregator: aggregator architecture
    
    Inputs:
        x: image (b, n_channels, h, w)

    Returns:
        pred: vector of class probabilities (b, n_classes)
        concepts: concept vector (b, n_concepts)
        relevances: vector of concept relevances (b, n_concepts, n_classes)
        x_reconstructed: reconstructed image (b, n_channels, h, w)
        log_var: log variance of concepts posteriors
    """
    def __init__(self, conceptizer, parametrizer, aggregator):
        super(VAESenn, self).__init__()
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
    
    def forward(self, x):
        concepts, mean, log_var, x_recon = self.conceptizer(x)
        relevances = self.parametrizer(x)
        pred = self.aggregator(concepts, relevances)
        return pred, (concepts, relevances), x_recon, log_var, mean
class GaussSiamSenn(nn.Module):
    """VSiamSENN 

        !! Naming not consistent with report

    Args:
        conceptizer: conceptizer architecture
        parametrizer: parametrizer architecture
        aggregator: aggregator architecture
    
    Inputs:
        x: image (b, n_channels, h, w)

    Returns:
        pred: vector of class probabilities (b, n_classes)
        concepts: concept vector (b, n_concepts)
        relevances: vector of concept relevances (b, n_concepts, n_classes)
    """
    def __init__(self, conceptizer, parametrizer, aggregator):
        super(GaussSiamSenn, self).__init__()
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
    
    def forward(self, x, x_eq = None, x_diff = None):
        if self.conceptizer.training:
            concepts, (L1, L2, KL) = self.conceptizer.forward_training(x, x_eq, x_diff)
        else:
            concepts = self.conceptizer(x)
        relevances = self.parametrizer(x)
        pred = self.aggregator(concepts, relevances)
        if self.conceptizer.training:
            return pred, (concepts, relevances), (L1, L2, KL)
        else:
            return pred, (concepts, relevances)

class InvarSennM(nn.Module):
    """InvarSENN 

    Args:
        m1: m1 architecture
        m2: m2 architecture

    Inputs:
        x: image (b, n_channels, h, w)

    Returns:
        pred: vector of class probabilities (b, n_classes)
        e1: concept vector (b, n_concepts)
        relevances: vector of concept relevances (b, n_concepts, n_classes)
        e2: noise vector (b, n_concepts)
        x_reconstructed: reconstructed input image
        e1_reconstructed: reconstructed concept vector
        e2_reconstructed: reconstructed noise vector
    """
    def __init__(self, m1, m2):
        super(InvarSennM, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        pred, (e1, relevances), e2, x_reconstructed = self.m1(x)
        e1_reconstructed, e2_reconstructed = self.m2(e1, e2)
        return pred, (e1, relevances), e2, x_reconstructed, (e1_reconstructed, e2_reconstructed)

