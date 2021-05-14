# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAggregator(nn.Module):
    """ Linear aggregator
        
    Aggregates a set of concept representations and their
    relevances and generates a prediction probability output from them.
        
    Inputs:
        concepts: concept vector (b, n_concepts)
        relevances: vector of concept relevances (b, n_concepts, n_classes)
    
    Returns:
        vector of class probabilities (b, n_classes)
    """
    def __init__(self):
        super(BaseAggregator, self).__init__()

    def forward(self, concepts, relevances):
        aggregated = torch.bmm(relevances.permute(0, 2, 1), concepts.unsqueeze(-1)).squeeze(-1)
        return F.log_softmax(aggregated, dim=1)
