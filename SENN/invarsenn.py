## torch
import torch.nn as nn


class InvarSennM1(nn.Module):
    def __init__(self, conceptizer, parametrizer, aggregator):
        super(InvarSennM1, self).__init__()
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator

    def forward(self, x):
        e1, e2, x_reconstructed = self.conceptizer(x)
        relevances = self.parametrizer(x) # TODO soll hier x oder concepts benutzt werden?
        pred = self.aggregator(e1, relevances)
        return pred, (e1, relevances), e2, x_reconstructed

class InvarSennM2(nn.Module):
    def __init__(self, disentangler1, disentangler2):
        super(InvarSennM2, self).__init__()
        self.disentangler1 = disentangler1 # predict e1 from e2
        self.disentangler2 = disentangler2 # predict e2 from e1

    def forward(self, e1, e2):
        e2_reconstructed = self.disentangler2(e1)
        e1_reconstructed = self.disentangler1(e2)
        return e1_reconstructed, e2_reconstructed
