# torch
import torch.nn as nn

# locals
from .backbones import VGG

class SENNParametrizer(nn.Module):
    """Class to reproduce Senn parametrizer architecture

    Args:
        n_concepts: number of concepts
        dataset: MNIST or CIFAR10. Defaults to "MNIST".
    
    Input:
        x: image (b, n_channels, h, w)
    
    Output:
        relevances: vector of concept relevances (b, n_concepts, n_classes)
    """
    def __init__(self, n_concepts, dataset = "MNIST"):
        super(SENNParametrizer, self).__init__()
        self.n_concepts = n_concepts
        if dataset == "MNIST":
            self._create_MNIST_parametrizer()
        elif dataset == "CIFAR10":
            self._create_CIFAR10_parametrizer()
        else:
            raise ValueError(f"Dataset {dataset} unknown")
    
    def forward(self, x):
        x = self.parametrizer(x)
        return x.view(-1, self.n_concepts, 10)
    
    def _create_MNIST_parametrizer(self):
        self.parametrizer = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(500, self.n_concepts*10),
            nn.Sigmoid()
        )

    def _create_CIFAR10_parametrizer(self):
        vgg = VGG("VGG8", 3)
        self.parametrizer = nn.Sequential(
            vgg,
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, self.n_concepts*10),
            nn.Sigmoid()
        )

class ConvParametrizer(nn.Module):
    def __init__(self, backbone, projector, n_concepts, n_classes):
        super(ConvParametrizer, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.backbone = backbone
        if projector == True:
            self.projector = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.backbone.out_dim, self.backbone.out_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.backbone.out_dim//2, self.n_concepts*self.n_classes),
                nn.Sigmoid()
            )
        elif isinstance(projector, nn.Module):
            self.projector = projector
        else:
            self.projector = nn.Sequential()

    def forward(self, x):
        x = self.backbone(x)
        x = self.projector(x)
        return x.view(-1, self.n_concepts, self.n_classes)

if __name__ == "__main__":
    pass