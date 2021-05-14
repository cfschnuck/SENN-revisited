import torch.nn as nn
import numpy as np

class VGG(nn.Module):
    """VGG net

    Args:
        vgg_name: vgg architecture
        n_channels: number of input channels
        image_size: image size. Defaults to 32.
    
    Input:
        x: image (b, n_channels, h, w)
        
    """
    def __init__(self, vgg_name, n_channels, image_size=32):
        super(VGG, self).__init__()
        self.n_channels = n_channels
        self.image_size = image_size
        self.cfg = {
    'VGG8':  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }   
        self.features = self._make_layers(self.cfg[vgg_name])
        self._set_output_size(self.cfg[vgg_name])
        self.out_dim = int(self.out_width * self.out_height * 512)

    def forward(self, x):
        out = self.features(x)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.n_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.Dropout2d(p=0.3),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def _set_output_size(self, cfg):
        if isinstance(self.image_size, int):
            height, width = self.image_size, self.image_size
        elif isinstance(self.image_size, list) or isinstance(self.image_size, tuple):
            height, width = self.image_size[0], self.image_size[1]
        else:
            raise ValueError
        
        for x in cfg:
            if x == 'M':
                width = self._maxpool2d_output_size(width)
                height = self._maxpool2d_output_size(height)
            else:
                width = self._conv2d_output_size(width)
                height = self._conv2d_output_size(height)
        self.out_width = width
        self.out_height = height
                
    @staticmethod
    def _maxpool2d_output_size(input_size):
        return np.floor((input_size-2)/2+1)
    
    @staticmethod
    def _conv2d_output_size(input_size):
        return np.floor((input_size-1)+1)
