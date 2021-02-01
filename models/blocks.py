import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):

    def __init__(self, feature_dim, n_resid):
        super(ResBlock, self).__init__()
        
        layers = []
        for i in range(n_resid):
            layers.append(
                nn.Sequential(
                    nn.Tanh(),

                    nn.Conv2d(feature_dim, feature_dim//2, (3, 3), stride=(1, 1), padding=1),
                    nn.BatchNorm2d(feature_dim//2),
                    nn.Tanh(),

                    nn.Conv2d(feature_dim//2, feature_dim, (3, 3), stride=(1, 1), padding=1),
                    nn.BatchNorm2d(feature_dim),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x

        return x


class ResTransposeBlock(nn.Module):

    def __init__(self, feature_dim, n_resid):
        super(ResTransposeBlock, self).__init__()
        
        layers = []
        for i in range(n_resid):
            layers.append(
                nn.Sequential(
                    nn.Tanh(),

                    nn.ConvTranspose2d(feature_dim, feature_dim//2, (3, 3), stride=(1, 1), padding=1, output_padding=0),
                    nn.BatchNorm2d(feature_dim//2),
                    nn.Tanh(),

                    nn.ConvTranspose2d(feature_dim//2, feature_dim, (3, 3), stride=(1, 1), padding=1, output_padding=0),
                    nn.BatchNorm2d(feature_dim),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x

        return x

