import torch
from torch import nn
from torch.nn import functional as F
from settings import *
from .blocks import ResBlock


class Encoder(nn.Module):

    def __init__(self, feature_dim, n_resid):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, feature_dim//2, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(feature_dim//2),
            nn.Tanh(),

            nn.Conv2d(feature_dim//2, feature_dim, (4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.Tanh(),
            
            nn.Conv2d(feature_dim, feature_dim, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(feature_dim),
            
            ResBlock(feature_dim, n_resid),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder(x)
        return features
