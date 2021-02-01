import torch
from torch import nn
from torch.nn import functional as F
from settings import *
from .blocks import ResTransposeBlock


class Decoder(nn.Module):

    def __init__(self, feature_dim, n_resid):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim, (3, 3), stride=(1, 1), padding=1, output_padding=0),
            nn.BatchNorm2d(feature_dim),

            ResTransposeBlock(feature_dim, n_resid),
            nn.Tanh(),

            nn.ConvTranspose2d(feature_dim, feature_dim//2, (4, 4), stride=(2, 2), padding=1, output_padding=0),
            nn.BatchNorm2d(feature_dim//2),
            nn.Tanh(),

            nn.ConvTranspose2d(feature_dim//2, CHANNEL, (4, 4), stride=(2, 2), padding=1, output_padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        reconst = self.decoder(x)
        return reconst
